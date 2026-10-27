#!/usr/bin/env python3
"""Strict Qwen3.5 LoRA smoke test for Shrikes/repair-sbn.

This is deliberately a smoke test, not a train/dev/test experiment.  It takes a
small, deterministic, variant-balanced subset from the dataset's sole `train`
split and runs 20--50 optimizer steps before saving and reloading the adapter.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET_ID = "Shrikes/repair-sbn"

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "2B": {
        "model_id": "Qwen/Qwen3.5-2B",
        "adapter_modules": 150,
        "trainable_parameters": 15_630_336,
        "layers": 24,
        "full_attention_layers": 6,
        "linear_attention_layers": 18,
    },
    "9B": {
        "model_id": "Qwen/Qwen3.5-9B",
        "adapter_modules": 200,
        "trainable_parameters": 40_108_032,
        "layers": 32,
        "full_attention_layers": 8,
        "linear_attention_layers": 24,
    },
    "27B": {
        "model_id": "Qwen/Qwen3.5-27B",
        "adapter_modules": 400,
        "trainable_parameters": 108_789_760,
        "layers": 64,
        "full_attention_layers": 16,
        "linear_attention_layers": 48,
    },
}

TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
)

EXPECTED_VARIANTS = {
    "pmb_fluent": 9_486,
    "rsr_base": 3_489,
    "rsr_editing_expression": 1_047,
}
REQUIRED_DATA_FIELDS = {"text", "sbn_clean", "sbn_repair", "variant", "doc_id"}
EXPECTED_VERSIONS = {
    "torch": "2.14.1",
    "transformers": "5.20.2",
    "peft": "0.21.2",
    "datasets": "5.1.0",
}
IGNORE_INDEX = -100


@dataclass(frozen=True)
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]
    prompt_length: int
    full_length: int
    variant: str
    doc_id: str


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("a", encoding="utf-8")

    def log(self, event: str, **values: Any) -> None:
        record = {"event": event, "unix_time": time.time(), **values}
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        print(line, flush=True)
        self.handle.write(line + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", required=True, choices=tuple(MODEL_SPECS))
    parser.add_argument(
        "--condition",
        required=True,
        choices=("clean", "repair"),
        help="clean supervises sbn_clean; repair supervises sbn_repair.",
    )
    parser.add_argument("--optimizer-steps", type=int, default=30)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--examples-per-variant", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument(
        "--model-revision",
        required=True,
        help="Pinned Hugging Face model commit.",
    )
    parser.add_argument(
        "--dataset-cache",
        type=Path,
        required=True,
        help="Datasets cache prepared before the GPU job.",
    )
    parser.add_argument(
        "--dataset-revision",
        required=True,
        help="Pinned Hugging Face dataset commit.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generation-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--strict-versions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the four package versions agreed for this smoke test.",
    )
    args = parser.parse_args()

    assert 20 <= args.optimizer_steps <= 50, "--optimizer-steps must be in [20, 50]"
    assert args.gradient_accumulation_steps >= 1
    assert args.micro_batch_size >= 1
    assert args.examples_per_variant >= 1
    assert args.max_length >= 32
    assert args.learning_rate > 0
    assert args.max_grad_norm > 0
    if not args.device.startswith("cuda"):
        raise AssertionError("This bf16 GPU smoke test requires a CUDA device")
    return args


def assert_package_versions(strict: bool) -> dict[str, str]:
    found: dict[str, str] = {}
    for package, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(package)
        found[package] = actual
        if strict:
            comparable = actual.split("+", 1)[0] if package == "torch" else actual
            assert comparable == expected, (
                f"{package}: expected {expected}, found {actual}"
            )
    return found


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_memory() -> dict[str, float]:
    gib = 1024.0**3
    return {
        "allocated_gib": torch.cuda.memory_allocated() / gib,
        "reserved_gib": torch.cuda.memory_reserved() / gib,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / gib,
    }


def sync_cuda() -> None:
    torch.cuda.synchronize()


def stable_row_key(row: dict[str, Any], seed: int) -> str:
    material = "\0".join(
        (str(seed), str(row["variant"]), str(row["doc_id"]), str(row["text"]))
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def load_balanced_smoke_subset(
    examples_per_variant: int,
    seed: int,
    dataset_cache: Path | None,
    dataset_revision: str,
) -> tuple[Dataset, list[dict[str, Any]]]:
    kwargs: dict[str, Any] = {"revision": dataset_revision}
    if dataset_cache is not None:
        kwargs["cache_dir"] = str(dataset_cache)
    dataset = load_dataset(DATASET_ID, **kwargs)
    assert isinstance(dataset, DatasetDict)
    assert list(dataset.keys()) == ["train"], (
        "Shrikes/repair-sbn must expose exactly its real source split ['train']; "
        f"found {list(dataset.keys())}. This script does not manufacture train/dev/test splits."
    )
    source_train = dataset["train"]
    assert REQUIRED_DATA_FIELDS <= set(source_train.column_names), (
        REQUIRED_DATA_FIELDS - set(source_train.column_names)
    )
    counts = Counter(source_train["variant"])
    assert counts == Counter(EXPECTED_VARIANTS), (
        f"dataset variant counts changed: expected {EXPECTED_VARIANTS}, found {dict(counts)}"
    )

    buckets: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for index, row in enumerate(source_train):
        buckets[row["variant"]].append((stable_row_key(row, seed), index))

    selected_indices: list[int] = []
    for variant in EXPECTED_VARIANTS:
        ranked = sorted(buckets[variant])
        assert len(ranked) >= examples_per_variant
        selected_indices.extend(index for _, index in ranked[:examples_per_variant])

    # A final stable ordering makes the subset independent of Dataset internals.
    selected_indices.sort(key=lambda i: stable_row_key(source_train[i], seed + 1))
    subset = source_train.select(selected_indices)
    subset_counts = Counter(subset["variant"])
    assert subset_counts == Counter(
        {variant: examples_per_variant for variant in EXPECTED_VARIANTS}
    )
    assert len(subset) == examples_per_variant * len(EXPECTED_VARIANTS)

    manifest = [
        {
            "source_split": "train",
            "source_index": index,
            "doc_id": source_train[index]["doc_id"],
            "variant": source_train[index]["variant"],
            "row_key": stable_row_key(source_train[index], seed),
        }
        for index in selected_indices
    ]
    return subset, manifest


def user_prompt(text: str, condition: str) -> str:
    target_description = {
        "clean": "the fluent/clean interpretation",
        "repair": "the repair-aware interpretation",
    }[condition]
    return (
        "Convert the utterance below to a single-line SBN representation for "
        f"{target_description}. Return only the SBN.\n\nUtterance: {text}"
    )


def render_chat_template(
    tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    # Qwen3.5 understands enable_thinking; using False keeps this supervised
    # formatting deterministic and avoids hidden reasoning tokens.
    try:
        rendered = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    assert isinstance(rendered, str), type(rendered)
    return rendered


def encode_chat_text(tokenizer: Any, rendered: str) -> list[int]:
    ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        assert len(ids) == 1
        ids = ids[0]
    return list(ids)


def apply_chat_template(
    tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool
) -> list[int]:
    rendered = render_chat_template(tokenizer, messages, add_generation_prompt)
    return encode_chat_text(tokenizer, rendered)


def tokenize_row(
    row: dict[str, Any], tokenizer: Any, condition: str, max_length: int
) -> TokenizedExample:
    target_field = {"clean": "sbn_clean", "repair": "sbn_repair"}[condition]
    target = row[target_field]
    assert isinstance(target, str) and target.strip()
    prompt_messages = [{"role": "user", "content": user_prompt(row["text"], condition)}]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": target.strip()},
    ]
    prompt_text = render_chat_template(
        tokenizer, prompt_messages, add_generation_prompt=True
    )
    full_text = render_chat_template(
        tokenizer, full_messages, add_generation_prompt=False
    )
    assert target.strip() in full_text, (
        f"chat template omitted the assistant target for doc_id={row['doc_id']!r}"
    )
    assert full_text.startswith(prompt_text), (
        "chat template did not preserve the rendered prompt as an exact prefix for "
        f"doc_id={row['doc_id']!r}; refusing an ambiguous prompt-loss mask"
    )
    prompt_ids = encode_chat_text(tokenizer, prompt_text)
    full_ids = encode_chat_text(tokenizer, full_text)
    assert full_ids[: len(prompt_ids)] == prompt_ids, (
        "tokenization did not preserve the rendered prompt as an exact prefix for "
        f"doc_id={row['doc_id']!r}; refusing an ambiguous prompt-loss mask"
    )
    assert len(prompt_ids) < max_length, (
        f"prompt for doc_id={row['doc_id']!r} has {len(prompt_ids)} tokens, "
        f"exceeding max_length={max_length}"
    )
    assert len(full_ids) > len(prompt_ids), (
        f"assistant target for doc_id={row['doc_id']!r} produced no tokens; "
        f"prompt_tokens={len(prompt_ids)}, full_tokens={len(full_ids)}"
    )
    input_ids = full_ids[:max_length]
    labels = [IGNORE_INDEX] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    assert len(input_ids) == len(labels)
    assert all(label == IGNORE_INDEX for label in labels[: len(prompt_ids)])
    assert any(label != IGNORE_INDEX for label in labels[len(prompt_ids) :]), (
        f"target for doc_id={row['doc_id']!r} was completely truncated; "
        f"prompt_tokens={len(prompt_ids)}, full_tokens={len(full_ids)}, "
        f"max_length={max_length}"
    )
    return TokenizedExample(
        input_ids=input_ids,
        labels=labels,
        prompt_length=len(prompt_ids),
        full_length=len(full_ids),
        variant=row["variant"],
        doc_id=row["doc_id"],
    )


def collate_examples(
    examples: list[TokenizedExample], pad_token_id: int
) -> dict[str, torch.Tensor]:
    max_len = max(len(example.input_ids) for example in examples)
    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    attention_mask: list[list[int]] = []
    for example in examples:
        padding = max_len - len(example.input_ids)
        input_ids.append(example.input_ids + [pad_token_id] * padding)
        labels.append(example.labels + [IGNORE_INDEX] * padding)
        attention_mask.append([1] * len(example.input_ids) + [0] * padding)
    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
    assert torch.all(batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX)
    assert torch.all(
        (batch["labels"] == IGNORE_INDEX) | (batch["attention_mask"] == 1)
    )
    return batch


def forever(loader: Iterable[dict[str, torch.Tensor]]) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def text_config(model: torch.nn.Module) -> Any:
    config = model.config
    return getattr(config, "text_config", config)


def assert_base_architecture(model: torch.nn.Module, spec: dict[str, Any]) -> dict[str, int]:
    config = text_config(model)
    layer_types = list(config.layer_types)
    observed = Counter(layer_types)
    assert int(config.num_hidden_layers) == spec["layers"]
    assert len(layer_types) == spec["layers"]
    assert observed["full_attention"] == spec["full_attention_layers"]
    assert observed["linear_attention"] == spec["linear_attention_layers"]
    assert set(observed) == {"full_attention", "linear_attention"}

    suffix_counts: Counter[str] = Counter()
    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix in TARGET_MODULES and isinstance(module, torch.nn.Linear):
            suffix_counts[suffix] += 1
    missing = set(TARGET_MODULES) - set(suffix_counts)
    assert not missing, f"missing LoRA target module types: {sorted(missing)}"
    assert sum(suffix_counts.values()) == spec["adapter_modules"], (
        f"base target count: expected {spec['adapter_modules']}, "
        f"found {sum(suffix_counts.values())}; detail={dict(suffix_counts)}"
    )
    return dict(sorted(suffix_counts.items()))


def assert_lora_state(model: torch.nn.Module, spec: dict[str, Any]) -> dict[str, Any]:
    lora_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, LoraLayer)]
    trainable_named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen_named = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
    trainable_count = sum(parameter.numel() for _, parameter in trainable_named)
    total_count = trainable_count + sum(parameter.numel() for _, parameter in frozen_named)

    assert len(lora_modules) == spec["adapter_modules"], (
        f"LoRA module count: expected {spec['adapter_modules']}, found {len(lora_modules)}"
    )
    assert trainable_count == spec["trainable_parameters"], (
        f"trainable parameters: expected {spec['trainable_parameters']:,}, "
        f"found {trainable_count:,}"
    )
    assert trainable_named
    assert all("lora_" in name for name, _ in trainable_named), (
        "a non-LoRA parameter is unexpectedly trainable"
    )
    assert all(torch.isfinite(parameter).all().item() for _, parameter in trainable_named)
    return {
        "lora_module_count": len(lora_modules),
        "trainable_parameters": trainable_count,
        "total_parameters": total_count,
        "trainable_fraction": trainable_count / total_count,
    }


def assert_finite_gradients(model: torch.nn.Module) -> tuple[float, int]:
    squared_norm = 0.0
    tensors = 0
    missing: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        assert torch.isfinite(parameter.grad).all().item(), f"non-finite gradient in {name}"
        squared_norm += float(parameter.grad.detach().float().norm(2).item()) ** 2
        tensors += 1
    assert not missing, f"trainable parameters without gradients: {missing[:10]}"
    norm = math.sqrt(squared_norm)
    assert math.isfinite(norm) and norm > 0.0, f"invalid global gradient norm: {norm}"
    return norm, tensors


def assert_finite_trainable_parameters(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert torch.isfinite(parameter).all().item(), f"non-finite parameter in {name}"


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def make_output_dir(args: argparse.Namespace) -> Path:
    path = args.output_dir
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_base_model(
    model_id: str,
    revision: str,
    cache_dir: Path,
    device: torch.device,
) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=str(cache_dir),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": str(device)},
        attn_implementation="sdpa",
    )


def main() -> None:
    args = parse_args()
    versions = assert_package_versions(args.strict_versions)
    assert torch.cuda.is_available(), "CUDA is unavailable"
    assert torch.cuda.is_bf16_supported(), "the selected CUDA device does not support bf16"
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    set_reproducibility(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    spec = MODEL_SPECS[args.model_size]
    output_dir = make_output_dir(args)
    logger = JsonlLogger(output_dir / "smoke_test.jsonl")
    started = time.perf_counter()
    try:
        logger.log(
            "run_start",
            argv=sys.argv,
            model_size=args.model_size,
            model_id=spec["model_id"],
            model_revision=args.model_revision,
            condition=args.condition,
            source_dataset=DATASET_ID,
            dataset_revision=args.dataset_revision,
            source_split="train",
            split_policy="deterministic balanced smoke subset only; not a formal train/dev/test split",
            versions=versions,
            device=str(device),
            gpu=torch.cuda.get_device_name(device),
            bf16=True,
        )

        subset, manifest = load_balanced_smoke_subset(
            args.examples_per_variant,
            args.seed,
            args.dataset_cache,
            args.dataset_revision,
        )
        (output_dir / "smoke_subset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.log(
            "dataset_ready",
            examples=len(subset),
            variant_counts=dict(Counter(subset["variant"])),
            manifest="smoke_subset_manifest.json",
        )

        tokenizer = AutoTokenizer.from_pretrained(
            spec["model_id"],
            revision=args.model_revision,
            cache_dir=str(args.model_cache),
            local_files_only=True,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            assert tokenizer.eos_token_id is not None
            tokenizer.pad_token = tokenizer.eos_token

        tokenized = [
            tokenize_row(row, tokenizer, args.condition, args.max_length) for row in subset
        ]
        supervised_tokens = sum(
            sum(label != IGNORE_INDEX for label in example.labels) for example in tokenized
        )
        masked_prompt_tokens = sum(example.prompt_length for example in tokenized)
        assert supervised_tokens > 0 and masked_prompt_tokens > 0
        logger.log(
            "tokenization_ready",
            max_sequence_length=max(len(example.input_ids) for example in tokenized),
            max_untruncated_sequence_length=max(
                example.full_length for example in tokenized
            ),
            truncated_examples=sum(
                example.full_length > args.max_length for example in tokenized
            ),
            supervised_target_tokens=supervised_tokens,
            masked_prompt_tokens=masked_prompt_tokens,
            loss_mask="prompt and padding labels are -100; only assistant SBN target tokens contribute",
        )

        generator = torch.Generator()
        generator.manual_seed(args.seed)
        loader = DataLoader(
            tokenized,
            batch_size=args.micro_batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=lambda batch: collate_examples(batch, tokenizer.pad_token_id),
            pin_memory=True,
            drop_last=False,
        )
        batches = forever(loader)

        torch.cuda.reset_peak_memory_stats(device)
        load_started = time.perf_counter()
        model = load_base_model(
            spec["model_id"], args.model_revision, args.model_cache, device
        )
        sync_cuda()
        assert next(model.parameters()).dtype == torch.bfloat16
        suffix_counts = assert_base_architecture(model, spec)
        logger.log(
            "base_model_loaded",
            seconds=time.perf_counter() - load_started,
            target_module_suffix_counts=suffix_counts,
            **cuda_memory(),
        )

        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(TARGET_MODULES),
        )
        model = get_peft_model(model, lora_config)
        lora_state = assert_lora_state(model, spec)
        logger.log(
            "lora_attached",
            target_modules=list(TARGET_MODULES),
            r=16,
            alpha=32,
            dropout=0.05,
            **lora_state,
            **cuda_memory(),
        )

        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=0.0,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        training_started = time.perf_counter()
        micro_step = 0
        for optimizer_step in range(1, args.optimizer_steps + 1):
            step_started = time.perf_counter()
            loss_sum = 0.0
            for _ in range(args.gradient_accumulation_steps):
                micro_step += 1
                batch = move_batch(next(batches), device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(**batch, use_cache=False)
                    loss = output.loss
                assert loss is not None and torch.isfinite(loss).item(), (
                    f"non-finite loss at micro-step {micro_step}: {loss}"
                )
                loss_sum += float(loss.detach())
                (loss / args.gradient_accumulation_steps).backward()

            raw_grad_norm, gradient_tensors = assert_finite_gradients(model)
            clipped_norm = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                args.max_grad_norm,
                error_if_nonfinite=True,
            )
            assert math.isfinite(float(clipped_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            assert_finite_trainable_parameters(model)
            sync_cuda()
            logger.log(
                "optimizer_step",
                optimizer_step=optimizer_step,
                micro_step=micro_step,
                mean_micro_loss=loss_sum / args.gradient_accumulation_steps,
                raw_gradient_norm=raw_grad_norm,
                clip_reported_norm=float(clipped_norm),
                gradient_tensors=gradient_tensors,
                step_seconds=time.perf_counter() - step_started,
                elapsed_training_seconds=time.perf_counter() - training_started,
                **cuda_memory(),
            )

        adapter_dir = output_dir / "adapter"
        save_started = time.perf_counter()
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        assert (adapter_dir / "adapter_config.json").is_file()
        assert (adapter_dir / "adapter_model.safetensors").is_file()
        logger.log(
            "adapter_saved",
            path=str(adapter_dir),
            seconds=time.perf_counter() - save_started,
            **cuda_memory(),
        )

        # Free the trained instance so the validation is a genuine base+adapter reload.
        del optimizer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        sync_cuda()
        logger.log("pre_reload_cleanup", **cuda_memory())

        reload_started = time.perf_counter()
        reloaded_base = load_base_model(
            spec["model_id"], args.model_revision, args.model_cache, device
        )
        reloaded = PeftModel.from_pretrained(
            reloaded_base,
            adapter_dir,
            is_trainable=False,
            local_files_only=True,
        )
        assert sum(isinstance(module, LoraLayer) for module in reloaded.modules()) == spec[
            "adapter_modules"
        ]
        reloaded.eval()
        sync_cuda()
        logger.log(
            "base_adapter_reloaded",
            seconds=time.perf_counter() - reload_started,
            **cuda_memory(),
        )

        validation_row = subset[0]
        validation_messages = [
            {
                "role": "user",
                "content": user_prompt(validation_row["text"], args.condition),
            }
        ]
        validation_ids = apply_chat_template(
            tokenizer, validation_messages, add_generation_prompt=True
        )
        generation_input = torch.tensor([validation_ids], dtype=torch.long, device=device)
        generation_mask = torch.ones_like(generation_input)
        generation_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated = reloaded.generate(
                input_ids=generation_input,
                attention_mask=generation_mask,
                max_new_tokens=args.generation_max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        sync_cuda()
        new_ids = generated[0, generation_input.shape[1] :]
        assert new_ids.numel() > 0, "reload generation produced no new tokens"
        generated_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        assert generated_text, "reload generation decoded to an empty string"
        logger.log(
            "reload_generation_verified",
            source_doc_id=validation_row["doc_id"],
            source_variant=validation_row["variant"],
            generated_tokens=int(new_ids.numel()),
            generated_text=generated_text,
            seconds=time.perf_counter() - generation_started,
            **cuda_memory(),
        )

        summary = {
            "status": "passed",
            "model_size": args.model_size,
            "model_id": spec["model_id"],
            "model_revision": args.model_revision,
            "condition": args.condition,
            "source_dataset": DATASET_ID,
            "dataset_revision": args.dataset_revision,
            "source_split": "train",
            "formal_split_created": False,
            "smoke_subset_examples": len(subset),
            "optimizer_steps": args.optimizer_steps,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "micro_batch_size": args.micro_batch_size,
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": list(TARGET_MODULES),
                **lora_state,
            },
            "adapter_dir": str(adapter_dir),
            "reload_generation": generated_text,
            "total_seconds": time.perf_counter() - started,
            "peak_cuda_memory": cuda_memory(),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        logger.log("run_passed", summary="summary.json", total_seconds=summary["total_seconds"])
    except Exception as error:
        logger.log(
            "run_failed",
            error_type=type(error).__name__,
            error=str(error),
            total_seconds=time.perf_counter() - started,
            **(cuda_memory() if torch.cuda.is_available() else {}),
        )
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
