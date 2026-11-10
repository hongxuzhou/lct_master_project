#!/usr/bin/env python3
"""Single-GPU Qwen3.5 development runner; formal runs await frozen configs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import socket
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import PROTOCOL_PATH, audit, condition_rows, load_rows, prompt_ids, read_protocol, select_train_subset


class Recorder:
    """Local records are authoritative; a dashboard failure cannot stop training."""

    def __init__(self, output, config, mode, wandb_config, external_run=None):
        self.handle = (output / "metrics.jsonl").open("a", buffering=1)
        self.run = external_run
        self.external = external_run is not None
        if self.external:
            self.run.config.update(config)
        elif mode != "disabled":
            try:
                import wandb
                self.run = wandb.init(
                    entity=wandb_config["entity"], project=wandb_config["project"],
                    name=config["run_id"], id=config["run_id"], resume="never",
                    group=f"{config['purpose']}-{config['model_size']}-s{config['seed']}",
                    job_type=config["purpose"], mode=mode, config=config,
                    tags=[config["cluster"], config["condition"], config["model_size"], config["purpose"]],
                    dir=str(output), settings=wandb.Settings(
                        base_url=wandb_config["base_url"], disable_git=True,
                        disable_code=True, console="off", init_timeout=30,
                        finish_timeout=30),
                )
            except Exception as error:
                self.log("wandb_unavailable", error=str(error))

    def log(self, event, step=None, **values):
        record = {"event": event, "unix_time": time.time(), "step": step, **values}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self.handle.write(line + "\n")
        self.handle.flush()
        print(line, flush=True)
        if self.run is not None:
            try:
                scalars = {k: v for k, v in values.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
                if step is not None and scalars:
                    self.run.log(scalars, step=step)
            except Exception as error:
                self.run = None
                self.log("wandb_logging_failed", error=str(error))
                if self.external:
                    raise RuntimeError('Sweep metrics could not be logged; local records retained') from error

    def summary(self, values):
        if self.run is not None:
            try:
                self.run.summary.update(values)
            except Exception:
                if self.external:
                    raise

    def close(self, exit_code):
        if self.run is not None and not self.external:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception as error:
                self.log("wandb_finish_failed", error=str(error))
        self.handle.close()


def collate(examples, pad_token_id):
    longest = max(len(e["input_ids"]) for e in examples)
    return {
        "input_ids": torch.tensor([e["input_ids"] + [pad_token_id] * (longest-len(e["input_ids"])) for e in examples]),
        "labels": torch.tensor([e["labels"] + [-100] * (longest-len(e["labels"])) for e in examples]),
        "attention_mask": torch.tensor([[1]*len(e["input_ids"]) + [0]*(longest-len(e["input_ids"])) for e in examples]),
    }


def token_count(batch):
    # Causal LM losses predict labels[:, 1:] from logits[:, :-1].
    return int((batch["labels"][:, 1:] != -100).sum())


def loss_sum(model, batch):
    count = token_count(batch)
    if count == 0:
        raise ValueError("Empty supervised batch")
    loss = model(**batch, use_cache=False).loss
    if not torch.isfinite(loss):
        raise ValueError("Non-finite target loss")
    return loss * count, count


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def evaluate(model, loader, device):
    model.eval()
    numerator, denominator = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device):
                total, count = loss_sum(model, batch)
            numerator += float(total)
            denominator += count
    return {"eval_loss": numerator / denominator, "eval_nll_sum": numerator,
            "eval_target_tokens": denominator}


def train_window(model, batches, optimizer, device, max_grad_norm):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    denominator = sum(token_count(batch) for batch in batches)
    numerator = 0.0
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(device):
            total, _ = loss_sum(model, batch)
        numerator += float(total.detach())
        (total / denominator).backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_grad_norm, error_if_nonfinite=True)
    if float(norm) <= 0:
        raise ValueError("Zero gradient norm")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"train_loss": numerator / denominator, "train_target_tokens": denominator,
            "gradient_norm": float(norm)}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def save_checkpoint(model, tokenizer, optimizer, scheduler, output, state):
    path = output / f"checkpoint-{state['step']}"
    temporary = output / f".checkpoint-{state['step']}.tmp"
    temporary.mkdir()
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               temporary / "training_state.pt")
    write_json(temporary / "state.json", state)
    temporary.rename(path)
    return path


def keep_checkpoints(paths, best, latest):
    for path in list(paths):
        if path not in (best, latest):
            shutil.rmtree(path)
            paths.remove(path)


def generate(model, tokenizer, rows, protocol, device, limit):
    model.eval()
    results = []
    # Round-robin variants so a small diagnostic set covers fluent/base/marked.
    groups = [[r for r in rows if r["variant"] == v] for v in
              ("pmb_fluent", "fluent_split_reference", "rsr_base", "rsr_editing_expression")]
    selected = []
    for index in range(max(map(len, groups))):
        selected.extend(group[index] for group in groups if index < len(group))
        if len(selected) >= limit:
            break
    for row in selected[:limit]:
        _, ids = prompt_ids(tokenizer, row["text"], protocol["instruction"])
        inputs = torch.tensor([ids], device=device)
        with torch.inference_mode(), autocast(device):
            tokens = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                    do_sample=False, use_cache=True,
                                    max_new_tokens=protocol["candidate"]["generation_max_new_tokens"],
                                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)[0, len(ids):].tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        if not text:
            raise ValueError(f"Empty generation: {row['example_id']}")
        results.append({"example_id": row["example_id"], "variant": row["variant"],
                        "text": text, "token_ids": tokens, "tokens": len(tokens),
                        "ended_with_eos": bool(tokens and tokens[-1] == tokenizer.eos_token_id)})
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH,
                        help="Versioned protocol; default preserves the original canary instruction")
    parser.add_argument("--model-size", choices=("2B", "9B", "27B"), required=True)
    parser.add_argument("--condition", choices=("C1", "C2", "C3"), required=True)
    parser.add_argument("--purpose", choices=("canary", "calibration", "tuning"), default="canary")
    parser.add_argument("--seed", type=int, default=17, help="Development seed; not the formal seed list")
    parser.add_argument("--max-steps", type=int, default=30, help="Use 0 for the full three-epoch development budget")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-epoch-fraction", type=float)
    parser.add_argument("--train-subset", type=Path)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--generation-examples", type=int, default=20)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, help="Local tokenizer snapshot for CPU audit")
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cluster", choices=("habrok", "lst", "local"), required=True)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.eval_every < 1 or args.max_steps < 0 or args.generation_examples < 0:
        parser.error("Invalid evaluation, step or generation budget")
    if args.micro_batch_size not in (1, 2, 4, 8, 16):
        parser.error("micro-batch-size must divide the effective batch of 16")
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("learning-rate must be finite and positive")
    if args.eval_epoch_fraction is not None and not (0 < args.eval_epoch_fraction <= 1):
        parser.error('eval-epoch-fraction must be in (0, 1]')
    if any(v is not None and v < 1 for v in (args.lora_rank, args.lora_alpha)):
        parser.error('LoRA rank and alpha must be positive')
    if args.purpose == 'tuning' and (not args.train_subset or args.max_steps != 0):
        parser.error('tuning requires a subset and max-steps 0')
    if args.purpose == "canary" and not (1 <= args.max_steps <= 50):
        parser.error("canary requires 1–50 steps; use calibration for a longer development run")
    if args.tokenizer_dir and not args.preflight_only:
        parser.error("Training must use the pinned cached tokenizer; tokenizer-dir is for CPU audit only")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This runner supports a single GPU only")
    return args


def training_plan(train_rows, protocol, args):
    steps = math.ceil(train_rows / protocol['candidate']['effective_batch_size'])
    budget = steps * protocol['candidate']['epochs']
    return {'steps_per_epoch': steps, 'budget_steps': budget,
            'total_steps': min(args.max_steps, budget) if args.max_steps else budget,
            'warmup_steps': math.ceil(budget * protocol['candidate']['warmup_ratio']),
            'eval_every': max(1, math.ceil(steps * args.eval_epoch_fraction))
                          if args.eval_epoch_fraction is not None else args.eval_every}


def expected_lora_parameters(candidates, rank):
    return sum(rank * (layer.in_features + layer.out_features) for _, layer in candidates)


def run_training(args, wandb_run=None):
    protocol = read_protocol(args.protocol)
    if args.learning_rate is not None:
        protocol["candidate"]["learning_rate"] = args.learning_rate
    if args.lora_rank is not None:
        protocol['lora']['r'] = args.lora_rank
    if args.lora_alpha is not None:
        protocol['lora']['lora_alpha'] = args.lora_alpha
    if args.purpose == 'tuning' and wandb_run is None and not args.preflight_only:
        raise ValueError('Use tuning/run_sweep.py for Sweep trials')
    os.environ["WANDB_DISABLE_CODE"] = "true"
    os.environ["WANDB_DISABLE_GIT"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    from transformers import AutoTokenizer
    spec = protocol["models"][args.model_size]
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_dir) if args.tokenizer_dir else spec["id"],
        revision=spec["revision"], cache_dir=str(args.model_cache), local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = load_rows(args.dataset_dir, protocol)
    subset_info = None
    if args.train_subset:
        rows, subset_info = select_train_subset(rows, args.train_subset, protocol)
    encoded, audit_report = audit(rows, tokenizer, protocol, args.condition)
    plan = training_plan(len(encoded['train']), protocol, args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir
    write_json(output / "data_audit.json", audit_report)
    write_json(output / 'training_plan.json', plan)
    if args.train_subset:
        shutil.copyfile(args.train_subset, output / 'subset_manifest.json')
    if args.preflight_only:
        print(json.dumps({"status": "preflight_passed", "audit": audit_report}))
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a BF16 CUDA GPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    from peft import LoraConfig, PeftModel, get_peft_model
    from peft.tuners.lora.layer import LoraLayer
    from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
    versions = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "datasets", "huggingface_hub")}
    optional_versions = {}
    for package in ("wandb", "causal-conv1d", "flash-linear-attention", "fla-core", "accelerate"):
        try:
            optional_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            optional_versions[package] = None
    for package, expected in {"torch": "2.14.1", "transformers": "5.20.2", "peft": "0.21.2"}.items():
        if versions[package].split("+")[0] != expected:
            raise RuntimeError(f"Unvalidated {package} version: {versions[package]}")
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "protocol": protocol, "versions": versions, "optional_versions": optional_versions,
              'learning_rate': protocol['candidate']['learning_rate'],
              'lora_rank': protocol['lora']['r'], 'lora_alpha': protocol['lora']['lora_alpha'],
              'subset': subset_info, 'training_plan': plan,
              'wandb_run_id': wandb_run.id if wandb_run is not None else output.name,
              'sweep_assignment': dict(wandb_run.config) if wandb_run is not None else None,
              "hostname": socket.gethostname(),
              "python": platform.python_version(), "gpu": torch.cuda.get_device_name(),
              "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
              "run_id": output.name, "code_sha256": {
                  p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (Path(__file__), Path(__file__).with_name("data.py"), args.protocol)}}
    write_json(output / "config.json", config)
    log = Recorder(output, config, args.wandb_mode, protocol["wandb"], external_run=wandb_run)
    started, exit_code = time.perf_counter(), 1
    try:
        def load_base():
            return AutoModelForCausalLM.from_pretrained(
                spec["id"], revision=spec["revision"], cache_dir=str(args.model_cache),
                local_files_only=True, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
                low_cpu_mem_usage=True, attn_implementation="sdpa")

        base = load_base()
        candidates = [(name, module) for name, module in base.named_modules()
                      if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in protocol["lora"]["target_modules"]]
        if len(candidates) != spec["modules"]:
            raise ValueError("Unexpected base target-module count")
        expected_parameters = expected_lora_parameters(candidates, protocol['lora']['r'])
        if expected_lora_parameters(candidates, 16) != spec['trainable_parameters']:
            raise ValueError('Base projection dimensions differ from the smoke-tested model')
        # Exact full paths prevent suffix matches in an unintended model tower.
        adapter_config = {**protocol["lora"], "target_modules": [name for name, _ in candidates]}
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **adapter_config))
        adapted = [name for name, module in model.named_modules() if isinstance(module, LoraLayer)]
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        parameter_count = sum(p.numel() for _, p in trainable)
        if len(adapted) != spec["modules"] or parameter_count != expected_parameters or not all("lora_" in n for n, _ in trainable):
            raise ValueError("LoRA coverage differs from passing smoke tests")
        write_json(output / "lora_modules.json", {"modules": adapted, "trainable_parameters": parameter_count})
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False
        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=protocol["candidate"]["learning_rate"], weight_decay=0.0)
        accumulation = protocol["candidate"]["effective_batch_size"] // args.micro_batch_size
        steps_per_epoch, budget, total_steps = plan['steps_per_epoch'], plan['budget_steps'], plan['total_steps']
        # The canary retains the full-budget schedule; it is not a learning-rate search.
        scheduler = get_cosine_schedule_with_warmup(optimizer, plan['warmup_steps'], budget)
        dev_loader = DataLoader(encoded["validation"], batch_size=args.micro_batch_size,
                                collate_fn=lambda x: collate(x, tokenizer.pad_token_id))
        best_loss, best_path, paths, step = math.inf, None, [], 0
        train_seconds, eval_seconds, checkpoint_seconds = 0.0, 0.0, 0.0
        log.log("run_start", steps=total_steps, full_budget_steps=budget, effective_batch=16,
                gradient_accumulation=accumulation, trainable_parameters=parameter_count)
        for epoch in range(protocol["candidate"]["epochs"]):
            generator = torch.Generator().manual_seed(args.seed + epoch)
            order = torch.randperm(len(encoded["train"]), generator=generator).tolist()
            log.log("epoch_order", epoch=epoch, sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest())
            loader = iter(DataLoader([encoded["train"][i] for i in order], batch_size=args.micro_batch_size,
                                     collate_fn=lambda x: collate(x, tokenizer.pad_token_id)))
            for _ in range(steps_per_epoch):
                batches = []
                for _ in range(accumulation):
                    batch = next(loader, None)
                    if batch is not None:
                        batches.append(batch)
                step_started = time.perf_counter()
                metrics = train_window(model, batches, optimizer, device, protocol["candidate"]["max_grad_norm"])
                scheduler.step()
                step += 1
                torch.cuda.synchronize()
                step_seconds = time.perf_counter()-step_started
                train_seconds += step_seconds
                log.log("train", step, **metrics, learning_rate=scheduler.get_last_lr()[0],
                        step_seconds=step_seconds,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
                if step % plan['eval_every'] == 0 or step == total_steps:
                    eval_started = time.perf_counter()
                    metrics = evaluate(model, dev_loader, device)
                    eval_elapsed = time.perf_counter()-eval_started
                    eval_seconds += eval_elapsed
                    state = {"step": step, "epoch": epoch, "data_order_sha256": hashlib.sha256(json.dumps(order).encode()).hexdigest(), **metrics}
                    save_started = time.perf_counter()
                    path = save_checkpoint(model, tokenizer, optimizer, scheduler, output, state)
                    paths.append(path)
                    if metrics["eval_loss"] < best_loss:
                        best_loss, best_path = metrics["eval_loss"], path
                    write_json(output / "selection.json", {"rule": "minimum global assistant-target dev loss; earliest exact tie",
                               "best_checkpoint": best_path.name, "best_eval_loss": best_loss, "last_checkpoint": path.name})
                    keep_checkpoints(paths, best_path, path)
                    save_elapsed = time.perf_counter()-save_started
                    checkpoint_seconds += save_elapsed
                    log.log("eval", step, **metrics, best_eval_loss=best_loss,
                            eval_seconds=eval_elapsed, checkpoint_seconds=save_elapsed)
                if step == total_steps:
                    break
            if step == total_steps:
                break
        # Release every reference to the trained weights and optimizer before reload.
        del batches, batch, optimizer, scheduler, model, base, candidates, trainable
        gc.collect()
        torch.cuda.empty_cache()
        reload_started = time.perf_counter()
        reloaded_base = load_base()
        reloaded = PeftModel.from_pretrained(reloaded_base, best_path, local_files_only=True)
        if sum(isinstance(m, LoraLayer) for m in reloaded.modules()) != spec["modules"]:
            raise ValueError("Reloaded adapter coverage changed")
        reloaded_loss = evaluate(reloaded, dev_loader, device)
        if not math.isclose(reloaded_loss["eval_loss"], best_loss, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("Selected checkpoint dev loss changed after reload")
        reload_seconds = time.perf_counter()-reload_started
        generated = []
        if args.generation_examples:
            generated = generate(reloaded, tokenizer, condition_rows(rows["validation"], args.condition),
                                 protocol, device, args.generation_examples)
            write_json(output / "dev_generations.json", generated)
        summary = {"status": "development_passed", "purpose": args.purpose, "formal_run": False,
                   "steps": step, "best_checkpoint": best_path.name, "best_eval_loss": best_loss,
                   'best_step': int(best_path.name.split('-')[-1]),
                   'train_seconds': train_seconds, 'eval_seconds': eval_seconds,
                   'checkpoint_seconds': checkpoint_seconds, 'reload_seconds': reload_seconds,
                   "reload_eval_loss": reloaded_loss["eval_loss"], "generation_examples": len(generated),
                   "generation_eos_count": sum(r["ended_with_eos"] for r in generated),
                   "seconds": time.perf_counter()-started,
                   "peak_allocated_gib": torch.cuda.max_memory_allocated()/1024**3}
        write_json(output / "summary.json", summary)
        log.log("run_complete", step, **summary)
        log.summary(summary)
        exit_code = 0
    except Exception as error:
        log.log("run_failed", error_type=type(error).__name__, error=str(error))
        write_json(output / "failure.json", {"error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        log.close(exit_code)


if __name__ == "__main__":
    run_training(parse_args())
