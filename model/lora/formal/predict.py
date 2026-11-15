"""Generate one formal run's predictions from a frozen checkpoint."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

from freeze import digest

LORA = Path(__file__).resolve().parents[1]
TEST_FILES = ("data/test.parquet", "hard_negative/test.parquet", "later_reference/test.parquet")


def panel(root, hashes, condition):
    import pyarrow.parquet as pq

    seen = set()
    rows = []
    for name in TEST_FILES:
        path = root / name
        if digest(path) != hashes[name]:
            raise ValueError(f"Test table hash mismatch: {path}")
        columns = ["example_id", "text"] + (["variant"] if name == TEST_FILES[0] else [])
        for row in pq.read_table(path, columns=columns).to_pylist():
            rid = row["example_id"]
            if not rid or rid in seen or not row["text"]:
                raise ValueError(f"Invalid or duplicate test input: {rid!r}")
            seen.add(rid)
            if name == TEST_FILES[0] and row["variant"] not in {
                "pmb_fluent", "rsr_base", "rsr_editing_expression"
            }:
                raise ValueError(f"Unknown test variant: {row['variant']}")
            if condition != "C1" or name == TEST_FILES[1] or row.get("variant") == "pmb_fluent":
                rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--test-root", required=True, type=Path)
    p.add_argument("--test-hashes", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--check", action="store_true", help="validate inputs without loading a model")
    args = p.parse_args()
    matches = [r for r in json.loads(args.manifest.read_text()) if r["run_id"] == args.run_id]
    if len(matches) != 1:
        p.error("run ID must identify exactly one frozen checkpoint")
    entry, = matches
    path = Path(entry["checkpoint_path"])
    config_path = path.parent / "config.json"
    if digest(config_path) != entry["config_sha256"]:
        raise ValueError("Run config changed")
    for name, expected in entry["files"].items():
        if digest(path / name) != expected:
            raise ValueError(f"Checkpoint file changed: {name}")
    config = json.loads(config_path.read_text())
    condition = entry["condition"]
    if condition not in ("C1", "C2", "C3"):
        raise ValueError("Unknown condition")
    rows = panel(args.test_root, json.loads(args.test_hashes.read_text()), condition)
    if not rows:
        raise ValueError("Empty test panel")
    print(f"{entry['run_id']}: verified checkpoint and {len(rows)} inputs", flush=True)
    if args.check:
        return
    if args.output.exists():
        raise FileExistsError(args.output)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(LORA))
    from data import prompt_ids

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 CUDA GPU required")
    protocol = config["protocol"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        entry["base_model"], revision=entry["base_revision"],
        cache_dir=os.environ.get("MODEL_CACHE", config["model_cache"]), local_files_only=True,
        torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        low_cpu_mem_usage=True, attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, path, local_files_only=True).eval()
    fields = ("example_id", "condition", "pred_sbn", "tokens", "ended_with_eos")
    with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                     dir=args.output.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i, row in enumerate(rows, 1):
                    _, ids = prompt_ids(tokenizer, row["text"], protocol["instruction"])
                    inputs = torch.tensor([ids], device="cuda:0")
                    tokens = model.generate(
                        input_ids=inputs, attention_mask=torch.ones_like(inputs),
                        do_sample=False, use_cache=True,
                        max_new_tokens=protocol["candidate"]["generation_max_new_tokens"],
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                    )[0, len(ids):].tolist()
                    writer.writerow(dict(example_id=row["example_id"], condition=condition,
                                         pred_sbn=tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                                         tokens=len(tokens),
                                         ended_with_eos=bool(tokens and tokens[-1] == tokenizer.eos_token_id)))
                    if i % 100 == 0:
                        print(f"{i}/{len(rows)} predictions", flush=True)
            stream.flush()
            os.link(temporary, args.output)
        finally:
            temporary.unlink()


if __name__ == "__main__":
    main()
