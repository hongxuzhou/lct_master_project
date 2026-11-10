#!/usr/bin/env python3
"""Audit full train/dev condition views without model weights or W&B."""
import argparse
import hashlib
import json
from pathlib import Path

from data import PROTOCOL_PATH, audit, load_rows, read_protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--model-sizes", nargs="+", choices=("2B", "9B", "27B"), default=["2B"])
    parser.add_argument("--tokenizer-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from transformers import AutoTokenizer
    protocol = read_protocol(args.protocol)
    rows = load_rows(args.dataset_dir, protocol)
    result = {"dataset_revision": protocol["dataset"]["revision"],
              "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
              "instruction": protocol["instruction"],
              "instruction_sha256": hashlib.sha256(protocol["instruction"].encode()).hexdigest(),
              "models": {}}
    for size in args.model_sizes:
        spec = protocol["models"][size]
        if args.tokenizer_root:
            directory = args.tokenizer_root / size
            # hf download records the resolved revision for every tokenizer file.
            for path in directory.iterdir():
                if path.is_file():
                    metadata = directory / ".cache/huggingface/download" / (path.name + ".metadata")
                    if metadata.read_text().splitlines()[0] != spec["revision"]:
                        raise ValueError(f"Unpinned tokenizer file: {path}")
            tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, use_fast=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"],
                cache_dir=str(args.model_cache), local_files_only=True, use_fast=True)
        reports = {}
        for condition in ("C1", "C2", "C3"):
            _, report = audit(rows, tokenizer, protocol, condition)
            reports[condition] = report
        for split in rows:
            if reports["C2"][split]["input_digest"] != reports["C3"][split]["input_digest"]:
                raise ValueError(f"C2/C3 {split} input mismatch")
        result["models"][size] = {"revision": spec["revision"], "conditions": reports}
    result["status"] = "preflight_passed"
    result["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (Path(__file__), Path(__file__).with_name("data.py"))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
