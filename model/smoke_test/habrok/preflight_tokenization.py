#!/usr/bin/env python3
"""CPU-only offline preflight for the Habrok smoke-test tokenization."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


HABROK_DIR = Path(__file__).resolve().parent
SMOKE_DIR = HABROK_DIR.parent
sys.path.insert(0, str(SMOKE_DIR))

# Hugging Face reads these switches during import, so set them before importing
# the shared smoke-test module and its datasets/transformers dependencies.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from qwen35_lora_smoke_test import (  # noqa: E402
    IGNORE_INDEX,
    MODEL_SPECS,
    AutoTokenizer,
    load_balanced_smoke_subset,
    tokenize_row,
)


MODEL_REVISIONS = {
    "2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER")
    if not user:
        raise RuntimeError("USER is unset")
    scratch = Path("/scratch") / user
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=tuple(MODEL_REVISIONS), default="2B")
    parser.add_argument("--condition", choices=("all", "clean", "repair"), default="all")
    parser.add_argument("--examples-per-variant", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--model-cache", type=Path, default=scratch / "huggingface" / "hub"
    )
    parser.add_argument(
        "--dataset-cache", type=Path, default=scratch / "huggingface" / "datasets"
    )
    parser.add_argument(
        "--revision-dir",
        type=Path,
        default=scratch / "thesis" / "qwen35_lora_smoke" / "revisions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    revision_file = args.revision_dir / "repair-sbn.revision"
    if not revision_file.is_file():
        raise FileNotFoundError(f"missing {revision_file}; run prepare.sh first")
    dataset_revision = revision_file.read_text(encoding="utf-8").strip()
    model_revision = MODEL_REVISIONS[args.model_size]
    model_id = MODEL_SPECS[args.model_size]["model_id"]

    subset, _ = load_balanced_smoke_subset(
        args.examples_per_variant,
        args.seed,
        args.dataset_cache,
        dataset_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=model_revision,
        cache_dir=str(args.model_cache),
        local_files_only=True,
        use_fast=True,
    )
    conditions = ("clean", "repair") if args.condition == "all" else (args.condition,)
    reports = []
    for condition in conditions:
        examples = [
            tokenize_row(row, tokenizer, condition, args.max_length) for row in subset
        ]
        lengths = [len(example.input_ids) for example in examples]
        supervised = [
            sum(label != IGNORE_INDEX for label in example.labels)
            for example in examples
        ]
        reports.append(
            {
                "condition": condition,
                "examples": len(examples),
                "variant_counts": dict(Counter(example.variant for example in examples)),
                "max_prompt_tokens": max(example.prompt_length for example in examples),
                "max_untruncated_sequence_tokens": max(
                    example.full_length for example in examples
                ),
                "max_sequence_tokens_after_truncation": max(lengths),
                "min_supervised_tokens": min(supervised),
                "max_supervised_tokens": max(supervised),
                "fully_supervised_examples": sum(
                    example.full_length <= args.max_length for example in examples
                ),
                "truncated_examples": sum(
                    example.full_length > args.max_length for example in examples
                ),
            }
        )

    print(
        json.dumps(
            {
                "status": "passed",
                "model_size": args.model_size,
                "model_revision": model_revision,
                "dataset_revision": dataset_revision,
                "max_length": args.max_length,
                "reports": reports,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
