#!/usr/bin/env python3
"""Cluster-neutral CPU/offline preflight for smoke-test tokenization."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


SMOKE_DIR = Path(__file__).resolve().parent
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


DEFAULT_MANIFEST = SMOKE_DIR / "revisions.json"


def load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["dataset"]["id"] == "Shrikes/repair-sbn"
    assert set(manifest["models"]) == set(MODEL_SPECS)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=tuple(MODEL_SPECS), default="2B")
    parser.add_argument("--condition", choices=("all", "clean", "repair"), default="all")
    parser.add_argument("--examples-per-variant", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    dataset_revision = manifest["dataset"]["revision"]
    model_entry = manifest["models"][args.model_size]
    model_revision = model_entry["revision"]
    model_id = model_entry["id"]
    assert model_id == MODEL_SPECS[args.model_size]["model_id"]

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
