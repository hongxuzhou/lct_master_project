#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local smoke test — repair-sensitivity ablation harness, MLX backend.
-- Hongxu Zhou, 18/Jul/2026

DEV-ONLY. Verifies the ablation's data plumbing (melt WIDE dataset -> LONG,
render both prompt variants, call the model, write the output table) on a
laptop-sized model (mlx-community/gemma-2-2b-it-4bit) before the real run
uses google/gemma-2-9b-it (base, no LoRA) on HPC via transformers/CUDA.

Do NOT read anything into this model's detection/localisation accuracy —
2B-4bit-MLX tells you nothing about 9B-bf16-CUDA. This script exists only to
freeze and validate the OUTPUT SCHEMA so the scoring script can be developed
against real (small, fast) data while the HPC job queues. The HPC version of
this script will be a separate file using transformers, matching
run_inference.py's model-loading conventions.

Output (LONG): columns [id, condition, prompt_variant, input_nl, raw_output].
No LoRA adapter — per the ablation design, this tests the base instruction-
tuned model's zero-shot repair sensitivity, decoupled from SBN fine-tuning.

Usage:
    python3 smoke_test_mlx.py --limit 5 --output smoke_out.parquet

Deps: mlx-lm, datasets, pandas, pyarrow.
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from mlx_lm import load, generate

# Same WIDE->LONG mapping as run_inference.py (colloquium_prep/pilot_eval/).
# "nl" = gold/no-repair control; the rest are the 6 repair conditions.
CONDITION_COLS = {
    "nl":                       "gold",
    "repair_head_nl":           "repair_head",
    "repair_mid_nl":            "repair_mid",
    "repair_tail_nl":           "repair_tail",
    "repair_head_interrug_nl":  "repair_head_interrug",
    "repair_mid_interrug_nl":   "repair_mid_interrug",
    "repair_tail_interrug_nl":  "repair_tail_interrug",
}

# --- Prompt variants (RQ-C) -------------------------------------------------
# DRAFT wording only — plan's "Open items" flags these as needing concrete
# wording at implementation time. Good enough to validate the harness; revisit
# before the real HPC run.

PROMPT_NO_DEF = (
    "Read the following sentence, which may be spoken and may contain the "
    "speaker interrupting themselves and restarting or correcting what they "
    "were saying.\n"
    "If it does, write down exactly the words the speaker abandoned, then on "
    "a new line the words of the correction they actually meant. Copy the "
    "words verbatim from the sentence, do not paraphrase.\n"
    "If the sentence has no such self-correction, reply with exactly: N/A\n\n"
    "Sentence: {sentence}"
)

PROMPT_WITH_DEF = (
    "Self-repair in spoken language has up to three parts: the REPARANDUM "
    "(the abandoned words the speaker meant to retract), an optional "
    "INTERREGNUM (a filler bridging the interruption, e.g. \"I mean,\"), and "
    "the REPAIR (the words the speaker replaces it with).\n"
    "Read the following sentence. If it contains a self-repair, output the "
    "REPARANDUM, the INTERREGNUM (if present), and the REPAIR, each on its "
    "own line and copied verbatim from the sentence.\n"
    "If the sentence has no self-repair, reply with exactly: N/A\n\n"
    "Sentence: {sentence}"
)

PROMPT_VARIANTS = {
    "no_def": PROMPT_NO_DEF,
    "with_def": PROMPT_WITH_DEF,
}


def melt_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in CONDITION_COLS if c in df.columns]
    missing = [c for c in CONDITION_COLS if c not in df.columns]
    if missing:
        print(f"[warn] columns absent from dataset, skipped: {missing}")
    rows = []
    for _, r in df.iterrows():
        for col in present:
            text = r[col]
            if text is None or str(text).strip() in ("", "nan"):
                continue
            rows.append({"id": r["id"], "condition": CONDITION_COLS[col],
                         "input_nl": str(text).strip()})
    long_df = pd.DataFrame(rows)
    print(f"[info] melted {len(df)} source rows -> {len(long_df)} (id, condition) rows "
          f"across {long_df['condition'].nunique()} conditions")
    return long_df


def build_prompt(tokenizer, template: str, sentence: str) -> str:
    messages = [{"role": "user", "content": template.format(sentence=sentence)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/gemma-2-2b-it-4bit",
                     help="MLX model repo. Smoke-test only, NOT the ablation model.")
    ap.add_argument("--dataset", default="Shrikes/self_repair_parsing_pilot_data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", required=True, help="Output LONG table (.parquet/.tsv/.csv).")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=5,
                     help="Smoke-test: only first N *source sentences* "
                          "(each expands to 7 conditions x 2 prompt variants).")
    return ap.parse_args()


def write_table(df: pd.DataFrame, path: Path):
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        sep = "\t" if path.suffix in (".tsv", ".txt") else ","
        df.to_csv(path, sep=sep, index=False)


def main():
    args = parse_args()
    out_path = Path(args.output)

    print(f"[info] loading dataset {args.dataset} [{args.split}]")
    ds = load_dataset(args.dataset, split=args.split)
    df = ds.to_pandas()
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
        print(f"[info] --limit {args.limit} source sentences")
    long_df = melt_wide_to_long(df)

    print(f"[info] loading MLX model: {args.model}")
    model, tokenizer = load(args.model)

    rows = []
    total = len(long_df) * len(PROMPT_VARIANTS)
    done = 0
    for _, r in long_df.iterrows():
        for variant_name, template in PROMPT_VARIANTS.items():
            prompt = build_prompt(tokenizer, template, r["input_nl"])
            raw_output = generate(
                model, tokenizer, prompt=prompt,
                max_tokens=args.max_tokens, verbose=False,
            ).strip()
            rows.append({
                "id": r["id"],
                "condition": r["condition"],
                "prompt_variant": variant_name,
                "input_nl": r["input_nl"],
                "raw_output": raw_output,
            })
            done += 1
            print(f"  generated {done:>4}/{total}", flush=True)

    out_df = pd.DataFrame(rows)
    write_table(out_df, out_path)
    print(f"\n[done] saved {len(out_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
