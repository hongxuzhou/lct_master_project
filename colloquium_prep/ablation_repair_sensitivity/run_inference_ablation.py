#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 — Repair-sensitivity ablation inference: base Gemma-2-9B-it, zero-shot.
-- Hongxu Zhou, 2026

HPC counterpart to smoke_test_mlx.py, same schema, transformers/CUDA backend.
Loads base `google/gemma-2-9b-it` — **no LoRA adapter** (design decision 1 in
the ablation plan: isolate the pretrained model's linguistic sensitivity to
repair from anything the narrow SBN-only fine-tuning changed). Melts the WIDE
pilot dataset (Shrikes/self_repair_parsing_pilot_data) into LONG (id,
condition), and generates one zero-shot repair-detection/localization
response per (id, condition) using a single knowledge-injection prompt
(design decision 7 — the no-definition variant was piloted locally and
dropped as uninterpretable; see the plan doc and PROMPT_WITH_DEF below).

Each (id, condition) is an INDEPENDENT forward pass — batching never leaks
context across examples (per-sequence attention masks). Batching is
throughput only.

Output (LONG): columns [id, condition, prompt_variant, input_nl, raw_output].
`prompt_variant` is always "with_def" here — kept as a column (not a bare
constant) so this table has the exact same schema evaluate_ablation.py
already expects from the local MLX smoke test. Feeds straight into
evaluate_ablation.py, meant to be run LOCALLY after pulling this parquet off
HPC (inference and evaluation are deliberately separate stages/scripts so the
scoring script can be developed and tested against a small local dry run
while the real job queues on HPC).

Usage (HPC):
    python3 run_inference_ablation.py \
        --base /scratch/common_models/.../models--google--gemma-2-9b-it \
        --output preds_ablation.parquet \
        --batch-size 16

Deps: torch, transformers, datasets, pandas, pyarrow.
"""

import os
import argparse
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Same WIDE->LONG mapping as pilot_eval/run_inference.py.
CONDITION_COLS = {
    "nl":                       "gold",
    "repair_head_nl":           "repair_head",
    "repair_mid_nl":            "repair_mid",
    "repair_tail_nl":           "repair_tail",
    "repair_head_interrug_nl":  "repair_head_interrug",
    "repair_mid_interrug_nl":   "repair_mid_interrug",
    "repair_tail_interrug_nl":  "repair_tail_interrug",
}

# Must match smoke_test_mlx.py's PROMPT_WITH_DEF exactly — same instrument,
# different backend. If you edit the wording, edit both files.
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


def resolve_model_path(p: str) -> str:
    """Accept either a plain dir or an HF-cache dir (snapshots/<hash>/)."""
    snap = os.path.join(p, "snapshots")
    if os.path.isdir(snap):
        subs = [os.path.join(snap, d) for d in os.listdir(snap)
                if os.path.isdir(os.path.join(snap, d))]
        if subs:
            return sorted(subs)[-1]
    return p


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


def build_prompt(tokenizer, sentence: str) -> str:
    messages = [{"role": "user", "content": PROMPT_WITH_DEF.format(sentence=sentence)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, max_new_tokens, eos_ids):
    enc = tokenizer(prompts, return_tensors="pt", padding=True,
                    add_special_tokens=False).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,                 # greedy, deterministic
        num_beams=1,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[:, enc["input_ids"].shape[1]:]     # left-pad => uniform input length
    texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [t.strip() for t in texts]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",
                    default="/scratch/common_models/HuggingFace/models--google--gemma-2-9b-it",
                    help="Base model path (shared cluster cache) or HF id. No adapter.")
    ap.add_argument("--dataset", default="Shrikes/self_repair_parsing_pilot_data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", required=True, help="Output LONG table (.parquet/.tsv/.csv).")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test: only first N long rows.")
    ap.add_argument("--use-4bit", action="store_true", help="4-bit if VRAM-constrained.")
    return ap.parse_args()


def write_table(df, path: Path):
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        sep = "\t" if path.suffix in (".tsv", ".txt") else ","
        df.to_csv(path, sep=sep, index=False)


def main():
    args = parse_args()
    out_path = Path(args.output)

    # ── Data: load WIDE, melt to LONG ───────────────────────────────────────
    print(f"[info] loading dataset {args.dataset} [{args.split}]")
    ds = load_dataset(args.dataset, split=args.split)
    long_df = melt_wide_to_long(ds.to_pandas())
    if args.limit:
        long_df = long_df.head(args.limit).reset_index(drop=True)
        print(f"[info] --limit {args.limit}: {len(long_df)} rows")

    # ── Tokenizer (from base — no adapter in this ablation) ─────────────────
    base_path = resolve_model_path(args.base)
    print(f"[info] loading tokenizer from base: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"      # generation convention

    # Stop on <end_of_turn> (Gemma-2 chat) as well as eos.
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [tokenizer.eos_token_id]
    if eot is not None and eot != tokenizer.unk_token_id:
        eos_ids.append(eot)

    # ── Model: base only ─────────────────────────────────────────────────────
    print(f"[info] loading base model: {base_path}  (bf16 supported: {torch.cuda.is_bf16_supported()})")
    quant = None
    if args.use_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.bfloat16,
        quantization_config=quant,
        attn_implementation="eager",     # Gemma-2 softcap correctness
        device_map={"": 0},
    )
    model.eval()

    # ── Generate ────────────────────────────────────────────────────────────
    preds = []
    n = len(long_df)
    for start in range(0, n, args.batch_size):
        batch = long_df.iloc[start:start + args.batch_size]
        prompts = [build_prompt(tokenizer, s) for s in batch["input_nl"]]
        preds.extend(generate_batch(model, tokenizer, prompts,
                                    args.max_new_tokens, eos_ids))
        done = min(start + args.batch_size, n)
        print(f"  generated {done:>5}/{n}", flush=True)

    long_df["prompt_variant"] = "with_def"
    long_df["raw_output"] = preds
    write_table(long_df[["id", "condition", "prompt_variant", "input_nl", "raw_output"]], out_path)
    print(f"\n[done] saved {n} predictions -> {out_path}")


if __name__ == "__main__":
    main()
