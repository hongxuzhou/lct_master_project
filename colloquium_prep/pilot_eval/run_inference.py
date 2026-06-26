#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 — Pilot inference: LoRA Gemma-2-9B-it parses the 7 repair conditions.
-- Hongxu Zhou, 2026

Loads base google/gemma-2-9b-it + the LoRA adapter
(Shrikes/sbn-gemma2-9b-lora-pmb), melts the WIDE pilot dataset
(Shrikes/self_repair_parsing_pilot_data) into a LONG (id, condition) table, and
generates one predicted SBN per (id, condition). Output feeds straight into
evaluate_smatchpp.py.

CRITICAL alignment with training (lora_data_prep/lora_data_gen.py):
  * prompt = INSTRUCTION + "\n" + <sentence>, wrapped via the tokenizer's chat
    template with add_generation_prompt=True.
  * The adapter repo ships the training-time tokenizer (custom Gemma-2 chat
    template); we load the tokenizer FROM THE ADAPTER so the prompt rendering is
    byte-identical to training.
  * greedy decode (deterministic, reproducible scoring).
  * left-padding for batched generation (Gemma-2 inference convention); training
    used right-padding, which is correct for SFT but WRONG for generation.

Each (id, condition) is an INDEPENDENT forward pass — batching never leaks
context across examples (per-sequence attention masks), so parsing a mid-repair
never "sees" a head-repair. Batching is throughput only.

Output (LONG): columns [id, condition, input_nl, pred_sbn]. Gold (mr) is NOT
included — it stays in the static HF dataset and is joined at scoring time via
evaluate_smatchpp.py --gold-file.

Usage (HPC):
    python3 run_inference.py \
        --adapter Shrikes/sbn-gemma2-9b-lora-pmb \
        --base /scratch/common_models/.../models--google--gemma-2-9b-it \
        --output preds_long.parquet \
        --batch-size 8

Deps: torch, transformers, peft, datasets, pandas, pyarrow.
"""

import os
import argparse
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Must match lora_data_prep/lora_data_gen.py exactly.
INSTRUCTION = "Please parse the following sentence to discourse representation structure:"

# WIDE pilot column -> condition label. `nl` is the clean gold-sentence baseline.
CONDITION_COLS = {
    "nl":                       "gold",
    "repair_head_nl":           "repair_head",
    "repair_mid_nl":            "repair_mid",
    "repair_tail_nl":           "repair_tail",
    "repair_head_interrug_nl":  "repair_head_interrug",
    "repair_mid_interrug_nl":   "repair_mid_interrug",
    "repair_tail_interrug_nl":  "repair_tail_interrug",
}


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
    """WIDE (one row per source sentence) -> LONG (one row per (id, condition))."""
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
    """Render the exact training prompt + the generation cue (model turn)."""
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n{sentence}"}]
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
    # School policy: load the base from the shared cluster cache, not a fresh
    # download. resolve_model_path() handles the HF-cache snapshots/<hash>/ layout.
    ap.add_argument("--base",
                    default="/scratch/common_models/HuggingFace/models--google--gemma-2-9b-it",
                    help="Base model path (shared cluster cache) or HF id.")
    ap.add_argument("--adapter", default="Shrikes/sbn-gemma2-9b-lora-pmb",
                    help="LoRA adapter path or HF id. Tokenizer is loaded from here.")
    ap.add_argument("--dataset", default="Shrikes/self_repair_parsing_pilot_data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", required=True, help="Output LONG table (.parquet/.tsv/.csv).")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="Smoke-test: only first N long rows.")
    ap.add_argument("--use-4bit", action="store_true", help="QLoRA 4-bit (40GB card).")
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

    # ── Tokenizer (from adapter = training-time template) ───────────────────
    print(f"[info] loading tokenizer from adapter: {args.adapter}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"      # generation convention

    # Stop on <end_of_turn> (Gemma-2 chat) as well as eos.
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [tokenizer.eos_token_id]
    if eot is not None and eot != tokenizer.unk_token_id:
        eos_ids.append(eot)

    # ── Model: base + adapter ───────────────────────────────────────────────
    base_path = resolve_model_path(args.base)
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
    print(f"[info] attaching adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
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

    long_df["pred_sbn"] = preds
    write_table(long_df[["id", "condition", "input_nl", "pred_sbn"]], out_path)
    print(f"\n[done] saved {n} predictions -> {out_path}")


if __name__ == "__main__":
    main()
