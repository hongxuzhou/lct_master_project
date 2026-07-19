#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 — repair-sensitivity ablation scoring.
-- Hongxu Zhou, 2026

Input: LONG table of raw zero-shot generations, one row per (id, condition[,
prompt_variant]) — the output of the ablation's inference script (currently
`smoke_test_mlx.py` for local dev; the HPC version follows the same schema).
Required columns: id, condition, input_nl, raw_output. `prompt_variant` is
optional — if present, all scoring is grouped by it (kept only so the already-
collected two-variant smoke data can still be scored; the real ablation run
uses a single knowledge-injection prompt, see plan design decision 7).

Ground truth is NOT stored anywhere — it's derived on the fly from the static
HF dataset via ground_truth.extract_ground_truth() (diff-based, see that
module's docstring). `--dataset` defaults to the same
Shrikes/self_repair_parsing_pilot_data used for inference.

Two-tier scoring (see ablation plan, "Scoring plan"):
  Tier 1 (detection):    N/A vs. non-N/A, cross-tabbed against known ground
                          truth (gold=negative, 6 repair conditions=positive).
  Tier 2 (localization):  among true positives (predicted positive AND actual
                          positive AND NOT ground_truth.suspect), token-overlap
                          P/R/F1 of the model's reparandum/interregnum/repair
                          fields against the diff-derived spans.

Paired significance tests across conditions reuse the same approach as Q2 in
pilot_report.ipynb (cochrans_q/mcnemar/holm_correct for Tier 1,
friedman/wilcoxon/holm_correct for Tier 2) — copied here rather than imported
since the originals live in a notebook.

Output: --output gets the row-level scored table (one row per input row, plus
        detected/gt_positive/f1_* columns). Summary tables print to stdout.

Usage:
    python3 evaluate_ablation.py -i smoke_out.parquet -o smoke_out_scored.parquet

Deps: datasets, pandas, numpy, scipy, pyarrow.
"""

import argparse
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import chi2, friedmanchisquare, wilcoxon

from ground_truth import extract_ground_truth

CONDITIONS = ["gold", "repair_head", "repair_mid", "repair_tail",
              "repair_head_interrug", "repair_mid_interrug", "repair_tail_interrug"]
REPAIR_CONDITIONS = [c for c in CONDITIONS if c != "gold"]
VARIETY_COL = {c: ("nl" if c == "gold" else f"{c}_nl") for c in CONDITIONS}


# ── Stats helpers (ported from pilot_report.ipynb Q2 analysis) ─────────────

def cochrans_q(mat):
    """mat: N x k binary matrix, 1 = success. Tests whether success rate differs across k paired conditions."""
    N, k = mat.shape
    col_sums, row_sums = mat.sum(axis=0), mat.sum(axis=1)
    Cbar = col_sums.mean()
    num = k * (k - 1) * np.sum((col_sums - Cbar) ** 2)
    den = k * row_sums.sum() - np.sum(row_sums ** 2)
    Q = num / den
    dfree = k - 1
    return Q, dfree, chi2.sf(Q, dfree)


def mcnemar(a, b):
    """Paired binary vectors. Compares only the discordant pairs, chi2 with continuity correction."""
    n01, n10 = np.sum((a == 0) & (b == 1)), np.sum((a == 1) & (b == 0))
    n = n01 + n10
    if n == 0:
        return np.nan, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / n
    return stat, chi2.sf(stat, 1)


def holm_correct(pvals):
    """Step-down Holm correction for a small family of pairwise tests."""
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adj, running_max = np.empty(len(pvals)), 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (len(pvals) - rank) * pvals[idx])
        adj[idx] = min(running_max, 1.0)
    return adj


# ── Model output parsing ────────────────────────────────────────────────────

_FIELD_RE = re.compile(
    r"REPARANDUM:\s*(?P<reparandum>[^\n]*)\n"
    r"INTERREGNUM:\s*(?P<interregnum>[^\n]*)\n"
    r"REPAIR:\s*(?P<repair>.*)",
    re.IGNORECASE | re.DOTALL,
)


def parse_output(raw: str) -> dict:
    """Returns dict(is_na, malformed, reparandum, interregnum, repair).
    Per plan design decision 4, N/A is the sole negative-class marker; any
    other output that doesn't parse into the 3-field block still counts as
    "detected something" for Tier 1 (malformed=True), just with no usable
    spans for Tier 2."""
    text = (raw or "").strip()
    if text.upper() == "N/A":
        return dict(is_na=True, malformed=False, reparandum=None, interregnum=None, repair=None)

    m = _FIELD_RE.search(text)
    if m is None:
        return dict(is_na=False, malformed=True, reparandum=None, interregnum=None, repair=None)

    return dict(
        is_na=False, malformed=False,
        reparandum=m.group("reparandum").strip() or None,
        interregnum=m.group("interregnum").strip() or None,
        repair=m.group("repair").strip() or None,
    )


# ── Token-overlap scoring (Tier 2) ──────────────────────────────────────────

def _norm_tokens(text):
    if not text:
        return []
    return [re.sub(r'^[\'"]+|[\'".,!?]+$', "", t.lower()) for t in text.split()]


def token_prf1(pred: str, gold: str):
    """Bag-of-words precision/recall/F1 (ROUGE-1 style, multiset overlap).
    Both-empty = perfect (correctly predicted absence); one-empty = zero."""
    from collections import Counter
    p_tok, g_tok = _norm_tokens(pred), _norm_tokens(gold)
    if not p_tok and not g_tok:
        return 1.0, 1.0, 1.0
    if not p_tok or not g_tok:
        return 0.0, 0.0, 0.0
    overlap = sum((Counter(p_tok) & Counter(g_tok)).values())
    prec = overlap / len(p_tok)
    rec = overlap / len(g_tok)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return prec, rec, f1


# ── Pipeline ─────────────────────────────────────────────────────────────

def load_gold_lookup(dataset: str, split: str) -> dict:
    """id -> {condition: (gold_nl, variety_nl)}."""
    ds = load_dataset(dataset, split=split).to_pandas()
    lookup = {}
    for _, r in ds.iterrows():
        per_id = {}
        for cond in CONDITIONS:
            col = VARIETY_COL[cond]
            if col in ds.columns and pd.notna(r[col]):
                per_id[cond] = (r["nl"], str(r[col]))
        lookup[r["id"]] = per_id
    return lookup


def score_rows(df: pd.DataFrame, gold_lookup: dict) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        cond = row["condition"]
        gold_nl, variety_nl = gold_lookup[row["id"]][cond]
        gt = extract_ground_truth(gold_nl, variety_nl, cond)
        parsed = parse_output(row["raw_output"])

        detected = not parsed["is_na"]
        gt_positive = cond != "gold"

        rec = {
            "id": row["id"], "condition": cond,
            "prompt_variant": row.get("prompt_variant", "with_def"),
            "detected": detected, "malformed": parsed["malformed"],
            "gt_positive": gt_positive, "suspect": gt.suspect,
        }

        is_tp = detected and gt_positive
        if is_tp and not gt.suspect:
            p_rec, p_prec, p_f1 = token_prf1(parsed["reparandum"], gt.reparandum)
            rec["reparandum_p"], rec["reparandum_r"], rec["reparandum_f1"] = p_rec, p_prec, p_f1
            i_rec, i_prec, i_f1 = token_prf1(parsed["interregnum"], gt.interregnum or "")
            rec["interregnum_p"], rec["interregnum_r"], rec["interregnum_f1"] = i_rec, i_prec, i_f1
            r_rec, r_prec, r_f1 = token_prf1(parsed["repair"], gt.repair)
            rec["repair_p"], rec["repair_r"], rec["repair_f1"] = r_rec, r_prec, r_f1
        else:
            for pfx in ("reparandum", "interregnum", "repair"):
                rec[f"{pfx}_p"] = rec[f"{pfx}_r"] = rec[f"{pfx}_f1"] = np.nan

        rows.append(rec)
    return pd.DataFrame(rows)


def run_group(scored: pd.DataFrame, conditions, label):
    sub = scored[scored["condition"].isin(conditions)].copy()
    sub["success"] = sub["detected"].astype(int)
    pairs = list(itertools.combinations(conditions, 2))

    # Tier 1: detection
    wide_det = sub.pivot(index="id", columns="condition", values="success").dropna()[conditions]
    Q, dfree, pQ = cochrans_q(wide_det.values)
    raw_p = np.array([mcnemar(wide_det[a].values, wide_det[b].values)[1] for a, b in pairs])
    holm_p = holm_correct(raw_p)
    det_pairwise = pd.DataFrame({
        "pair": [f"{a} vs {b}" for a, b in pairs],
        "p_raw": raw_p, "p_holm": holm_p, "significant": holm_p < 0.05,
    })

    print(f"=== {label}: Tier 1 detection rate (N={len(wide_det)}) ===")
    print(wide_det.mean().round(4).to_string())
    print(f"Cochran's Q = {Q:.2f}, df={dfree}, p = {pQ:.3g}\n")
    print(det_pairwise.to_string(index=False))
    print()

    # Tier 2: localization F1 among clean true positives
    tp = sub[sub["detected"] & sub["gt_positive"] & ~sub["suspect"]]
    f1_means = {}
    for metric in ("reparandum_f1", "repair_f1"):
        wide_f1 = tp.pivot(index="id", columns="condition", values=metric)
        wide_f1 = wide_f1[[c for c in conditions if c in wide_f1.columns]].dropna()
        if len(wide_f1) >= 3 and wide_f1.shape[1] >= 2:
            stat_fr, p_fr = friedmanchisquare(*[wide_f1[c] for c in wide_f1.columns])
            print(f"=== {label}: Tier 2 {metric} among clean TPs (N={len(wide_f1)}) ===")
            print(wide_f1.mean().round(4).to_string())
            print(f"Friedman chi2 = {stat_fr:.2f}, p = {p_fr:.3g}\n")
        else:
            print(f"=== {label}: Tier 2 {metric} — not enough paired clean TPs to test (N={len(wide_f1)}) ===\n")
        f1_means[metric] = wide_f1.mean() if len(wide_f1) else None

    return dict(det_rate=wide_det.mean(), det_pairwise=det_pairwise, **f1_means)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Raw generations LONG table.")
    ap.add_argument("-o", "--output", required=True, help="Scored LONG table.")
    ap.add_argument("--dataset", default="Shrikes/self_repair_parsing_pilot_data")
    ap.add_argument("--split", default="train")
    return ap.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    sep = "\t" if path.suffix in (".tsv", ".txt") else ","
    return pd.read_csv(path, sep=sep)


def write_table(df: pd.DataFrame, path: Path):
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        sep = "\t" if path.suffix in (".tsv", ".txt") else ","
        df.to_csv(path, sep=sep, index=False)


def main():
    args = parse_args()
    df = read_table(Path(args.input))
    print(f"[info] loaded {len(df)} rows from {args.input}")

    print(f"[info] loading dataset {args.dataset} [{args.split}] for ground truth")
    gold_lookup = load_gold_lookup(args.dataset, args.split)

    scored = score_rows(df, gold_lookup)
    write_table(scored, Path(args.output))
    print(f"[info] saved scored table -> {args.output}\n")

    variants = scored["prompt_variant"].unique()
    for variant in variants:
        v_scored = scored[scored["prompt_variant"] == variant]
        print(f"\n{'#'*70}\n# prompt_variant = {variant}\n{'#'*70}\n")
        run_group(v_scored, CONDITIONS, "All conditions")

        n_malformed = v_scored["malformed"].sum()
        n_suspect_tp = ((v_scored["detected"] & v_scored["gt_positive"] & v_scored["suspect"])).sum()
        print(f"[note] {n_malformed} malformed (non-N/A, unparseable) outputs; "
              f"{n_suspect_tp} true positives excluded from Tier 2 (suspect ground truth)")


if __name__ == "__main__":
    main()
