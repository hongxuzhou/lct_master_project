#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smatch++ evaluation toolkit for the self-repair parsing pilot study.
-- Hongxu Zhou, 2026

Replaces the legacy `evaluate_repair_smatch.py` (which used the bundled
`smatch.score_amr_pairs`) with flipz357/smatchpp, matching current PMB practice.

Pipeline per (gold_sbn, pred_sbn) pair:
    SBN string --[sbn_smatch.SBNGraph.to_penman_string]--> Penman
    (gold_penman, pred_penman) --[smatchpp]--> F1

Input  : LONG-format table, one row = one (id, condition) prediction.
         Required columns (names configurable):
             --id-col        default: id
             --condition-col default: condition
             --pred-col      default: pred_sbn
         Gold SBN comes from EITHER:
             --gold-col      a column in the same table (inline gold), OR
             --gold-file     a separate table mapping id -> gold SBN
                             (use --gold-file-id-col / --gold-file-gold-col)
Output : same rows + two columns:
             status : success | ill_formed | parse_error | gold_error | smatch_error
             f1     : float in [0,1] when status == success, else NaN

Both .parquet and .tsv/.csv are accepted (inferred from extension).

Usage:
    python3 evaluate_smatchpp.py \
        -i preds_long.parquet -o preds_long_scored.parquet \
        --gold-file pilot_dataset.parquet --gold-file-gold-col mr \
        --solver ilp

The folder `sbn_lib/` (sbn_smatch.py, sbn_spec.py, graph_base.py,
penman_model.py) must sit next to this script. Requires: smatchpp, penman,
networkx, pandas (+ pyarrow for parquet).
"""

import os
import sys
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

# Make the bundled SBN modules importable regardless of CWD.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sbn_lib"))

try:
    from sbn_smatch import SBNGraph
    from sbn_spec import SBNError
except ImportError as e:
    sys.exit(f"Import error: {e}\nMake sure ./sbn_lib/ sits next to this script.")

try:
    from smatchpp import Smatchpp, solvers
except ImportError as e:
    sys.exit(f"smatchpp not installed: {e}\nInstall with: pip install smatchpp")


# ── SBN -> Penman ────────────────────────────────────────────────────────────

def sbn_to_penman(sbn_str: str):
    """Parse a one-line SBN and convert to Penman.

    Returns (penman_str, None) on success, else (None, err) where err is
    'ill_formed' (Strict-mode rejection of a possibly ill-formed graph) or
    'parse_error' (any other failure).
    """
    try:
        penman = SBNGraph().from_string(sbn_str, is_single_line=True).to_penman_string()
        return penman, None
    except SBNError as e:
        msg = str(e)
        if "ill-formed" in msg or "Strict" in msg:
            return None, "ill_formed"
        return None, "parse_error"
    except Exception:
        return None, "parse_error"


# ── Scoring ──────────────────────────────────────────────────────────────────

def make_scorer(solver_name: str = "ilp") -> Smatchpp:
    """Build a Smatchpp scorer. 'ilp' = exact (recommended); 'hillclimber' = fast approx."""
    solver_name = solver_name.lower()
    if solver_name == "ilp":
        return Smatchpp(alignmentsolver=solvers.ILP())
    if solver_name in ("hillclimber", "hc"):
        return Smatchpp(alignmentsolver=solvers.HillClimber())
    raise ValueError(f"Unknown solver '{solver_name}' (use 'ilp' or 'hillclimber').")


def score_penman_pair(scorer: Smatchpp, gold_penman: str, pred_penman: str):
    """Return (status, f1_in_0_1). f1 is None unless status == 'success'."""
    try:
        f1_0_100 = scorer.score_pair(gold_penman, pred_penman)["main"]["F1"]
        return "success", round(float(f1_0_100) / 100.0, 6)
    except Exception:
        return "smatch_error", None


# ── Driver ───────────────────────────────────────────────────────────────────

def evaluate_frame(df: pd.DataFrame, *, gold_lookup, id_col, pred_col, scorer):
    """Score every row. `gold_lookup` maps id -> gold SBN string.

    Gold SBNs are converted to Penman ONCE per id (cached); a gold that fails
    conversion marks every row with that id as 'gold_error'.
    """
    gold_penman_cache = {}  # id -> (penman | None, err | None)
    statuses, f1s = [], []
    total = len(df)

    for n, (_, row) in enumerate(df.iterrows(), 1):
        rid = row[id_col]

        if rid not in gold_penman_cache:
            gold_sbn = gold_lookup.get(rid)
            if gold_sbn is None or str(gold_sbn).strip() in ("", "nan"):
                gold_penman_cache[rid] = (None, "gold_error")
            else:
                gp, gerr = sbn_to_penman(str(gold_sbn).strip())
                gold_penman_cache[rid] = (gp, "gold_error" if gerr else None)
        gold_penman, gold_err = gold_penman_cache[rid]

        if gold_err:
            statuses.append("gold_error"); f1s.append(None); continue

        pred_sbn = str(row[pred_col]).strip()
        pred_penman, pred_err = sbn_to_penman(pred_sbn)
        if pred_err:
            statuses.append(pred_err); f1s.append(None); continue

        status, f1 = score_penman_pair(scorer, gold_penman, pred_penman)
        statuses.append(status); f1s.append(f1)

        if n % 200 == 0:
            print(f"  {n:>5}/{total} rows scored...", flush=True)

    out = df.copy()
    out["status"] = statuses
    out["f1"] = pd.array(f1s, dtype="Float64")
    return out


def print_summary(df: pd.DataFrame, condition_col: str):
    """Per-condition success rate + mean/median F1 (over successes only)."""
    print("\n" + "=" * 64)
    print("SUMMARY (by condition)")
    print("=" * 64)
    print(f"{'condition':<24}{'n':>6}{'success%':>10}{'meanF1':>9}{'medF1':>8}")
    print("-" * 64)
    group_keys = df[condition_col].unique() if condition_col in df.columns else ["<all>"]
    for cond in group_keys:
        sub = df if condition_col not in df.columns else df[df[condition_col] == cond]
        n = len(sub)
        ok = (sub["status"] == "success").sum()
        valid = sub.loc[sub["status"] == "success", "f1"]
        pct = 100 * ok / n if n else 0
        mean = f"{valid.mean():.4f}" if len(valid) else "N/A"
        med = f"{valid.median():.4f}" if len(valid) else "N/A"
        print(f"{str(cond):<24}{n:>6}{pct:>9.1f}%{mean:>9}{med:>8}")
    print("-" * 64)
    print(f"overall status breakdown: {dict(Counter(df['status']))}")
    print("=" * 64)


# ── IO helpers ───────────────────────────────────────────────────────────────

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    sep = "\t" if path.suffix in (".tsv", ".txt") else ","
    return pd.read_csv(path, sep=sep, dtype=str)


def write_table(df: pd.DataFrame, path: Path):
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        sep = "\t" if path.suffix in (".tsv", ".txt") else ","
        df.to_csv(path, sep=sep, index=False)


def build_gold_lookup(args, preds_df) -> dict:
    if args.gold_col:
        if args.gold_col not in preds_df.columns:
            sys.exit(f"--gold-col '{args.gold_col}' not in input table.")
        return dict(zip(preds_df[args.id_col], preds_df[args.gold_col]))
    if args.gold_file:
        g = read_table(Path(args.gold_file))
        if args.gold_file_id_col not in g.columns or args.gold_file_gold_col not in g.columns:
            sys.exit("--gold-file is missing the id/gold columns "
                     f"('{args.gold_file_id_col}', '{args.gold_file_gold_col}').")
        return dict(zip(g[args.gold_file_id_col], g[args.gold_file_gold_col]))
    sys.exit("Provide gold via --gold-col (inline) or --gold-file (separate table).")


def create_arg_parser():
    p = argparse.ArgumentParser(description="Smatch++ evaluation for repair SBN predictions.")
    p.add_argument("-i", "--input", required=True, help="LONG-format predictions table.")
    p.add_argument("-o", "--output", required=True, help="Output table (.parquet/.tsv/.csv).")
    p.add_argument("--id-col", default="id")
    p.add_argument("--condition-col", default="condition")
    p.add_argument("--pred-col", default="pred_sbn")
    p.add_argument("--gold-col", default=None, help="Inline gold SBN column in the input table.")
    p.add_argument("--gold-file", default=None, help="Separate table mapping id -> gold SBN.")
    p.add_argument("--gold-file-id-col", default="id")
    p.add_argument("--gold-file-gold-col", default="mr")
    p.add_argument("--solver", default="ilp", choices=["ilp", "hillclimber", "hc"])
    return p


if __name__ == "__main__":
    args = create_arg_parser().parse_args()

    in_path, out_path = Path(args.input), Path(args.output)
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")

    print(f"Loading {in_path} ...")
    df = read_table(in_path)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")
    for c in (args.id_col, args.pred_col):
        if c not in df.columns:
            sys.exit(f"Required column '{c}' not in input table.")

    gold_lookup = build_gold_lookup(args, df)
    print(f"Gold lookup: {len(gold_lookup)} ids. Solver: {args.solver}")

    scorer = make_scorer(args.solver)
    scored = evaluate_frame(df, gold_lookup=gold_lookup, id_col=args.id_col,
                            pred_col=args.pred_col, scorer=scorer)
    print_summary(scored, args.condition_col)

    write_table(scored, out_path)
    print(f"\nSaved -> {out_path}")
