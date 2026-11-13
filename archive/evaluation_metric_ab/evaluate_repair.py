#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2: score repair-aware parses on Metric A (structure) and Metric B (semantics).
-- Hongxu Zhou, 2026

Input
-----
predictions : LONG table, one row per (id, condition)
                  id, [condition], pred_sbn
gold        : the table `build_gold.py` produced
                  id, sbn_repair, penman_repair, penman_clean, partition

Output = the prediction rows plus:

    parse_status    success | ill_formed | parse_error
    a_*             Metric A. Raw counts are kept alongside the per-item F1s
                    because micro-averaging needs them; the notebook can then
                    re-aggregate over any slice without re-running scoring.
    b_status/b_f1   Metric B, on the core partition only.
    full_status     the plain repair-aware graph score, gold vs prediction
    full_f1         with no stripping at all -- the obvious baseline number,
                    reported so nobody has to ask what it would have been.

Metric B failure policy: a prediction that will not parse, or whose own
structure falls outside the stripper's domain, gets `b_f1 = NaN` and a status
saying which. The summary reports both means -- over successes only, and
penalised (failures counted as 0). Quote the penalised one as the headline;
the success-only mean silently rewards a parser for failing loudly.

Usage:
    python3 evaluate_repair.py -i preds_long.parquet -g gold.parquet \\
        -o preds_scored.parquet --solver ilp
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sbn_env import ensure_on_path  # noqa: E402

ensure_on_path()

from sbn_smatch import SBNGraph                                   # noqa: E402
from sbn_spec import SBNError                                     # noqa: E402
from repair_strip import strip_repair, StripError                 # noqa: E402
from repair_metrics import ItemScore, score_item, aggregate, format_summary  # noqa: E402
from tables import read_table, write_table                        # noqa: E402

try:
    from smatchpp import Smatchpp, solvers
    from smatchpp.formalism.generic.tools import GenericStandardizer
except ImportError as e:
    sys.exit(f"smatchpp not installed: {e}\nInstall with: pip install smatchpp")


# ── scoring helpers ──────────────────────────────────────────────────────────

def make_scorer(solver_name: str = "ilp") -> Smatchpp:
    """Build a Smatchpp scorer.

    GenericStandardizer is mandatory, not a preference: without it smatch++
    does not de-invert `:X-of`, so every inverted role -- which is how the
    notation avoids illegal positive indices into a CORRECTION box -- is scored
    as a wholly different edge. See colloquium_prep/pilot_eval/evaluate_smatchpp.py
    for the measured effect.
    """
    solver_name = solver_name.lower()
    if solver_name == "ilp":
        alignment_solver = solvers.ILP()
    elif solver_name in ("hillclimber", "hc"):
        alignment_solver = solvers.HillClimber()
    else:
        raise ValueError(f"Unknown solver '{solver_name}' (use 'ilp' or 'hillclimber').")
    return Smatchpp(alignmentsolver=alignment_solver,
                    graph_standardizer=GenericStandardizer())


def parse_sbn(sbn_string: str):
    """Return (graph, None) or (None, 'ill_formed' | 'parse_error')."""
    try:
        return SBNGraph().from_string(sbn_string, is_single_line=True), None
    except SBNError as e:
        msg = str(e)
        if "ill-formed" in msg or "Strict" in msg:
            return None, "ill_formed"
        return None, "parse_error"
    except Exception:
        return None, "parse_error"


def to_penman(graph):
    try:
        return graph.to_penman_string(), None
    except Exception as e:
        return None, f"penman_error: {type(e).__name__}"


def smatch_f1(scorer, gold_penman: str, pred_penman: str):
    try:
        f1 = scorer.score_pair(gold_penman, pred_penman)["main"]["F1"]
        return "success", round(float(f1) / 100.0, 6)
    except Exception:
        return "smatch_error", None


# ── driver ───────────────────────────────────────────────────────────────────

def evaluate(preds: pd.DataFrame, gold: pd.DataFrame, *, id_col, pred_col,
             scorer) -> pd.DataFrame:
    gold_by_id = gold.set_index(id_col).to_dict("index")
    gold_graph_cache = {}

    rows, scores = [], []
    total = len(preds)

    for n, (_, row) in enumerate(preds.iterrows(), 1):
        rid = row[id_col]
        g = gold_by_id.get(rid)
        rec = {"parse_status": "success", "partition": "",
               "b_status": "", "b_f1": None,
               "full_status": "", "full_f1": None}

        if g is None:
            rec["parse_status"] = "no_gold"
            rows.append(rec)
            scores.append(ItemScore())
            continue

        rec["partition"] = g.get("partition", "")

        # gold graph, parsed once per id (Metric A needs the graph, not Penman)
        if rid not in gold_graph_cache:
            gold_graph_cache[rid], _ = parse_sbn(str(g.get("sbn_repair", "")).strip())
        gold_graph = gold_graph_cache[rid]

        pred_graph, perr = parse_sbn(str(row[pred_col]).strip())
        if perr:
            rec["parse_status"] = perr

        # ── Metric A: runs on every item, challenge set included ──
        score = score_item(gold_graph, pred_graph)
        scores.append(score)

        # ── full repair-aware graph, no stripping (baseline) ──
        gold_full = str(g.get("penman_repair", "") or "")
        if pred_graph is None:
            rec["full_status"] = "pred_parse_error"
        elif not gold_full:
            rec["full_status"] = "gold_error"
        else:
            pred_full, ferr = to_penman(pred_graph)
            if ferr:
                rec["full_status"] = ferr
            else:
                rec["full_status"], rec["full_f1"] = smatch_f1(
                    scorer, gold_full, pred_full)

        # ── Metric B: core partition only ──
        gold_clean = str(g.get("penman_clean", "") or "")
        if rec["partition"] != "core" or not gold_clean:
            rec["b_status"] = "gold_challenge"
        elif pred_graph is None:
            rec["b_status"] = "pred_parse_error"
        else:
            try:
                stripped = strip_repair(pred_graph)
            except StripError as e:
                stripped = None
                rec["b_status"] = f"pred_strip_error: {type(e).__name__}"
            if stripped is not None:
                if stripped.status == "na":
                    rec["b_status"] = "pred_na"
                else:
                    pred_clean, cerr = to_penman(stripped.graph)
                    if cerr:
                        rec["b_status"] = cerr
                    else:
                        rec["b_status"], rec["b_f1"] = smatch_f1(
                            scorer, gold_clean, pred_clean)

        rows.append(rec)
        if n % 200 == 0:
            print(f"  {n:>5}/{total} rows scored...", flush=True)

    out = preds.copy().reset_index(drop=True)
    extra = pd.DataFrame(rows)
    for col in extra.columns:
        out[col] = extra[col]

    # Metric A columns, raw counts first so any slice can be re-aggregated.
    out["a_gold_n"] = [s.gold_n for s in scores]
    out["a_pred_n"] = [s.pred_n for s in scores]
    out["a_detected"] = [s.pred_n > 0 for s in scores]
    for field in ("token_tp", "token_pred", "token_gold",
                  "span_tp", "span_pred", "span_gold",
                  "merge_correct", "merge_total"):
        out[f"a_{field}"] = [getattr(s, field) for s in scores]
    out["a_token_f1"] = pd.array([s.token_f1 for s in scores], dtype="Float64")
    out["a_span_f1"] = pd.array([s.span_f1 for s in scores], dtype="Float64")
    out["b_f1"] = pd.array(out["b_f1"], dtype="Float64")
    out["full_f1"] = pd.array(out["full_f1"], dtype="Float64")

    out.attrs["scores"] = scores
    return out


def scores_from_frame(df: pd.DataFrame) -> list:
    """Rebuild ItemScore objects from the stored raw counts, for re-aggregation."""
    return [
        ItemScore(
            gold_n=int(r["a_gold_n"]), pred_n=int(r["a_pred_n"]),
            token_tp=int(r["a_token_tp"]), token_pred=int(r["a_token_pred"]),
            token_gold=int(r["a_token_gold"]), span_tp=int(r["a_span_tp"]),
            span_pred=int(r["a_span_pred"]), span_gold=int(r["a_span_gold"]),
            merge_correct=int(r["a_merge_correct"]),
            merge_total=int(r["a_merge_total"]),
        )
        for _, r in df.iterrows()
    ]


def print_summary(df: pd.DataFrame, condition_col: str) -> None:
    groups = [("<all>", df)]
    if condition_col in df.columns and df[condition_col].nunique() > 1:
        groups += [(c, df[df[condition_col] == c])
                   for c in df[condition_col].unique()]

    for name, sub in groups:
        print(f"\n### condition: {name}   (n = {len(sub)})")
        print(format_summary(aggregate(scores_from_frame(sub))))

        core = sub[sub["partition"] == "core"]
        ok = core["b_f1"].notna()
        n_core = len(core)
        mean_success = core.loc[ok, "b_f1"].mean() if ok.any() else float("nan")
        penalised = core["b_f1"].fillna(0.0).mean() if n_core else float("nan")
        print("METRIC B -- clean semantics (core partition)")
        print(f"  scoreable      {int(ok.sum())}/{n_core}")
        print(f"  mean F1        {mean_success:.4f}  (successes only)")
        print(f"  mean F1        {penalised:.4f}  (penalised: failures = 0)  <- headline")
        full_ok = sub["full_f1"].notna()
        if full_ok.any():
            print(f"  repair-aware   {sub.loc[full_ok, 'full_f1'].mean():.4f} "
                  f"(no stripping, baseline)")
        print(f"  parse status   {dict(Counter(sub['parse_status']))}")
        print(f"  b status       {dict(Counter(sub['b_status']))}")


def create_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score repair-aware SBN predictions on Metric A and Metric B.")
    p.add_argument("-i", "--input", required=True, help="LONG predictions table.")
    p.add_argument("-g", "--gold", required=True, help="Gold table from build_gold.py.")
    p.add_argument("-o", "--output", required=True, help="Scored table to write.")
    p.add_argument("--id-col", default="id")
    p.add_argument("--condition-col", default="condition")
    p.add_argument("--pred-col", default="pred_sbn")
    p.add_argument("--solver", default="ilp", choices=["ilp", "hillclimber", "hc"])
    return p


def main() -> int:
    args = create_arg_parser().parse_args()
    in_path, gold_path, out_path = Path(args.input), Path(args.gold), Path(args.output)
    for path in (in_path, gold_path):
        if not path.exists():
            sys.exit(f"Not found: {path}")

    preds = read_table(in_path)
    gold = read_table(gold_path)
    print(f"predictions {in_path}: {len(preds)} rows")
    print(f"gold        {gold_path}: {len(gold)} rows")

    for col in (args.id_col, args.pred_col):
        if col not in preds.columns:
            sys.exit(f"Required column '{col}' not in the predictions table.")
    for col in (args.id_col, "sbn_repair", "penman_repair", "penman_clean", "partition"):
        if col not in gold.columns:
            sys.exit(f"Gold table is missing '{col}'. Produce it with build_gold.py.")

    scored = evaluate(preds, gold, id_col=args.id_col, pred_col=args.pred_col,
                      scorer=make_scorer(args.solver))
    print_summary(scored, args.condition_col)

    write_table(scored, out_path)
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
