#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 0: turn an annotation table into the gold table the evaluator consumes.
-- Hongxu Zhou, 2026

The data synthesis script only has to produce three columns:

    id            unique item identifier
    sentence      the disfluent utterance, byte-identical to what the parser
                  will be fed
    sbn_repair    the gold repair-aware SBN, on a single line

Everything else the evaluation needs is DERIVED here, never annotated:

    penman_repair   gold graph in Penman (reference for the repair-aware score)
    penman_clean    gold graph with the repair stripped (reference for Metric B)
    partition       core | challenge
    strip_reason    which gate sent it to the challenge set
    n_repairs       intra-turn repairs found
    gold_status     ok | parse_error | strip_error

Deriving rather than annotating `penman_clean` is the point: gold and
prediction then pass through byte-identical code, so any difference between
them is the parser's, not two annotators'. Annotating a second "clean" SBN by
hand would add an inconsistency source the metric cannot separate out.

`partition` is likewise computed here, not at scoring time, so the core/
challenge split is a stable, reviewable property of the dataset. If the
stripper's gates ever change, that shows up as a diff in this table -- which is
what you want -- instead of silently moving items between metrics.

Any other columns in the input (repair_type, device, has_interregnum,
condition, ...) are passed through untouched and are what the analysis notebook
slices on.

Usage:
    python3 build_gold.py -i annotations.tsv -o gold.parquet
    python3 build_gold.py -i annotations.tsv -o gold.parquet --strict
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sbn_env import ensure_on_path  # noqa: E402

ensure_on_path()

from sbn_smatch import SBNGraph      # noqa: E402
from sbn_spec import SBNError        # noqa: E402
from repair_strip import strip_repair, StripError, find_repairs  # noqa: E402
from tables import read_table, write_table  # noqa: E402


def process_row(sbn_repair: str) -> dict:
    """Derive every gold column from one repair-aware SBN string."""
    out = {
        "gold_status": "ok",
        "partition": "",
        "strip_reason": "",
        "n_repairs": 0,
        "penman_repair": "",
        "penman_clean": "",
    }

    text = (sbn_repair or "").strip()
    if not text:
        out["gold_status"] = "parse_error"
        out["strip_reason"] = "empty SBN"
        return out

    try:
        graph = SBNGraph().from_string(text, is_single_line=True)
        out["penman_repair"] = graph.to_penman_string()
    except (SBNError, Exception) as e:  # noqa: B014 - SBNError is an Exception
        out["gold_status"] = "parse_error"
        out["strip_reason"] = f"{type(e).__name__}: {e}"
        return out

    out["n_repairs"] = len(find_repairs(graph)[0])

    try:
        res = strip_repair(graph)
    except StripError as e:
        out["gold_status"] = "strip_error"
        out["partition"] = "challenge"
        out["strip_reason"] = f"StripError: {e}"
        return out

    if res.status == "na":
        out["partition"] = "challenge"
        out["strip_reason"] = res.reason
        return out

    out["partition"] = "core"
    try:
        out["penman_clean"] = res.graph.to_penman_string()
    except Exception as e:
        # A clean graph that will not serialise is a stripper bug, not a
        # property of the item; surface it rather than quietly dropping it.
        out["gold_status"] = "strip_error"
        out["partition"] = "challenge"
        out["strip_reason"] = f"clean graph not exportable: {e}"
    return out


def create_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Derive the gold evaluation table from repair-aware annotations.")
    p.add_argument("-i", "--input", required=True,
                   help="Annotation table (.parquet/.tsv/.csv).")
    p.add_argument("-o", "--output", required=True, help="Gold table to write.")
    p.add_argument("--id-col", default="id")
    p.add_argument("--sbn-col", default="sbn_repair")
    p.add_argument("--sentence-col", default="sentence")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if any row fails to parse or strip.")
    return p


def main() -> int:
    args = create_arg_parser().parse_args()
    in_path, out_path = Path(args.input), Path(args.output)
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")

    df = read_table(in_path)
    print(f"Loaded {in_path}: {len(df)} rows, columns {list(df.columns)}")

    for col in (args.id_col, args.sbn_col):
        if col not in df.columns:
            sys.exit(f"Required column '{col}' not in input table.")
    if args.sentence_col not in df.columns:
        print(f"  warning: no '{args.sentence_col}' column; "
              "the evaluator does not need it but the analysis will want it.")

    dupes = df[args.id_col][df[args.id_col].duplicated()].unique()
    if len(dupes):
        sys.exit(f"Duplicate ids in the annotation table: {list(dupes)[:10]}")

    derived = [process_row(str(v)) for v in df[args.sbn_col]]
    out = df.copy()
    for key in ("gold_status", "partition", "strip_reason", "n_repairs",
                "penman_repair", "penman_clean"):
        out[key] = [d[key] for d in derived]

    status = Counter(out["gold_status"])
    part = Counter(out["partition"])
    with_repair = int((out["n_repairs"] > 0).sum())

    print("\n" + "=" * 60)
    print("GOLD TABLE")
    print("=" * 60)
    print(f"rows                 {len(out)}")
    print(f"with a repair        {with_repair}")
    print(f"status               {dict(status)}")
    print(f"partition            {dict(part)}")
    if part.get("challenge"):
        print("\nchallenge-set reasons (Metric A only):")
        reasons = Counter(
            r.split(";")[0][:70]
            for r in out.loc[out["partition"] == "challenge", "strip_reason"]
        )
        for reason, n in reasons.most_common():
            print(f"  {n:>4}  {reason}")
    print("=" * 60)

    write_table(out, out_path)
    print(f"\nSaved -> {out_path}")

    failed = status.get("parse_error", 0) + status.get("strip_error", 0)
    if args.strict and failed:
        print(f"\n--strict: {failed} row(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
