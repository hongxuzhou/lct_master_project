"""
postprocess_truncation.py
──────────────────────────────────────────────────────────────────────────────
Post-processing step for the output of evaluate_repair_smatch.py.

For each repair variant, this script adds one boolean column:

    <variant>_truncated : bool
        True  when is_obvious_truncation() fires on the predicted SBN.
        False otherwise.

The existing *_status and *_f1 columns are left completely untouched.
'Obvious truncation' is a lower-bound proxy: it catches only mid-token cuts
(visible fragments such as 'R', 'Nam', '"Eng').  Silent token-boundary
truncations are out of scope.

Relationship between truncated flag and existing status values:

    status           | truncated=True         | truncated=False
    ─────────────────┼────────────────────────┼───────────────────────────
    parse_error      | mid-token truncation   | genuine parse error
                     | (main Interrug cause)  | (unknown/invalid token)
    ill_formed       | token-boundary trunc.  | structural ill-formedness
                     | (possible but rare)    | (out-of-range index, etc.)
    success          | should not occur *     | normal success
    gold_error       | n/a (gold failed)      | n/a
    smatch_error     | n/a (rare)             | n/a

  * A sequence flagged as obviously truncated would have caused from_string()
    to raise SBNError on the invalid last token, so status=success with
    truncated=True is logically impossible.

Usage
─────
    python3 postprocess_truncation.py -i evaluated.tsv -o evaluated_trunc.tsv

The script expects to be placed in the same directory as:
    truncation_detect.py, sbn_spec.py

-- Hongxu Zhou, 14/May/2026
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from truncation_detect import is_obvious_truncation
except ImportError:
    sys.exit(
        "ImportError: truncation_detect.py not found.\n"
        "Place this file in the same directory as truncation_detect.py."
    )

# ── Configuration — must match evaluate_repair_smatch.py ──────────────────────

# (predicted_sbn_col, status_col, truncated_col)
VARIANTS = [
    ("repair_head_sbn",     "repair_head_status",     "repair_head_truncated"),
    ("repair_mid_sbn",      "repair_mid_status",      "repair_mid_truncated"),
    ("repair_tail_sbn",     "repair_tail_status",     "repair_tail_truncated"),
    ("repair_interrug_sbn", "repair_interrug_status", "repair_interrug_truncated"),
]


# ── Core post-processing ───────────────────────────────────────────────────────

def add_truncation_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add one *_truncated boolean column per variant.
    Rows where the SBN column is missing get False.
    """
    for sbn_col, status_col, trunc_col in VARIANTS:
        if sbn_col not in df.columns:
            print(f"  [WARNING] Column '{sbn_col}' not found — setting {trunc_col} = False.")
            df[trunc_col] = False
            continue

        flags = df[sbn_col].apply(
            lambda cell: is_obvious_truncation(str(cell).strip())
        )
        df[trunc_col] = flags

    return df


def print_truncation_summary(df: pd.DataFrame) -> None:
    """
    Print a cross-variant summary showing, for each variant:
      - total obvious truncations (truncated=True)
      - breakdown of status among truncated rows
      - truncation rate among parse_error rows specifically
    """
    total = len(df)
    print("\n" + "═" * 72)
    print("TRUNCATION SUMMARY  (obvious mid-token truncation, lower-bound proxy)")
    print("═" * 72)

    for sbn_col, status_col, trunc_col in VARIANTS:
        if trunc_col not in df.columns:
            continue

        variant_name = sbn_col.replace("_sbn", "")
        trunc_mask   = df[trunc_col]
        n_trunc      = trunc_mask.sum()
        pct_trunc    = 100 * n_trunc / total if total > 0 else 0

        # Status breakdown among truncated rows
        if status_col in df.columns:
            status_among_trunc = Counter(df.loc[trunc_mask, status_col].tolist())
            # Truncation rate among parse_error rows
            pe_mask   = df[status_col] == "parse_error"
            n_pe      = pe_mask.sum()
            n_pe_trunc = (pe_mask & trunc_mask).sum()
            pe_trunc_pct = 100 * n_pe_trunc / n_pe if n_pe > 0 else 0.0
        else:
            status_among_trunc = {}
            pe_trunc_pct = float("nan")
            n_pe_trunc   = 0
            n_pe         = 0

        print(f"\n  {variant_name}")
        print(f"    Obvious truncations     : {n_trunc:>4} / {total}  ({pct_trunc:.1f}%)")
        print(f"    Status breakdown        : {dict(status_among_trunc)}")
        print(f"    parse_error → truncated : {n_pe_trunc:>4} / {n_pe:<4}  ({pe_trunc_pct:.1f}%)")

    print("\n" + "═" * 72)
    print("NOTE: 'obvious truncation' detects only mid-token cuts (visible")
    print("      fragments).  Silent token-boundary truncations are not counted.")
    print("═" * 72)


# ── CLI ────────────────────────────────────────────────────────────────────────

def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Add *_truncated boolean columns to an already-evaluated repair TSV."
        )
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to input TSV (output of evaluate_repair_smatch.py)"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path to output TSV (input columns preserved, *_truncated columns appended)"
    )
    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path, sep="\t", dtype=str)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    df = add_truncation_flags(df)
    print_truncation_summary(df)

    df.to_csv(output_path, sep="\t", index=False)
    print(f"\nOutput saved to {output_path}")
