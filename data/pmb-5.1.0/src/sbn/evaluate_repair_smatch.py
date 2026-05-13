"""
For my pilot study, evaluating the performance of the byT5 parser on four varieties of PMB SBN gold data.
-- Hongxu Zhou, 13/May/2026

Batch smatch evaluation for repair SBN variants.
For each row in the input TSV, compares the gold SBN (column: mr)
against each of the 4 repair variant SBNs, and records:
  - <variant>_status : 'success' | 'ill_formed' | 'parse_error' | 'gold_error' | 'smatch_error'
  - <variant>_f1     : float (only populated when status == 'success', else NaN)

Usage:
    python3 evaluate_repair_smatch.py -i data.tsv -o data_evaluated.tsv

The script expects to be placed in the same directory as:
    sbn_smatch.py, sbn_spec.py, graph_base.py, penman_model.py,
    smatch.py, smatch_fromlists.py, amr.py, long.py
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

# ── Core imports from the PMB evaluation codebase ─────────────────────────────
try:
    from sbn_smatch import SBNGraph
    from sbn_spec import SBNError
    from smatch import score_amr_pairs
except ImportError as e:
    sys.exit(
        f"Import error: {e}\n"
        "Make sure this script is in the same directory as the PMB evaluation scripts."
    )

# ── Configuration ──────────────────────────────────────────────────────────────

GOLD_COL = "mr"

# (input_sbn_col, output_status_col, output_f1_col)
VARIANTS = [
    ("repair_head_sbn",     "repair_head_status",     "repair_head_f1"),
    ("repair_mid_sbn",      "repair_mid_status",      "repair_mid_f1"),
    ("repair_tail_sbn",     "repair_tail_status",     "repair_tail_f1"),
    ("repair_interrug_sbn", "repair_interrug_status", "repair_interrug_f1"),
]


# ── Core evaluation functions ──────────────────────────────────────────────────

def sbn_to_penman(sbn_str: str):
    """
    Parse a single-line SBN string and convert to Penman notation.

    Returns:
        (penman_str, None)       on success
        (None, 'ill_formed')     when to_penman_string() raises Strict mode error
        (None, 'parse_error')    for any other parsing failure
    """
    try:
        penman = SBNGraph().from_string(sbn_str, is_single_line=True).to_penman_string()
        return penman, None
    except SBNError as e:
        if "ill-formed" in str(e) or "Strict" in str(e):
            return None, "ill_formed"
        return None, "parse_error"
    except Exception:
        return None, "parse_error"


def evaluate_pair(gold_sbn: str, pred_sbn: str):
    """
    Compare a gold SBN against a predicted SBN using smatch.

    Returns:
        (status, f1_score)
        status is one of: 'success', 'ill_formed', 'parse_error',
                          'gold_error', 'smatch_error'
        f1_score is a float when status == 'success', else None
    """
    # Step 1: parse gold
    gold_penman, gold_err = sbn_to_penman(gold_sbn)
    if gold_err:
        return "gold_error", None

    # Step 2: parse predicted
    pred_penman, pred_err = sbn_to_penman(pred_sbn)
    if pred_err:
        return pred_err, None  # 'ill_formed' or 'parse_error'

    # Step 3: smatch scoring
    try:
        for _precision, _recall, f1 in score_amr_pairs([gold_penman], [pred_penman]):
            return "success", round(float(f1), 6)
    except Exception:
        pass

    return "smatch_error", None


# ── Main evaluation loop ───────────────────────────────────────────────────────

def evaluate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterates over all rows and all 4 variants.
    Adds two new columns per variant: <variant>_status and <variant>_f1.
    """
    total_rows = len(df)

    for sbn_col, status_col, f1_col in VARIANTS:

        if sbn_col not in df.columns:
            print(f"  [WARNING] Column '{sbn_col}' not found, skipping.")
            df[status_col] = "missing_column"
            df[f1_col] = float("nan")
            continue

        print(f"\n── Evaluating {sbn_col} ({'─'*40})")
        statuses = []
        f1_scores = []

        for i, row in df.iterrows():
            gold_sbn = str(row[GOLD_COL]).strip()
            pred_sbn = str(row[sbn_col]).strip()
            status, f1 = evaluate_pair(gold_sbn, pred_sbn)
            statuses.append(status)
            f1_scores.append(f1)

            if (i + 1) % 200 == 0:
                print(f"  {i + 1:>4}/{total_rows} rows processed...")

        df[status_col] = statuses
        df[f1_col] = pd.array(f1_scores, dtype="Float64")  # pandas nullable float → NaN for None

        # Per-variant summary
        counts = Counter(statuses)
        success_n = counts.get("success", 0)
        valid_f1 = df.loc[df[status_col] == "success", f1_col]

        print(f"  Status breakdown : {dict(counts)}")
        print(f"  Successfully evaluated : {success_n}/{total_rows} "
              f"({100 * success_n / total_rows:.1f}%)")
        if len(valid_f1) > 0:
            print(f"  Mean F1 (valid pairs)  : {valid_f1.mean():.4f}")
            print(f"  Median F1 (valid pairs): {valid_f1.median():.4f}")

    return df


def print_final_summary(df: pd.DataFrame):
    """Print a compact cross-variant summary table after all variants are done."""
    print("\n" + "═" * 60)
    print("SUMMARY")
    print("═" * 60)
    header = f"{'Variant':<22} {'Success%':>9} {'Mean F1':>9} {'Median F1':>10}"
    print(header)
    print("─" * 60)
    for _, status_col, f1_col in VARIANTS:
        if status_col not in df.columns:
            continue
        total = len(df)
        success_n = (df[status_col] == "success").sum()
        valid_f1 = df.loc[df[status_col] == "success", f1_col]
        pct = 100 * success_n / total if total > 0 else 0
        mean_f1 = f"{valid_f1.mean():.4f}" if len(valid_f1) > 0 else "  N/A"
        median_f1 = f"{valid_f1.median():.4f}" if len(valid_f1) > 0 else "    N/A"
        variant_name = status_col.replace("_status", "")
        print(f"  {variant_name:<20} {pct:>8.1f}% {mean_f1:>9} {median_f1:>10}")
    print("═" * 60)


# ── CLI entry point ────────────────────────────────────────────────────────────

def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate repair SBN variants against PMB gold SBN using smatch."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to input TSV (must contain columns: mr, repair_*_sbn)"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path to output TSV (input columns preserved, new *_status and *_f1 columns appended)"
    )
    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path, sep="\t", dtype=str)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    df = evaluate_dataframe(df)
    print_final_summary(df)

    df.to_csv(output_path, sep="\t", index=False)
    print(f"\nOutput saved to {output_path}")
