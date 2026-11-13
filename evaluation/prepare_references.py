#!/usr/bin/env python3
"""Select one test slice and condition and write a reference TSV for score_basic.py."""

import argparse
import csv
from pathlib import Path

import pyarrow.parquet as pq

# A slice identifies a file and, for the shared default file, its input variants.
SLICES = {
    "pmb_fluent": ("data/test.parquet", {"pmb_fluent"}),
    "core_rsr": ("data/test.parquet", {"rsr_base", "rsr_editing_expression"}),
    "hard_negative": ("hard_negative/test.parquet", None),
    "later_reference": ("later_reference/test.parquet", None),
}


def select_references(test_root, slice_name, condition):
    """Return the selected gold column and one reference row per expected input."""
    if condition == "C1" and slice_name in {"core_rsr", "later_reference"}:
        raise ValueError(f"The formal C1 run does not include {slice_name}")
    filename, variants = SLICES[slice_name]
    # C2 normalises core repairs. For later reference, both conditions use
    # each input's own gold, including on the RSR side (which has no sbn_clean).
    gold_column = "sbn_clean" if slice_name == "core_rsr" and condition == "C2" else "sbn_repair"
    columns = ["example_id", gold_column] + (["variant"] if variants else [])
    table = pq.read_table(test_root / filename, columns=columns)

    references = []
    seen = set()
    for row in table.to_pylist():
        if variants and row["variant"] not in variants:
            continue
        example_id = row["example_id"]
        if not isinstance(example_id, str) or not example_id.strip() or example_id in seen:
            raise ValueError(f"Blank or duplicate example_id in selected slice: {example_id!r}")
        seen.add(example_id)
        # Copy the gold as supplied. The scorer reports any invalid gold;
        # this step does not parse, fix, or silently drop it.
        references.append((example_id, condition, row[gold_column]))
    if not references:
        raise ValueError(f"No rows selected for {slice_name}")
    return gold_column, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", required=True, type=Path,
                        help="directory containing data/, hard_negative/ and later_reference/")
    parser.add_argument("--slice", required=True, choices=SLICES)
    parser.add_argument("--condition", required=True, choices=("C1", "C2", "C3"))
    parser.add_argument("--output", required=True, type=Path, help="new reference TSV")
    args = parser.parse_args()
    try:
        gold_column, rows = select_references(args.test_root, args.slice, args.condition)
        # Exclusive creation protects both source files and previous outputs.
        with args.output.open("x", encoding="utf-8", newline="") as out:
            writer = csv.writer(out, delimiter="\t")
            writer.writerow(("example_id", "condition", "gold_sbn"))
            writer.writerows(rows)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{args.slice} / {args.condition}: {len(rows)} references from {gold_column} -> {args.output}")


if __name__ == "__main__":
    main()
