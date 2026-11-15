#!/usr/bin/env python3
"""Check C–C false positives in basic scoring results for hard negatives.

Input: JSONL from score_basic.py for the hard-negative slice, one run at a
time. Output: one verdict per expected row and a summary per condition.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from formal_graph import Graph, InvalidGraph
from score_basic import require_panel
from tables import read_table


def check_rows(basic_rows):
    """Use basic scoring's validity verdict; inspect only valid predictions."""
    results = []
    seen = set()
    for row in basic_rows:
        key = (row["example_id"], row["condition"])
        if key in seen:
            raise ValueError(f"Duplicate basic scoring row: {key!r}")
        seen.add(key)
        # Reopen the exact SBN saved by basic scoring to inspect its boxes.
        # This step does not need the gold graph or a SMATCH++ alignment.
        pred_status = row["pred_status"]
        sbn = row["pred_sbn"]
        false_positive = None
        status, reason = "not_run", pred_status
        if pred_status == "valid":
            try:
                graph = Graph(sbn)
            except InvalidGraph as exc:
                raise ValueError(f"Previously valid prediction {key!r} cannot be read: {exc}") from exc
            false_positive = any(graph.pairs())
            status, reason = "checked", None
        results.append(dict(example_id=key[0], condition=key[1], pred_status=pred_status,
                            status=status, reason=reason, false_positive=false_positive))
    return results


def summarise(results):
    """Report the numerator and denominator together, separately by condition."""
    groups = defaultdict(list)
    for row in results:
        groups[row["condition"]].append(row)
    summary = {}
    for condition, rows in sorted(groups.items()):
        checked = sum(r["status"] == "checked" for r in rows)
        false_positives = sum(r["false_positive"] is True for r in rows)
        summary[condition] = dict(expected=len(rows), checked=checked,
                                  false_positives=false_positives,
                                  false_positive_rate=false_positives / checked if checked else None)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basic", required=True, type=Path, help="hard-negative basic scoring JSONL")
    parser.add_argument("--references", required=True, type=Path, help="complete slice reference table")
    parser.add_argument("--output", required=True, type=Path, help="new JSON report (items and summary)")
    args = parser.parse_args()
    try:
        with args.basic.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
        require_panel(rows, read_table(args.references, keep_default_na=False).to_dict("records"))
        results = check_rows(rows)
        report = dict(summary=summarise(results), items=results)
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(report, target, ensure_ascii=False, indent=2)
            target.write("\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
