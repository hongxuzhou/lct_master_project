#!/usr/bin/env python3
"""Check prediction graph validity and score full graphs against selected gold.

The reference table lists every expected (example_id, condition) and the
gold SBN to use. The prediction table supplies the model's SBN for those
same keys. This script does not decide which gold to use for a condition.

Run from the repository root:
    python evaluation/score_basic.py --references refs.tsv --predictions preds.tsv --output basic.jsonl

The output has one JSON object per expected row, even when the prediction
is missing. Read pred_status for graph validity and score_status to tell
whether the metrics were computed, penalised, or unavailable.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from formal_graph import Graph, InvalidGraph, full_score, make_scorer
from tables import read_table

KEY = ("example_id", "condition")


def index_rows(rows, name):
    """Look up rows by ID and condition, independently of file order."""
    # Reject duplicates instead of silently choosing one of two predictions.
    indexed = {}
    for row in rows:
        key = tuple(row[k] for k in KEY)
        if any(not isinstance(v, str) or not v.strip() for v in key):
            raise ValueError(f"{name}: empty or non-string key {key!r}")
        if key in indexed:
            raise ValueError(f"{name}: duplicate key {key!r}")
        indexed[key] = row
    return indexed


def parse(sbn, cache=None):
    """Return (graph, status, reason). An unparseable SBN has no graph."""
    # The same reference SBN may be used by multiple conditions. Reuse its
    # parsed graph (or its error) rather than reading it again for each row.
    if cache is not None and isinstance(sbn, str) and sbn in cache:
        return cache[sbn]
    try:
        result = Graph(sbn), "valid", None
    except InvalidGraph as exc:
        result = None, "invalid", str(exc)
    except Exception as exc:
        # Unexpected reader errors are not evidence of invalid model output.
        result = None, "scorer_error", f"{type(exc).__name__}: {exc}"
    if cache is not None and isinstance(sbn, str):
        cache[sbn] = result
    return result


def score_rows(references, predictions, scorer):
    """Yield one result per reference, in reference-table order."""
    refs = index_rows(references, "references")
    preds = index_rows(predictions, "predictions")
    # Missing predictions are scored as incomplete below. A prediction for
    # an unknown key is an input mismatch, so stop before scoring any rows.
    extra = preds.keys() - refs.keys()
    if extra:
        raise ValueError(f"prediction has no matching reference key: {min(extra)!r}")
    cache = {}
    for key, ref in refs.items():
        gold, gold_status, gold_reason = parse(ref["gold_sbn"], cache)
        pred_sbn = preds[key]["pred_sbn"] if key in preds else None
        pred, pred_status, pred_reason = (parse(pred_sbn) if key in preds
                                         else (None, "missing", "prediction row absent"))
        # Check the gold and prediction separately so both problems remain
        # visible if both inputs are invalid.
        metrics = dict(precision=None, recall=None, f1=None)
        score_reason = None
        # Score policy: unusable gold or scorer error -> no metric.
        # Usable gold but missing/invalid prediction -> F1 penalty only.
        # Two valid graphs -> precision, recall and F1 from one alignment.
        if gold_status != "valid":
            score_status = "gold_error" if gold_status == "invalid" else "scorer_error"
            score_reason = gold_reason
        elif pred_status in ("missing", "invalid"):
            score_status = "incomplete"
            metrics["f1"] = 0.0
        elif pred_status == "scorer_error":
            score_status, score_reason = pred_status, pred_reason
        else:
            try:
                metrics = full_score(scorer, gold, pred)
                score_status = "scored"
            except Exception as exc:
                score_status = "scorer_error"
                score_reason = f"{type(exc).__name__}: {exc}"
        yield dict(zip(KEY, key)) | dict(
            gold_status=gold_status, gold_reason=gold_reason,
            pred_status=pred_status, pred_reason=pred_reason,
            score_status=score_status, score_reason=score_reason, **metrics)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--references", required=True, type=Path, help="example_id, condition, gold_sbn")
    cli.add_argument("--predictions", required=True, type=Path, help="example_id, condition, pred_sbn")
    cli.add_argument("--output", required=True, type=Path, help="new JSONL file")
    args = cli.parse_args()

    def read(path, column):
        # Preserve literal strings such as 'NA' and distinguish blank cells
        # from missing prediction rows.
        table = read_table(path, keep_default_na=False)
        missing = set((*KEY, column)) - set(table)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return table.to_dict("records")

    try:
        records = list(score_rows(read(args.references, "gold_sbn"),
                                  read(args.predictions, "pred_sbn"), make_scorer()))
        # Create a new result file. Do not replace any input or earlier run.
        with args.output.open("x", encoding="utf-8") as out:
            for row in records:
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    except (OSError, ValueError) as exc:
        cli.error(str(exc))
    print(f"{len(records)} rows written to {args.output}")
    for field in ("pred_status", "score_status"):
        print(f"{field}: {dict(Counter(r[field] for r in records))}")


if __name__ == "__main__":
    main()
