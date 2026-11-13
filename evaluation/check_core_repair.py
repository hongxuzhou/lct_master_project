#!/usr/bin/env python3
"""Check local repair on core C3, or audit gold coverage before inference."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq

from formal_gold import GoldError, NeedsReview, annotated_pattern, core_pattern, validate_pattern
from formal_graph import Graph, InvalidGraph
from formal_local import Result, check_repair
from score_basic import index_rows, parse

CORE_VARIANTS = {"rsr_base", "rsr_editing_expression"}


def prepare_gold(row, annotation):
    """Return a checked pattern, or explain why the gold needs review."""
    try:
        graph = Graph(row["sbn_repair"])
        if annotation is not None:
            digest = hashlib.sha256(row["sbn_repair"].encode()).hexdigest()
            if annotation.get("gold_sha256") != digest:
                raise GoldError("reviewed pattern targets a different gold graph")
            return annotated_pattern(graph, annotation), Result("ready")
        return validate_pattern(graph, core_pattern(graph, row)), Result("ready")
    except NeedsReview as exc:
        return None, Result("review", f"gold coverage: {exc}")
    except (GoldError, InvalidGraph, KeyError) as exc:
        return None, Result("review", f"gold error: {exc}")
    except Exception as exc:
        return None, Result("review", f"scorer error: {type(exc).__name__}: {exc}")


def check_rows(gold_rows, basic_rows=None, annotations=None):
    """The gold defines the panel; a missing basic row is an input mismatch."""
    gold = index_rows((dict(r, condition="C3") for r in gold_rows
                       if r["variant"] in CORE_VARIANTS), "core gold")
    if not gold:
        raise ValueError("No core RSR rows in gold")
    annotations = annotations or {}
    if set(annotations) - {key[0] for key in gold}:
        raise ValueError("Reviewed pattern ID absent from core gold")
    basic = None if basic_rows is None else index_rows(basic_rows, "basic results")
    if basic is not None and basic.keys() != gold.keys():
        missing = sorted(gold.keys() - basic.keys())
        extra = sorted(basic.keys() - gold.keys())
        raise ValueError(f"Expected the full core C3 basic panel; missing {len(missing)}, "
                         f"extra {len(extra)} (examples: {missing[:2]}, {extra[:2]})")
    for key, row in gold.items():
        pattern, coverage = prepare_gold(row, annotations.get(key[0]))
        item = {k: row[k] for k in ("example_id", "item_id", "variant", "repair_type")}
        item.update(condition="C3", gold_pattern=asdict(coverage))
        if basic is None:
            yield item
            continue

        # C–C presence needs only a valid prediction. Local correctness
        # additionally needs a usable gold pattern.
        prediction = basic[key]
        status = prediction["pred_status"]
        item.update(pred_status=status, cc_present=None)
        result = coverage
        if status == "valid":
            graph, parsed, reason = parse(prediction["pred_sbn"])
            if parsed == "valid":
                item["cc_present"] = any(graph.pairs())
                if pattern is not None:
                    try:
                        result, _ = check_repair(graph, pattern)
                    except Exception as exc:
                        result = Result("review", f"scorer error: {type(exc).__name__}: {exc}")
            else:
                result = Result("review", f"could not reread valid prediction: {reason}")
        elif pattern is not None:
            result = (Result("incomplete", status) if status in {"invalid", "missing"}
                      else Result("review", "basic scoring reader error"))
        item["repair"] = asdict(result)
        yield item


def rate(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                rate=numerator / denominator if denominator else None)


def summarise(items, scored):
    coverage = Counter(r["gold_pattern"]["status"] for r in items)
    result = dict(expected=len(items), gold_coverage=dict(coverage),
                  gold_review_reasons=dict(Counter(r["gold_pattern"]["reason"] for r in items
                                                   if r["gold_pattern"]["status"] == "review")))
    by_type = {}
    for row in items:
        by_type.setdefault(row["repair_type"], Counter())[row["gold_pattern"]["status"]] += 1
    result["gold_coverage_by_type"] = {kind: dict(counts) for kind, counts in sorted(by_type.items())}
    if scored:
        counts = Counter(r["repair"]["status"] for r in items)
        detected = [r for r in items if r["cc_present"] is True]
        result.update(
            repair_counts=dict(counts),
            cc_detection=rate(len(detected), sum(r["cc_present"] is not None for r in items)),
            correct_among_detections=rate(
                sum(r["repair"]["status"] == "pass" for r in detected),
                sum(r["repair"]["status"] in {"pass", "fail"} for r in detected)),
            end_to_end=rate(counts["pass"], counts["pass"] + counts["fail"] + counts["incomplete"]),
        )
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--gold", required=True, type=Path, help="published data/test.parquet")
    cli.add_argument("--basic", type=Path, help="full core C3 basic JSONL; omit for gold audit")
    cli.add_argument("--reviewed-repairs", type=Path, help="reviewed gold patterns by example_id")
    cli.add_argument("--output", required=True, type=Path, help="new JSON report")
    args = cli.parse_args()
    try:
        basic = None
        if args.basic:
            with args.basic.open(encoding="utf-8") as source:
                basic = [json.loads(line) for line in source if line.strip()]
        annotations = json.loads(args.reviewed_repairs.read_text()) if args.reviewed_repairs else None
        items = list(check_rows(pq.read_table(args.gold).to_pylist(), basic, annotations))
        report = dict(summary=summarise(items, basic is not None), items=items)
        with args.output.open("x", encoding="utf-8") as out:
            json.dump(report, out, ensure_ascii=False, indent=2)
            out.write("\n")
    except (OSError, ValueError, KeyError) as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    main()
