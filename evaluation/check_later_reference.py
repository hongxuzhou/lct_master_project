#!/usr/bin/env python3
"""Check later reference after basic scoring, using each input's own gold."""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import pyarrow.parquet as pq

from formal_gold import GoldError, NeedsReview, later_pattern, later_spec, validate_pattern
from formal_graph import Graph, InvalidGraph
from formal_local import Result, check_repair, inspect_reference, joint


def assess(row, gold):
    """Keep C–C presence separate from member identity and joint correctness."""
    condition = row["condition"]
    side = gold["language_condition"]
    if condition not in ("C2", "C3") or side not in ("fluent", "rsr"):
        raise ValueError(f"Unsupported condition or later-reference side: {condition}, {side}")
    combined = condition == "C3" and side == "rsr"
    cc_expected = True if combined else False if side == "fluent" else None
    result = dict(example_id=row["example_id"], pair_id=gold["pair_id"],
                  condition=condition, language_condition=side, pred_status=row["pred_status"],
                  cc_expected=cc_expected, cc_present=None, cc_outcome=None, cc_pairs=None,
                  sub_to_cc=None,
                  member_recovered=[None, None], extra_members=None, sub_targets=None,
                  withdrawn_member_recovered=None)
    members = repair = combined_result = Result("not_run", "not applicable")
    # Prepare gold independently: gold coverage gaps must not count as model failures.
    spec = pattern = None
    try:
        reference = Graph(gold["sbn_repair"])
        spec, withdrawn = later_spec(reference, gold)
    except NeedsReview as exc:
        members = Result("review", f"gold coverage: {exc}")
    except (InvalidGraph, GoldError, KeyError) as exc:
        members = Result("review", f"gold error: {exc}")
    if combined:
        repair = members
        if spec is not None:
            try:
                pattern = validate_pattern(reference, later_pattern(reference, spec, withdrawn))
            except NeedsReview as exc:
                repair = Result("review", f"gold coverage: {exc}")
            except GoldError as exc:
                repair = Result("review", f"gold error: {exc}")

    graph = None
    if row["pred_status"] == "valid":
        try:
            graph = Graph(row["pred_sbn"])
        except InvalidGraph as exc:
            raise ValueError(f"Previously valid prediction {row['example_id']} cannot be read: {exc}") from exc
        pairs = list(graph.pairs())
        result["cc_pairs"] = [dict(parent=p[1], withdrawn=w[1], accepted=a[1])
                              for p, w, a in pairs]
        present = result["cc_present"] = bool(pairs)
        if cc_expected is not None:
            result["cc_outcome"] = (
                ("true_positive" if present else "false_negative") if cc_expected
                else ("false_positive" if present else "true_negative")
            )

    if graph is None:
        unavailable = (Result("review", "basic scoring reader error")
                       if row["pred_status"] == "scorer_error"
                       else Result("incomplete", row["pred_status"]))
        if spec is not None:
            members = unavailable
        if pattern is not None:
            repair = unavailable
        if combined:
            combined_result = joint((repair, []), (members, [])) if (
                repair.status == "review" or members.status == "review") else unavailable
    else:
        member_bindings = repair_bindings = []
        if spec is not None:
            members, member_bindings, evidence = inspect_reference(graph, spec)
            result.update(evidence)
            # Report where the actual Sub targets land, separately from
            # whether they are the intended members and the repair is correct.
            if combined and evidence["sub_targets"] is not None:
                boxes = [target["box"] for target in evidence["sub_targets"]]
                result["sub_to_cc"] = len(boxes) == 2 and any(
                    set(boxes) == {pair["withdrawn"], pair["accepted"]}
                    for pair in result["cc_pairs"])
            if withdrawn:
                i = list(spec.members).index(withdrawn[0])
                result["withdrawn_member_recovered"] = evidence["member_recovered"][i]
        if combined:
            if pattern is not None:
                repair, repair_bindings = check_repair(graph, pattern)
            combined_result = joint((repair, repair_bindings), (members, member_bindings))
    result.update(members=asdict(members), repair=asdict(repair), joint=asdict(combined_result))
    return result


def check_rows(rows, gold_rows):
    """Join by example_id. Keep fluent and RSR sides labelled for analysis."""
    gold = {}
    for row in gold_rows:
        key = row["example_id"]
        if key in gold:
            raise ValueError(f"Duplicate gold ID: {key}")
        gold[key] = row
    seen = set()
    for row in rows:
        key = (row["example_id"], row["condition"])
        if key in seen:
            raise ValueError(f"Duplicate basic scoring row: {key}")
        seen.add(key)
        yield assess(row, gold[key[0]])


def summarise(items):
    groups = defaultdict(list)
    for row in items:
        groups[row["condition"], row["language_condition"]].append(row)
    result = []
    for (condition, side), rows in sorted(groups.items()):
        cc = Counter(r["cc_outcome"] for r in rows if r["cc_outcome"] is not None)
        entry = dict(condition=condition, language_condition=side, expected=len(rows),
                     cc_valid=sum(r["cc_present"] is not None for r in rows),
                     cc_present=sum(r["cc_present"] is True for r in rows))
        # C2 RSR records presence only; there is no expected C–C outcome.
        for kind, false, true in (("false_positive", "false_positive", "true_negative"),
                                  ("false_negative", "false_negative", "true_positive")):
            denom = cc[false] + cc[true]
            entry[kind] = dict(count=cc[false], denominator=denom,
                               rate=cc[false] / denom if denom else None)
        for check in ("members", "joint"):
            counts = Counter(r[check]["status"] for r in rows)
            denom = counts["pass"] + counts["fail"] + counts["incomplete"]
            entry[check] = dict(counts=dict(counts), denominator=denom,
                                rate=counts["pass"] / denom if denom else None)
        result.append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basic", required=True, type=Path, help="later-reference basic scoring JSONL")
    parser.add_argument("--gold", required=True, type=Path, help="released later_reference/test.parquet")
    parser.add_argument("--output", required=True, type=Path, help="new JSON report")
    args = parser.parse_args()
    try:
        with args.basic.open(encoding="utf-8") as source:
            items = list(check_rows((json.loads(line) for line in source if line.strip()),
                                    pq.read_table(args.gold).to_pylist()))
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(dict(summary=summarise(items), items=items), target, ensure_ascii=False, indent=2)
            target.write("\n")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
