#!/usr/bin/env python3
"""Compare core C3 withdrawal graphs with gold and retain basic scores."""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import pyarrow.parquet as pq

from formal_graph import Node
from formal_local import Result, scope_path
from score_basic import index_rows, parse

CORE_VARIANTS = {"rsr_base", "rsr_editing_expression"}


def withdrawal_graph(graph, pair):
    """Represent the whole withdrawn box, plus the endpoints of its links."""
    parent, withdrawn, _ = pair
    # Embedding links and separators can introduce nested boxes.
    children = {}
    for a, _, b in graph.box_links:
        children.setdefault(a, set()).add(b)
    for _, _, b, owner in graph.relations:
        if b[0] == Node.BOX:
            children.setdefault(owner, set()).add(b)
    inside, pending = {withdrawn}, [withdrawn]
    while pending:
        nested = children.get(pending.pop(), set()) - inside
        pending.extend(nested)
        inside.update(nested)

    out = nx.MultiDiGraph()

    def add(node):
        if node in out:
            return
        kind = node[0]
        if kind == Node.BOX:
            label = ("withdrawn",) if node == withdrawn else (
                ("nested",) if node in inside else
                ("external", tuple(step for step, _ in scope_path(graph, node))))
        else:
            label = (graph.labels[node],)
            if kind == Node.SYNSET and graph.box[node] not in inside:
                # Names and literal identities distinguish external referents
                # with the same synset, without comparing the rest of the clause.
                label += tuple(sorted((r, graph.labels[b]) for a, r, b, _ in graph.relations
                                      if a == node and r in {"Name", "EQU"}
                                      and b[0] == Node.CONSTANT))
        out.add_node(node, label=(kind, label))
        if kind == Node.SYNSET:
            box = graph.box[node]
            add(box)
            out.add_edge(box, node, label="member")

    for box in inside | {parent}:
        add(box)
    for a, role, b in graph.box_links:
        if a in inside or b in inside:
            add(a)
            add(b)
            out.add_edge(a, b, label=role)
    for node, box in graph.box.items():
        if box in inside:
            add(node)
    for i, (a, role, b, owner) in enumerate(graph.relations):
        if owner in inside or graph.box.get(a, a) in inside or graph.box.get(b, b) in inside:
            add(a)
            add(b)
            add(owner)
            # Keep condition scope even after normalising inverse roles.
            condition = ("condition", i)
            out.add_node(condition, label=("condition", role))
            for endpoint, label in ((a, "source"), (b, "target"), (owner, "scope")):
                out.add_edge(condition, endpoint, label=label)
    return out


def same_withdrawal(gold, pred):
    return nx.is_isomorphic(
        gold, pred, node_match=nx.algorithms.isomorphism.categorical_node_match("label", None),
        edge_match=lambda a, b: Counter(e["label"] for e in a.values())
        == Counter(e["label"] for e in b.values()),
    )


def check_rows(gold_rows, basic_rows=None):
    """Use the released core panel, including rows with missing predictions."""
    gold = index_rows((dict(r, condition="C3") for r in gold_rows
                       if r["variant"] in CORE_VARIANTS), "core gold")
    if not gold:
        raise ValueError("No core RSR rows in gold")
    basic = None if basic_rows is None else index_rows(basic_rows, "basic results")
    if basic is not None and basic.keys() != gold.keys():
        missing = sorted(gold.keys() - basic.keys())
        extra = sorted(basic.keys() - gold.keys())
        raise ValueError(f"Expected the full core C3 basic panel; missing {len(missing)}, "
                         f"extra {len(extra)} (examples: {missing[:2]}, {extra[:2]})")
    for key, row in gold.items():
        reference, gold_status, reason = parse(row["sbn_repair"])
        expected = None
        coverage = Result("review", f"gold {gold_status}: {reason}")
        if reference is not None:
            pairs = list(reference.pairs())
            if len(pairs) != 1:
                coverage = Result("review", "gold does not have exactly one C–C pair")
            else:
                try:
                    expected = withdrawal_graph(reference, pairs[0])
                    coverage = Result("ready")
                except ValueError as exc:
                    coverage = Result("review", f"gold scope: {exc}")
        item = {k: row[k] for k in ("example_id", "item_id", "variant", "repair_type")}
        item.update(condition="C3", gold_withdrawal=asdict(coverage))
        if basic is None:
            yield item
            continue

        prediction = basic[key]
        status = prediction["pred_status"]
        item.update(pred_status=status, cc_present=None)
        item["full_graph"] = {k: prediction.get(k) for k in ("score_status", "precision", "recall", "f1")}
        result = coverage
        if status == "valid":
            graph, parsed, reason = parse(prediction["pred_sbn"])
            if parsed != "valid":
                result = Result("review", f"could not reread valid prediction: {reason}")
            else:
                pairs = list(graph.pairs())
                item["cc_present"] = bool(pairs)
                if expected is not None:
                    if len(pairs) != 1:
                        result = Result("fail", "expected exactly one C–C pair")
                    else:
                        try:
                            same = same_withdrawal(expected, withdrawal_graph(graph, pairs[0]))
                            result = Result("pass") if same else Result("fail", "withdrawal graph differs")
                        except ValueError as exc:
                            result = Result("review", f"prediction scope: {exc}")
        elif expected is not None:
            result = (Result("incomplete", status) if status in {"invalid", "missing"}
                      else Result("review", "basic scoring reader error"))
        item["withdrawal"] = asdict(result)
        yield item


def rate(numerator, denominator):
    return dict(numerator=numerator, denominator=denominator,
                rate=numerator / denominator if denominator else None)


def summarise(items, scored):
    coverage = Counter(r["gold_withdrawal"]["status"] for r in items)
    result = dict(expected=len(items), gold_coverage=dict(coverage),
                  gold_review_reasons=dict(Counter(r["gold_withdrawal"]["reason"] for r in items
                                                   if r["gold_withdrawal"]["status"] == "review")))
    by_type = {}
    for row in items:
        by_type.setdefault(row["repair_type"], Counter())[row["gold_withdrawal"]["status"]] += 1
    result["gold_coverage_by_type"] = {kind: dict(counts) for kind, counts in sorted(by_type.items())}
    if scored:
        counts = Counter(r["withdrawal"]["status"] for r in items)
        detected = [r for r in items if r["cc_present"] is True]
        result.update(
            withdrawal_counts=dict(counts),
            cc_detection=rate(len(detected), sum(r["cc_present"] is not None for r in items)),
            withdrawal_match_among_detections=rate(
                sum(r["withdrawal"]["status"] == "pass" for r in detected),
                sum(r["withdrawal"]["status"] in {"pass", "fail"} for r in detected)),
            withdrawal_match_end_to_end=rate(counts["pass"], counts["pass"] + counts["fail"] + counts["incomplete"]),
        )
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--gold", required=True, type=Path, help="published data/test.parquet")
    cli.add_argument("--basic", type=Path, help="full core C3 basic JSONL; omit for gold audit")
    cli.add_argument("--output", required=True, type=Path, help="new JSON report")
    args = cli.parse_args()
    try:
        basic = None
        if args.basic:
            with args.basic.open(encoding="utf-8") as source:
                basic = [json.loads(line) for line in source if line.strip()]
        items = list(check_rows(pq.read_table(args.gold).to_pylist(), basic))
        report = dict(summary=summarise(items, basic is not None), items=items)
        with args.output.open("x", encoding="utf-8") as out:
            json.dump(report, out, ensure_ascii=False, indent=2)
            out.write("\n")
    except (OSError, ValueError, KeyError) as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    main()
