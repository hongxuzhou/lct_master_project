#!/usr/bin/env python3
"""Audit test gold and export a separate, pair-preserving scoring panel."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from check_core_repair import same_withdrawal, withdrawal_graph
from formal_gold import GoldError, NeedsReview, later_pattern, later_spec, validate_pattern
from formal_graph import Graph, InvalidGraph, Node, full_score, make_scorer
from prepare_references import SLICES, select_references

FILES = ("data/test.parquet", "hard_negative/test.parquet", "later_reference/test.parquet")


def audit_rows(panels):
    """Use only gold. A failed partner removes its whole comparison pair."""
    scorer = make_scorer()
    cache = {}
    items = []
    groups = defaultdict(list)
    seen = set()

    def graph(sbn):
        if not isinstance(sbn, str):
            raise InvalidGraph("missing SBN")
        if sbn not in cache:
            try:
                g = Graph(sbn)
                score = full_score(scorer, g, g, relation_recall=True)
                if any(score[k] != 1 for k in ("precision", "recall", "f1")):
                    raise RuntimeError("gold self-score is not exactly one")
                if score["relation_total"] and score["relation_recall"] != 1:
                    raise RuntimeError("gold self relation recall is not exactly one")
                cache[sbn] = (g, None)
            except Exception as exc:
                cache[sbn] = (None, exc)
        g, error = cache[sbn]
        if error is not None:
            raise error
        return g

    def issue(item, check, reason, status="exclude"):
        item["issues"].append(dict(check=check, reason=reason, status=status))

    fluent = {r["doc_id"]: r for r in panels[FILES[0]] if r["variant"] == "pmb_fluent"}
    for filename in FILES:
        for row in panels[filename]:
            if filename == FILES[0]:
                variant = row["variant"]
                if variant not in {"pmb_fluent", "rsr_base", "rsr_editing_expression"}:
                    raise ValueError(f"Unknown default test variant: {variant}")
                slice_name = "pmb_fluent" if variant == "pmb_fluent" else "core_rsr"
            else:
                slice_name = "hard_negative" if filename == FILES[1] else "later_reference"
            rid = row["example_id"]
            if not isinstance(rid, str) or not rid.strip() or rid in seen:
                raise ValueError(f"Blank or duplicate test ID: {rid!r}")
            seen.add(rid)
            item = dict(example_id=rid, source_file=filename, slice=slice_name,
                        pair_id=None, text=row["text"], issues=[], graph_checks={})
            items.append(item)
            if not isinstance(row["text"], str) or not row["text"].strip():
                issue(item, "text", "missing input text")
            if slice_name in {"core_rsr", "later_reference"}:
                key = row["item_id"] if slice_name == "core_rsr" else row["pair_id"]
                if not key:
                    raise ValueError(f"Missing pair ID: {rid}")
                item["pair_id"] = key
                groups[slice_name, key].append((row, item))
            graphs = {}
            for field in ("sbn_clean", "sbn_repair"):
                if field == "sbn_clean" and slice_name == "later_reference":
                    continue
                try:
                    graphs[field] = graph(row.get(field))
                    item["graph_checks"][field] = "pass"
                except InvalidGraph as exc:
                    item["graph_checks"][field] = "invalid"
                    issue(item, field, str(exc))
                except Exception as exc:
                    item["graph_checks"][field] = "audit_error"
                    issue(item, field, f"{type(exc).__name__}: {exc}", "hold")

            repair = graphs.get("sbn_repair")
            for field, g in graphs.items():
                labels = Counter(r for _, r, _ in g.box_links)
                expected = (field == "sbn_repair" and (slice_name == "core_rsr" or
                            slice_name == "later_reference" and row["language_condition"] == "rsr"))
                if expected:
                    if labels["CORRECTION"] != 1 or len(list(g.pairs())) != 1:
                        issue(item, field + ".repair_structure", "expected exactly one paired CORRECTION")
                elif labels["CORRECTION"]:
                    issue(item, field + ".repair_structure", "unexpected CORRECTION in ordinary gold")
            if slice_name in {"pmb_fluent", "hard_negative"}:
                if row.get("sbn_clean") != row["sbn_repair"]:
                    issue(item, "ordinary_targets", "sbn_clean and sbn_repair differ")
            if slice_name == "core_rsr":
                original = fluent.get(row["doc_id"])
                if original is None or row["sbn_clean"] != original["sbn_clean"]:
                    issue(item, "source_target", "clean target differs from source fluent gold")
                if repair is not None and len(list(repair.pairs())) == 1:
                    try:
                        view = withdrawal_graph(repair, next(repair.pairs()))
                        if not same_withdrawal(view, view):
                            raise RuntimeError("withdrawal self-check failed")
                    except Exception as exc:
                        issue(item, "withdrawal_selfcheck", f"{type(exc).__name__}: {exc}", "hold")
            if slice_name == "later_reference" and repair is not None:
                try:
                    if row["language_condition"] not in {"fluent", "rsr"}:
                        raise GoldError("unknown language_condition")
                    spec, withdrawn = later_spec(repair, row)
                    if row["language_condition"] == "rsr":
                        validate_pattern(repair, later_pattern(repair, spec, withdrawn))
                    group = repair.concepts[row["reference_group_index"]]
                    quantity = [repair.labels[b].strip('"') for a, r, b, _ in repair.relations
                                if a == group and r == "Quantity" and b[0] == Node.CONSTANT]
                    expected_quantity = [] if row["quantity"] is None else [str(row["quantity"])]
                    if quantity != expected_quantity:
                        raise GoldError("group Quantity disagrees with metadata")
                except GoldError as exc:
                    issue(item, "reference_metadata", str(exc))
                except NeedsReview as exc:
                    issue(item, "reference_selfcheck", str(exc), "hold")
                except Exception as exc:
                    issue(item, "reference_selfcheck", f"{type(exc).__name__}: {exc}", "hold")

    for (slice_name, _), partners in groups.items():
        field = "variant" if slice_name == "core_rsr" else "language_condition"
        expected = {"rsr_base", "rsr_editing_expression"} if slice_name == "core_rsr" else {"fluent", "rsr"}
        problems = []
        if len(partners) != 2 or {r[field] for r, _ in partners} != expected:
            problems.append("comparison pair is incomplete or has duplicate sides")
        else:
            left, right = (r for r, _ in partners)
            fields = ("sbn_clean", "sbn_repair", "doc_id", "repair_type") if slice_name == "core_rsr" else (
                "family_id", "partition_unit", "construction_type", "referent_type", "quantity")
            problems.extend(f"paired {f} differs" for f in fields if left[f] != right[f])
        for _, item in partners:
            for problem in problems:
                issue(item, "pair_consistency", problem)
        failed = [item["example_id"] for _, item in partners if item["issues"]]
        if failed:
            for _, item in partners:
                if not item["issues"]:
                    issue(item, "paired_exclusion", "partner not retained: " + ", ".join(failed))
    for item in items:
        item["status"] = ("hold" if any(i["status"] == "hold" for i in item["issues"])
                          else "exclude" if item["issues"] else "retain")
    return items, len(cache)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--test-root", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path, help="new audit directory")
    args = cli.parse_args()
    if args.output.exists():
        cli.error("output already exists; use a new directory")
    tables = {f: pq.read_table(args.test_root / f) for f in FILES}
    items, unique_graphs = audit_rows({f: t.to_pylist() for f, t in tables.items()})
    args.output.mkdir(parents=True)
    retained_ids = {r["example_id"] for r in items if r["status"] == "retain"}
    for f, table in tables.items():
        target = args.output / "retained" / f
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.filter(pa.array([rid in retained_ids for rid in table["example_id"].to_pylist()])), target)
    with (args.output / "items.jsonl").open("x") as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    for status in ("retain", "exclude", "hold"):
        with (args.output / f"{status}.tsv").open("x", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(("example_id", "slice", "pair_id", "reason"))
            for item in items:
                if item["status"] == status:
                    writer.writerow((item["example_id"], item["slice"], item["pair_id"],
                                     "; ".join(i["check"] + ": " + i["reason"] for i in item["issues"])))
    summary = dict(input_rows=len(items), unique_target_strings_checked=unique_graphs,
                   statuses=dict(Counter(r["status"] for r in items)), slices={}, references={},
                   rule="Current PMB reader and evaluation Graph constraints, including raw DAG; no rewrites.",
                   scope="Graph validity, full-score self-check, target/metadata and pair consistency; not a fresh human semantic review.")
    refs = args.output / "references"
    refs.mkdir()
    for slice_name in SLICES:
        summary["slices"][slice_name] = dict(Counter(r["status"] for r in items if r["slice"] == slice_name))
        for condition in ("C1", "C2", "C3"):
            if condition == "C1" and slice_name in {"core_rsr", "later_reference"}:
                continue
            _, rows = select_references(args.output / "retained", slice_name, condition)
            with (refs / f"{slice_name}.{condition}.tsv").open("x", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t")
                writer.writerow(("example_id", "condition", "gold_sbn"))
                writer.writerows(rows)
            summary["references"][f"{slice_name}.{condition}"] = len(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
