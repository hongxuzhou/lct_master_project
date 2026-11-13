#!/usr/bin/env python3
"""Research prototype for the combined RepairSBN evaluation design.

This script illustrates how the planned checks fit together. It is a
theoretical prototype, not the operational evaluation pipeline.
Use score_basic.py for the standalone graph validity and SMATCH++ step.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from formal_gold import (GoldError, NeedsReview, annotated_pattern, core_pattern,
                         later_pattern, later_spec, validate_pattern)
from formal_graph import Graph, InvalidGraph, full_score, make_scorer
from formal_local import PASS, Result, check_reference, check_repair, joint
from sbn_env import sbn_src_dir
from tables import read_table

CONDITIONS = ("C1", "C2", "C3")
CHECKS = ("validity", "full", "false_repair", "repair", "members", "joint")
NA = Result("not_run", "not applicable")
INVALID = Result("fail", "prediction cannot form a valid graph")
MISSING = Result("not_run", "prediction row absent")


@dataclass
class Gold:
    value: object = None
    issue: Result | None = None


def prepare(fn, *args):
    try:
        return Gold(fn(*args))
    except NeedsReview as exc:
        return Gold(issue=Result("review", f"gold coverage: {exc}"))
    except (GoldError, InvalidGraph, KeyError) as exc:
        return Gold(issue=Result("review", f"gold error: {exc}"))
    except Exception as exc:
        return Gold(issue=Result("review", f"scorer error: {type(exc).__name__}: {exc}"))


def _core(g, row):
    return validate_pattern(g, core_pattern(g, row))


def _later(g, spec, withdrawn):
    return validate_pattern(g, later_pattern(g, spec, withdrawn))


def classify(config, row):
    if config == "hard_negative":
        return "hard_negative"
    if config == "later_reference":
        side = row["language_condition"]
        if side not in ("fluent", "rsr"):
            raise ValueError(f"Unknown later-reference side: {side}")
        return "later_" + side
    variant = row["variant"]
    if variant == "pmb_fluent":
        return "pmb_fluent"
    if variant in ("rsr_base", "rsr_editing_expression"):
        return "core_rsr"
    raise ValueError(f"Unexpected default/test variant: {variant}")


def full_field(slice_, condition):
    if slice_ == "core_rsr" and condition in ("C1", "C2"):
        return "sbn_clean"
    return "sbn_repair"


def prepare_row(row, slice_, conditions, annotation=None):
    fields = {full_field(slice_, c) for c in conditions}
    graphs = {field: prepare(lambda f: Graph(row[f]), field) for field in fields}
    repair = reference = Gold()
    if "C3" in conditions and slice_ in ("core_rsr", "later_rsr"):
        repair = graphs["sbn_repair"]
    if slice_.startswith("later_"):
        source = graphs["sbn_repair"]
        reference = source if source.issue else prepare(later_spec, source.value, row)
        if slice_ == "later_rsr" and "C3" in conditions and not repair.issue:
            repair = reference if reference.issue else prepare(_later, source.value, *reference.value)
    if slice_ == "core_rsr" and "C3" in conditions and not repair.issue:
        repair = prepare(_core, repair.value, row)
    if annotation is not None and "C3" in conditions and slice_ in ("core_rsr", "later_rsr"):
        source = graphs["sbn_repair"]
        def reviewed(g):
            digest = hashlib.sha256(row["sbn_repair"].encode()).hexdigest()
            if annotation.get("gold_sha256") != digest:
                raise GoldError("reviewed pattern targets a different gold graph")
            return annotated_pattern(g, annotation)
        repair = source if source.issue else prepare(reviewed, source.value)
    return graphs, repair, reference


def _checked(fn, *args):
    try:
        return fn(*args)
    except Exception as exc:
        return Result("review", f"scorer error: {type(exc).__name__}: {exc}"), []


def score_row(scorer, row, slice_, condition, prepared, prediction):
    graphs, repair, reference = prepared
    field = full_field(slice_, condition)
    gold = graphs[field]
    checks = dict.fromkeys(CHECKS, NA)
    rec = dict(example_id=row["example_id"], slice=slice_, condition=condition,
               variant=row.get("variant"), pair_id=row.get("pair_id") if slice_.startswith("later_")
               else row.get("item_id") if slice_ == "core_rsr" else None,
               doc_id=row.get("doc_id"), repair_type=row.get("repair_type"),
               construction_type=row.get("construction_type"), category=row.get("category"),
               editing_expression=row.get("editing_expression"),
               gold_field=field, pred_sbn=None if prediction is None else prediction["pred_sbn"],
               f1=None, relation_recall=None, relation_matched=None, relation_total=None)
    g = None
    try:
        if prediction is not None:
            g = Graph(prediction["pred_sbn"])
        checks["validity"] = PASS if g else MISSING
        unavailable = MISSING
    except InvalidGraph as exc:
        checks["validity"] = Result("fail", str(exc))
        unavailable = INVALID
    except Exception as exc:
        unavailable = Result("review", f"scorer error: {type(exc).__name__}: {exc}")
        checks["validity"] = unavailable

    if gold.issue:
        checks["full"] = gold.issue
    elif g is None:
        checks["full"] = unavailable
        if unavailable in (INVALID, MISSING):
            rec["f1"] = 0.0
    else:
        try:
            rec.update(full_score(scorer, gold.value, g, slice_ == "hard_negative"))
            checks["full"] = (PASS if rec["f1"] == 1 else Result("fail", "full graph differs"))
        except Exception as exc:
            checks["full"] = Result("review", f"scorer error: {type(exc).__name__}: {exc}")

    if slice_ in ("pmb_fluent", "hard_negative"):
        if g is None:
            checks["false_repair"] = Result("not_run", "no valid prediction graph")
        else:
            checks["false_repair"] = (Result("fail", "paired CORRECTION and CONJUNCTION present")
                                      if any(g.pairs()) else PASS)

    do_members = slice_.startswith("later_")
    do_repair = condition == "C3" and slice_ in ("core_rsr", "later_rsr")
    if do_members:
        members = ((reference.issue or unavailable, []) if reference.issue or g is None
                   else _checked(check_reference, g, reference.value[0]))
        checks["members"] = members[0]
    if do_repair:
        local = ((repair.issue or unavailable, []) if repair.issue or g is None
                 else _checked(check_repair, g, repair.value))
        checks["repair"] = local[0]
    if do_members and do_repair:
        if g is None:
            checks["joint"] = repair.issue or reference.issue or unavailable
        else:
            checks["joint"] = _checked(lambda: (joint(local, members), []))[0]
    rec["checks"] = {name: asdict(result) for name, result in checks.items()}
    return rec


def summarise(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["slice"], r["condition"], None)].append(r)
        if r["slice"] == "core_rsr":
            groups[(r["slice"], r["condition"], r["variant"])].append(r)
    output = []
    for (slice_, condition, variant), rows in sorted(groups.items(), key=lambda x: str(x[0])):
        group = dict(slice=slice_, condition=condition, variant=variant, rows=len(rows))
        for name in CHECKS:
            counts = Counter(r["checks"][name]["status"] for r in rows)
            absent = sum(r["checks"][name]["reason"] == MISSING.reason for r in rows)
            denom = counts["pass"] + counts["fail"] + absent
            numerator = counts["fail"] if name == "false_repair" else counts["pass"]
            group[name] = dict(counts={s: counts[s] for s in ("pass", "fail", "not_run", "review")},
                               denominator=denom, numerator=numerator,
                               rate=numerator / denom if denom else None)
        for metric in ("f1", "relation_recall"):
            values = [r[metric] for r in rows if r[metric] is not None]
            group[metric] = dict(n=len(values), mean=sum(values) / len(values) if values else None)
        output.append(group)
    return output


def paired_results(records):
    pairs = defaultdict(dict)
    for r in records:
        if r["pair_id"] is not None and not pd.isna(r["pair_id"]):
            if r["slice"] not in {"core_rsr", "later_fluent", "later_rsr"}:
                continue
            key = ("core_rsr" if r["slice"] == "core_rsr" else "later_reference",
                   r["pair_id"], r["condition"])
            if r["variant"] in pairs[key]:
                raise ValueError(f"Duplicate pair side: {key}, {r['variant']}")
            pairs[key][r["variant"]] = {k: r[k] for k in ("example_id", "checks", "f1")}
    return [dict(slice=s, pair_id=p, condition=c, sides=sides)
            for (s, p, c), sides in sorted(pairs.items())]


def conditions_for(slice_, requested, core_c1):
    allowed = (CONDITIONS if slice_ in ("pmb_fluent", "hard_negative")
               or (slice_ == "core_rsr" and core_c1) else ("C2", "C3"))
    return tuple(c for c in allowed if c in requested)


def evaluate(tables, predictions, conditions=CONDITIONS, core_c1=False, annotations=None):
    annotations = annotations or {}
    inputs = {}
    for config, rows in tables.items():
        for row in rows:
            rid = row["example_id"]
            if not isinstance(rid, str) or not rid or rid in inputs:
                raise ValueError(f"Missing or duplicate gold ID: {rid!r}")
            inputs[rid] = (classify(config, row), row)
    if annotations.keys() - inputs.keys():
        raise ValueError("Reviewed pattern ID absent from selected gold")
    pred_by_key = {}
    for row in predictions:
        rid, condition = row["example_id"], row["condition"]
        if rid not in inputs or condition not in conditions_for(inputs[rid][0], conditions, core_c1):
            raise ValueError(f"Prediction outside selected panel: {rid}, {condition}")
        if (rid, condition) in pred_by_key:
            raise ValueError(f"Duplicate prediction: {rid}, {condition}")
        pred_by_key[rid, condition] = row
    scorer = make_scorer()
    for rid, (slice_, row) in inputs.items():
        panel = conditions_for(slice_, conditions, core_c1)
        prepared = prepare_row(row, slice_, panel, annotations.get(rid))
        for condition in panel:
            yield score_row(scorer, row, slice_, condition, prepared, pred_by_key.get((rid, condition)))


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default", type=Path, help="default/test parquet")
    parser.add_argument("--hard-negative", type=Path, help="hard_negative/test parquet")
    parser.add_argument("--later-reference", type=Path, help="later_reference/test parquet")
    parser.add_argument("--predictions", type=Path, required=True, help="example_id, condition, pred_sbn")
    parser.add_argument("--output", type=Path, required=True, help="per-item JSONL")
    parser.add_argument("--summary", type=Path, required=True, help="summary and paired outcomes JSON")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--core-c1", action="store_true", help="include optional C1 core comparison")
    parser.add_argument("--reviewed-repairs", type=Path, help="JSON sidecar keyed by example_id")
    args = parser.parse_args()
    sources = {k: getattr(args, k) for k in ("default", "hard_negative", "later_reference")
               if getattr(args, k)}
    if not sources:
        parser.error("select at least one gold table")
    files = list(sources.values()) + [args.predictions]
    if args.reviewed_repairs:
        files.append(args.reviewed_repairs)
    paths = [p.resolve() for p in (*files, args.output, args.summary)]
    if paths[-1] == paths[-2] or any(p in paths[:-2] for p in paths[-2:]):
        parser.error("outputs must be separate from each other and from inputs")

    def read(path, required):
        table = read_table(path)
        if required - set(table):
            parser.error(f"{path} lacks fields: {sorted(required - set(table))}")
        return table.astype(object).where(table.notna(), None).to_dict("records")

    gold_tables = {k: read(path, {"example_id"} | {
        "default": {"variant", "sbn_clean", "sbn_repair", "item_id"},
        "hard_negative": {"sbn_repair"},
        "later_reference": {"variant", "pair_id", "language_condition", "sbn_repair",
                            "reference_group_index", "antecedent_indices", "withdrawn_antecedent_indices"},
    }[k]) for k, path in sources.items()}
    predictions = read(args.predictions, {"example_id", "condition", "pred_sbn"})
    annotations = json.loads(args.reviewed_repairs.read_text()) if args.reviewed_repairs else None
    records = list(evaluate(gold_tables, predictions, args.conditions, args.core_c1, annotations))
    reader = sbn_src_dir()
    summary = dict(groups=summarise(records), pairs=paired_results(records),
                   smatchpp=version("smatchpp"), solver="ilp",
                   sbn_src=str(reader), sbn_reader_sha256={
                       f: hashlib.sha256((reader / f).read_bytes()).hexdigest()
                       for f in ("sbn_smatch.py", "sbn_spec.py", "graph_base.py")}, inputs={
                       str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    with args.output.open("w") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False, default=_json_default, allow_nan=False) + "\n")
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                      default=_json_default, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
