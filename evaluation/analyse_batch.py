#!/usr/bin/env python3
"""Build scoped-v2 tables and selected document-paired comparisons for formal runs."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from score_batch import AUDIT, BATCH, HERE, digest
from score_basic import require_panel
from sbn_env import sbn_src_dir


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def load_panel(sizes=("2B", "9B", "27B")):
    main = pd.read_parquet(AUDIT / "retained/data/test.parquet")
    later = pd.read_parquet(AUDIT / "retained/later_reference/test.parquet")
    hard = pd.read_parquet(AUDIT / "retained/hard_negative/test.parquet")
    metadata = pd.concat([main, later, hard], ignore_index=True).set_index("example_id")
    records, sources = [], []
    for size in sizes:
        for condition in ("C1", "C2", "C3"):
            for seed in (24, 42, 4224):
                run = BATCH / size / condition / f"seed{seed}"
                pred = pd.read_csv(run / "predictions.tsv", sep="\t", keep_default_na=False)
                if pred.duplicated(["example_id", "condition"]).any() or set(pred.condition) != {condition}:
                    raise ValueError(f"Bad raw prediction keys: {run}")
                slices = ["pmb_fluent", "hard_negative"] + ([] if condition == "C1" else ["core_rsr", "later_reference"])
                expected_raw = {"C1": 1260, "C2": 2586, "C3": 2586}[condition]
                if len(pred) != expected_raw:
                    raise ValueError(f"Unexpected raw prediction panel: {run}: {len(pred)}")
                retained_keys = set()
                for slice_name in slices:
                    reference = AUDIT / "references" / f"{slice_name}.{condition}.tsv"
                    refs = pd.read_csv(reference, sep="\t", keep_default_na=False)
                    retained_keys.update(refs.example_id)
                    gold = refs.set_index("example_id").gold_sbn.to_dict()
                    basic = run / "scores-scoped-v2" / f"{slice_name}.basic.jsonl"
                    rows = read_jsonl(basic)
                    require_panel(rows, refs.to_dict("records"))
                    if any(r["score_version"] != "scoped_sbn_v2" or r["score_status"] not in {"scored", "incomplete"} for r in rows):
                        raise ValueError(f"Unresolved scoring/version problem: {basic}")
                    checks = basic.with_name(f"{slice_name}.checks.json")
                    targeted = {}
                    if slice_name != "core_rsr" or condition == "C3":
                        items = json.loads(checks.read_text())["items"]
                        require_panel(items, refs.to_dict("records"))
                        targeted = {r["example_id"]: r for r in items}
                        sources.append(checks)
                    raw = pred.set_index("example_id")
                    for row in rows:
                        key = row["example_id"]
                        meta = metadata.loc[key].to_dict()
                        out = dict(size=size, seed=seed, slice=slice_name, **row)
                        out.update({k: meta.get(k) for k in ["text", "doc_id", "item_id", "variant", "pair_id", "family_id", "partition_unit", "language_condition", "repair_type", "editing_expression", "construction_type", "category"]})
                        out["unit_id"] = (meta["pair_id"] if slice_name == "later_reference" else meta["doc_id"])
                        out["gold_column"] = "sbn_clean" if slice_name == "core_rsr" and condition == "C2" else "sbn_repair"
                        out["gold_sbn"] = gold[key]
                        out["tokens"] = raw.loc[key, "tokens"] if key in raw.index else None
                        out["ended_with_eos"] = raw.loc[key, "ended_with_eos"] if key in raw.index else None
                        if key in raw.index and row["pred_sbn"] != raw.loc[key, "pred_sbn"]:
                            raise ValueError(f"Raw/scored prediction mismatch: {run}/{key}")
                        check = targeted.get(key, {})
                        out["cc_present"] = check.get("cc_present")
                        out["false_positive"] = check.get("false_positive")
                        out["cc_outcome"] = check.get("cc_outcome")
                        out["sub_targets_json"] = json.dumps(check.get("sub_targets"))
                        out["cc_pairs_json"] = json.dumps(check.get("cc_pairs"))
                        for name in ("withdrawal", "members", "repair", "joint"):
                            value = check.get(name, {})
                            out[name + "_status"] = value.get("status")
                            out[name + "_reason"] = value.get("reason")
                        member = check.get("member_recovered", [None, None])
                        out.update(member1_recovered=member[0], member2_recovered=member[1],
                                   extra_members=check.get("extra_members"),
                                   withdrawn_member_recovered=check.get("withdrawn_member_recovered"),
                                   sub_to_cc=check.get("sub_to_cc"))
                        records.append(out)
                    sources += [reference, basic]
                extra = set(pred.example_id) - retained_keys
                if extra != {"pmb_test::p32/d1803"}:
                    raise ValueError(f"Unexpected unused raw predictions: {run}: {extra}")
                sources.append(run / "predictions.tsv")
    data = pd.DataFrame(records)
    if data.duplicated(["size", "condition", "seed", "example_id"]).any() or data.unit_id.isna().any():
        raise ValueError("Duplicate observations or missing statistical units")
    if data.f1.isna().any() or not data.f1.between(0, 1).all():
        raise ValueError("Missing or out-of-range failure-penalised F1")
    return data, sorted(set(sources))


def group_summary(data):
    result = []
    for key, panel in data.groupby(["size", "condition", "seed", "slice"], sort=True):
        partitions = {"all": panel}
        if key[-1] == "core_rsr":
            partitions.update({name: group for name, group in panel.groupby("variant")})
        elif key[-1] == "later_reference":
            partitions.update({name: group for name, group in panel.groupby("language_condition")})
        for part, d in partitions.items():
            row = dict(zip(["size", "condition", "seed", "slice"], key)) | dict(part=part,
                       n=len(d), n_units=d.unit_id.nunique(), gold_column=d.gold_column.iloc[0],
                       n_valid=int(d.pred_status.eq("valid").sum()),
                       n_invalid=int(d.pred_status.eq("invalid").sum()),
                       n_missing=int(d.pred_status.eq("missing").sum()),
                       n_scored=int(d.score_status.eq("scored").sum()),
                       f1_mean=d.f1.mean(), f1_document_mean=d.groupby("unit_id").f1.mean().mean(),
                       precision_aligned_mean=d.precision.mean(), recall_aligned_mean=d.recall.mean(),
                       exact_full_graph_n=int(d.f1.ge(1-1e-9).sum()),
                       valid_rate=d.pred_status.eq("valid").mean(),
                       relation_n=int(d.relation_recall.notna().sum()),
                       relation_recall_aligned_mean=d.relation_recall.mean(),
                       relation_matched_aligned=d.relation_matched.sum(min_count=1),
                       relation_total_aligned=d.relation_total.sum(min_count=1))
            row["relation_recall_aligned_micro"] = (row["relation_matched_aligned"] / row["relation_total_aligned"] if row["relation_total_aligned"] else None)
            for metric in ("withdrawal", "members", "repair", "joint"):
                counts = Counter(d[metric + "_status"].dropna())
                for status in ("pass", "fail", "incomplete", "review"):
                    row[f"{metric}_{status}"] = counts[status]
                denominator = sum(counts[s] for s in ("pass", "fail", "incomplete"))
                row[f"{metric}_denominator"] = denominator
                row[f"{metric}_rate"] = counts["pass"] / denominator if denominator else None
                applicable = denominator + counts["review"]
                row[f"{metric}_fixed_panel_lower"] = counts["pass"] / applicable if applicable else None
                row[f"{metric}_fixed_panel_upper"] = (counts["pass"] + counts["review"]) / applicable if applicable else None
            for metric in ("cc_present", "false_positive", "member1_recovered", "member2_recovered",
                           "withdrawn_member_recovered", "sub_to_cc"):
                denominator = int(d[metric].notna().sum())
                numerator = int(d[metric].eq(True).sum())
                row[f"{metric}_n"] = numerator
                row[f"{metric}_denominator"] = denominator
                row[f"{metric}_rate"] = numerator / denominator if denominator else None
            row["extra_members_resolved_n"] = int(d.extra_members.notna().sum())
            row["extra_members_resolved_mean"] = d.extra_members.mean()
            result.append(row)
    return pd.DataFrame(result)


def paired_comparisons(data):
    """Pair equal units within seed; never pool seeds as independent data."""
    effects, units = [], []
    sizes = [s for s in ("2B", "9B", "27B") if s in set(data["size"])]
    scale_pairs = [(a, b) for i, a in enumerate(sizes) for b in sizes[i+1:]]

    def compare(name, metric, left, right, size, condition):
        # Metric is averaged over all variants of a source document first.
        # Targeted reviews are unavailable, and remain explicit paired exclusions.
        def values(d):
            if metric == "f1":
                v = d.f1
            else:
                status = d[metric + "_status"]
                v = status.eq("pass").astype(float).where(status.isin(["pass", "fail", "incomplete"]))
            return d.assign(value=v).groupby("unit_id").agg(value=("value", "mean"), n=("value", "size"), available=("value", "count"))
        a, b = values(left), values(right)
        if set(a.index) != set(b.index):
            raise ValueError(f"Unmatched units in {name}")
        paired = a.join(b, lsuffix="_left", rsuffix="_right")
        # Do not silently average a partial set of variants when reviews occur.
        keep = paired.n_left.eq(paired.available_left) & paired.n_right.eq(paired.available_right)
        for unit_id, r in paired.iterrows():
            units.append(dict(comparison=name, metric=metric, size=size, condition=condition,
                              seed=int(left.seed.iloc[0]), unit_id=unit_id, included=bool(keep.loc[unit_id]),
                              left=r.value_left, right=r.value_right,
                              difference=r.value_right-r.value_left if keep.loc[unit_id] else None))
        delta = (paired.loc[keep, "value_right"] - paired.loc[keep, "value_left"]).to_numpy()
        seed = int(left.seed.iloc[0])
        rng_seed = int(hashlib.sha256(f"{name}/{size}/{condition}/{seed}/{metric}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(rng_seed)
        draws = np.concatenate([rng.choice(delta, size=(min(250, 10000-i), len(delta)), replace=True).mean(axis=1)
                                for i in range(0, 10000, 250)]) if len(delta) else np.array([np.nan])
        low, high = np.quantile(draws, [0.025, 0.975])
        effects.append(dict(comparison=name, metric=metric, size=size, condition=condition, seed=seed,
                            n_units=len(delta), excluded_review_units=int((~keep).sum()),
                            left_mean=paired.loc[keep, "value_left"].mean(), right_mean=paired.loc[keep, "value_right"].mean(),
                            difference=delta.mean() if len(delta) else None, ci_low=low, ci_high=high,
                            bootstrap_samples=10000, bootstrap_seed=rng_seed))

    for seed in (24, 42, 4224):
        d = data[data.seed.eq(seed)]
        for condition in ("C1", "C2", "C3"):
            for slice_name in ("pmb_fluent", "core_rsr"):
                if condition == "C1" and slice_name == "core_rsr":
                    continue
                p = d[d.condition.eq(condition) & d.slice.eq(slice_name)]
                for smaller, larger in scale_pairs:
                    compare(f"scale_{slice_name}_{larger}_minus_{smaller}", "f1", p[p["size"].eq(smaller)], p[p["size"].eq(larger)], f"{smaller}_to_{larger}", condition)
        for size in sizes:
            p = d[d['size'].eq(size)]
            for baseline in ("C1", "C2"):
                q = p[p.slice.eq("pmb_fluent")]
                compare(f"fluent_C3_minus_{baseline}", "f1", q[q.condition.eq(baseline)], q[q.condition.eq("C3")], size, f"{baseline}_to_C3")
            for condition in ("C2", "C3"):
                q = p[p.slice.eq("core_rsr") & p.condition.eq(condition)]
                compare("core_marked_minus_unmarked", "f1", q[q.variant.eq("rsr_base")], q[q.variant.eq("rsr_editing_expression")], size, condition)
                if condition == "C3":
                    compare("core_marked_minus_unmarked", "withdrawal", q[q.variant.eq("rsr_base")], q[q.variant.eq("rsr_editing_expression")], size, condition)
                    sorry = q[q.editing_expression.eq("sorry")]
                    unmarked = q[q.variant.eq("rsr_base") & q.item_id.isin(sorry.item_id)]
                    compare("core_sorry_minus_its_unmarked_partner", "withdrawal", unmarked, sorry, size, condition)
            for side in ("fluent", "rsr"):
                q = p[p.slice.eq("later_reference") & p.language_condition.eq(side)]
                compare(f"later_{side}_C3_minus_C2", "members", q[q.condition.eq("C2")], q[q.condition.eq("C3")], size, "C2_to_C3")
            q = p[p.slice.eq("later_reference") & p.condition.eq("C3")]
            compare("later_C3_rsr_minus_fluent", "members", q[q.language_condition.eq("fluent")], q[q.language_condition.eq("rsr")], size, "C3")
        q = d[d.slice.eq("core_rsr") & d.condition.eq("C3")]
        for smaller, larger in scale_pairs:
            compare(f"scale_core_{larger}_minus_{smaller}", "withdrawal", q[q["size"].eq(smaller)], q[q["size"].eq(larger)], f"{smaller}_to_{larger}", "C3")
            for condition in ("C2", "C3"):
                for side in ("fluent", "rsr"):
                    later = d[d.slice.eq("later_reference") & d.condition.eq(condition)
                              & d.language_condition.eq(side)]
                    compare(f"scale_later_{side}_{larger}_minus_{smaller}", "members",
                            later[later["size"].eq(smaller)], later[later["size"].eq(larger)],
                            f"{smaller}_to_{larger}", condition)
                    if condition == "C3" and side == "rsr":
                        compare(f"scale_later_{side}_{larger}_minus_{smaller}", "joint",
                                later[later["size"].eq(smaller)], later[later["size"].eq(larger)],
                                f"{smaller}_to_{larger}", condition)
    effects = pd.DataFrame(effects)
    groups = ["comparison", "metric", "size", "condition"]
    summary = effects.groupby(groups).agg(n_seeds=("seed", "size"), mean_difference=("difference", "mean"),
                  min_difference=("difference", "min"), max_difference=("difference", "max"),
                  min_units=("n_units", "min"), max_units=("n_units", "max")).reset_index()
    summary["direction_consistent"] = (summary.min_difference.gt(0) | summary.max_difference.lt(0))
    summary["all_zero"] = summary.min_difference.eq(0) & summary.max_difference.eq(0)
    return effects, summary, pd.DataFrame(units)


def diagnostics(data):
    rows = []
    for size in (s for s in ("2B", "9B", "27B") if s in set(data["size"])):
        for seed in (24, 42, 4224):
            core = data[data['size'].eq(size) & data.seed.eq(seed) & data.condition.eq("C3") & data.slice.eq("core_rsr")]
            for kind in ("repair_type", "editing_expression"):
                for value, d in core.groupby(kind, dropna=False):
                    rows.append(dict(size=size, seed=seed, diagnostic=kind, value="unmarked" if pd.isna(value) else value,
                                     n=len(d), n_units=d.unit_id.nunique(), valid=int(d.pred_status.eq("valid").sum()),
                                     f1_mean=d.f1.mean(), withdrawal_pass=int(d.withdrawal_status.eq("pass").sum()),
                                     withdrawal_review=int(d.withdrawal_status.eq("review").sum())))
    return pd.DataFrame(rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=BATCH / "analysis-scoped-v2-all-sizes")
    cli.add_argument("--sizes", nargs="+", choices=("2B", "9B", "27B"), default=["2B", "9B", "27B"])
    cli.add_argument("--update-index", action="store_true", help="Replace the batch summary index; previous snapshots remain in their analysis directories")
    args = cli.parse_args()
    if args.output.exists():
        cli.error(f"Output already exists: {args.output}; choose a new directory")
    data, sources = load_panel(args.sizes)
    summary = group_summary(data)
    effects, comparisons, units = paired_comparisons(data)
    args.output.mkdir(exist_ok=False)
    data.to_parquet(args.output / "items.parquet", index=False)
    summary.to_csv(args.output / "summary.tsv", sep="\t", index=False)
    metric_columns = ["f1_mean", "f1_document_mean", "valid_rate", "relation_recall_aligned_mean",
                      "withdrawal_rate", "members_rate", "joint_rate", "false_positive_rate"]
    seed_summary = summary.groupby(["size", "condition", "slice", "part"])[metric_columns].agg(["mean", "min", "max"])
    seed_summary.columns = ["_".join(c) for c in seed_summary.columns]
    seed_summary.reset_index().to_csv(args.output / "seed_summary.tsv", sep="\t", index=False)
    effects.to_csv(args.output / "paired_effects.tsv", sep="\t", index=False)
    comparisons.to_csv(args.output / "comparisons.tsv", sep="\t", index=False)
    units.to_parquet(args.output / "paired_units.parquet", index=False)
    diagnostics(data).to_csv(args.output / "core_diagnostics.tsv", sep="\t", index=False)
    sources += list(HERE.glob("*.py")) + list((AUDIT / "retained").rglob("*.parquet"))
    sources += [BATCH / name for name in ("frozen-lst.json", "test-hashes.json")
                if "27B" in args.sizes]
    if any(size in args.sizes for size in ("2B", "9B")):
        sources.append(BATCH / "frozen-habrok.json")
    sources += [AUDIT / name for name in ("summary.json", "exclude.tsv", "hold.tsv")]
    sources += list(sbn_src_dir().glob("*.py"))
    manifest = dict(score_version="scoped_sbn_v2", sizes=args.sizes, seeds=[24, 42, 4224],
                    runs=len(data[["size", "condition", "seed"]].drop_duplicates()),
                    panels=len(data[["size", "condition", "seed", "slice"]].drop_duplicates()),
                    rows=len(data), n_gold_errors=0, n_scorer_errors=0,
                    pred_status=dict(Counter(data.pred_status)), score_status=dict(Counter(data.score_status)),
                    excluded_gold=["pmb_test::p32/d1803"],
                    source_sha256={os.path.relpath(p, HERE): digest(p) for p in sorted(set(sources))},
                    output_sha256={p.name: digest(p) for p in sorted(args.output.iterdir())})
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.update_index:
        (BATCH / "summary.tsv").write_bytes((args.output / "summary.tsv").read_bytes())
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("sha256")}, indent=2))


if __name__ == "__main__":
    main()
