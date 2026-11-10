"""Apply the recorded test review policy to machine diagnostics, without labels."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .analyse_split import ROOT, GEN, read, write, sha, pair_key, failure_flags, read_split
from .prepare_dev_review import decision, normalise_surface
import check_repair_part


def normalise_candidate(row):
    normalised, aligned = normalise_surface(row)
    # The legacy fallback can reinterpret a word-final apostrophe as a quote
    # or strip punctuation before a missing citation. Keep the model text
    # whenever gold alignment is incomplete; surface review resolves it later.
    return (normalised if aligned else dict(row)), aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workspace = ROOT / "repair_data_gen/splits/test"
    raw_summary = json.loads((args.analysis / "summary.json").read_text())
    if raw_summary["integrity_errors"]:
        raise ValueError("source integrity errors must be resolved before review")
    policy_path = workspace / "review_policy.json"
    policy = json.loads(policy_path.read_text())
    assert policy["pair_overlap_action"] == "report_only"
    rows = read(args.analysis / "all_checked.tsv")
    candidates, exclusions, no_site, dispositions, changes, unaligned = [], [], [], [], [], []
    for original in rows:
        row = dict(original)
        eligible, reason = decision(row, policy)
        disposition = dict(request_id=row["request_id"], repair_type=row["repair_type"],
                           raw_machine_pass=row["machine_pass"], policy_eligible=str(eligible),
                           reason=reason,
                           raw_failure_checks=";".join(failure_flags(row)) if row["status"] == "ok" else "")
        dispositions.append(disposition)
        if row["status"] == "no_site":
            no_site.append(row)
        elif not eligible:
            exclusions.append({**row, **disposition})
        else:
            normalised, aligned = normalise_candidate(row)
            assert all(normalised[k] == row[k] for k in
                       ("sbn_clean", "sbn_repair", "site_index", "repair_synset", "request_id"))
            normalised.update(nl_repair_model=row["nl_repair"], repair_onset_model=row["repair_onset"],
                              policy_reason=reason, raw_machine_pass=row["machine_pass"],
                              surface_aligned=str(aligned),
                              normalised_repair_part=";".join(check_repair_part.problems(normalised)) or "ok")
            candidates.append(normalised)
            if not aligned:
                unaligned.append(normalised)
            if (normalised["nl_repair"], normalised["repair_onset"]) != (row["nl_repair"], row["repair_onset"]):
                changes.append(dict(request_id=row["request_id"], repair_type=row["repair_type"],
                                    original=row["nl_repair"], normalised=normalised["nl_repair"],
                                    original_onset=row["repair_onset"], normalised_onset=normalised["repair_onset"],
                                    reason=reason))
    # Verify both source partitions independently of replacement-pair diagnostics.
    source_paths = {"test": ROOT / "data/pmb-5.1.0/split/en/test/standard.sbn",
                    "dev": ROOT / "data/pmb-5.1.0/split/en/dev/standard.sbn",
                    "train": ROOT / "data/pmb-5.1.0/split/en/train/gold.sbn"}
    sources = {name: read_split(path) for name, path in source_paths.items()}
    isolation = {}
    for name in ("train", "dev"):
        ids, texts = {d.doc_id for d in sources[name]}, {d.sentence for d in sources[name]}
        isolation[name] = dict(document_overlap=sum(d.doc_id in ids for d in sources["test"]),
                               exact_text_overlap=sum(d.sentence in texts for d in sources["test"]))
        assert isolation[name] == dict(document_overlap=0, exact_text_overlap=0)
    references = {
        "train": ROOT / "repair_data_gen/rsr_generation/dataset/train_rsr.base.tsv",
        "dev": ROOT / "repair_data_gen/splits/dev/dataset/dev_rsr.base.tsv",
    }
    reference = defaultdict(list)
    unresolved, overlaps = [], []
    for split, path in references.items():
        for row in read(path):
            try:
                reference[pair_key(row)].append((split, row["item_id"]))
            except (ValueError, IndexError, TypeError) as error:
                unresolved.append(dict(split=split, item_id=row["item_id"], error=str(error)))
    for row in candidates:
        try:
            key = pair_key(row)
            for direction, candidate in (("same_direction", key), ("reverse_direction", (*key[:2], key[3], key[2]))):
                for split in ("train", "dev"):
                    matches = [item for ref_split, item in reference.get(candidate, []) if ref_split == split]
                    if matches:
                        overlaps.append(dict(request_id=row["request_id"], repair_type=row["repair_type"],
                                             comparison_split=split, direction=direction,
                                             key=json.dumps(key), comparison_item_ids=";".join(matches)))
        except (ValueError, IndexError, TypeError) as error:
            unresolved.append(dict(split="test", item_id=row["request_id"], error=str(error)))
    duplicates = []
    for name, key in (("same_input", lambda r: r["nl_repair"]),
                      ("same_source_type_site", lambda r: (r["doc_id"], r["repair_type"], r["site_index"]))):
        groups = defaultdict(list)
        for row in candidates:
            groups[key(row)].append(row["request_id"])
        duplicates.extend(dict(check=name, request_ids=";".join(ids)) for ids in groups.values() if len(ids) > 1)
    assert len(candidates) + len(exclusions) + len(no_site) == len(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "review_candidates.tsv", candidates)
    write(args.out / "excluded.tsv", exclusions)
    write(args.out / "no_site.tsv", no_site)
    write(args.out / "dispositions.tsv", dispositions)
    write(args.out / "surface_changes.tsv", changes,
          ["request_id", "repair_type", "original", "normalised", "original_onset", "normalised_onset", "reason"])
    write(args.out / "unaligned.tsv", unaligned, list(candidates[0]))
    write(args.out / "duplicates.tsv", duplicates, ["check", "request_ids"])
    write(args.out / "pair_collisions.tsv", overlaps,
          ["request_id", "repair_type", "comparison_split", "direction", "key", "comparison_item_ids"])
    write(args.out / "pair_unresolved.tsv", unresolved, ["split", "item_id", "error"])
    inputs = [policy_path, args.analysis / "summary.json", args.analysis / "all_checked.tsv",
              *source_paths.values(), *references.values()]
    code = [Path(__file__), Path(__file__).with_name("prepare_dev_review.py"),
            Path(__file__).with_name("analyse_split.py"), GEN / "dataset/apply_fixes.py"]
    summary = dict(status="machine-policy candidates only; full human review outstanding", returned=len(rows),
                   model_ok=sum(r["status"] == "ok" for r in rows), raw_machine_pass=raw_summary["machine_pass"],
                   policy_eligible=len(candidates), excluded=len(exclusions), no_site=len(no_site),
                   transport_missing=raw_summary["missing"],
                   surface_only_reclassified=sum(r["raw_machine_pass"] != "True" for r in candidates),
                   surface_changes=len(changes), not_fully_aligned=len(unaligned),
                   normalised_repair_part=dict(Counter(r["normalised_repair_part"] for r in candidates)),
                   source_isolation=isolation, duplicates=duplicates,
                   pair_policy="report_only; no pair-based exclusions", pair_overlap_exclusions=0,
                   pair_overlap_items={s: len({r["request_id"] for r in overlaps if r["comparison_split"] == s})
                                       for s in references}, pair_unresolved=dict(Counter(r["split"] for r in unresolved)),
                   context_near_duplicate_check="not_run; no claim of absence",
                   per_type={t: dict(model_ok=sum(r["repair_type"] == t and r["status"] == "ok" for r in rows),
                                     raw_machine_pass=sum(r["repair_type"] == t and r["machine_pass"] == "True" for r in rows),
                                     policy_eligible=sum(r["repair_type"] == t for r in candidates),
                                     excluded=sum(r["repair_type"] == t for r in exclusions))
                             for t in sorted({r["repair_type"] for r in rows})},
                   input_sha256={str(p): sha(p) for p in inputs}, code_sha256={str(p): sha(p) for p in code})
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"input_sha256", "code_sha256"}}, indent=2))


if __name__ == "__main__":
    main()
