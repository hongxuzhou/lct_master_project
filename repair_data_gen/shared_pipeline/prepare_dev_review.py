"""Apply the dev review policy and prepare a blind, append-safe audit queue.

Raw synthesis outputs, first-pass diagnostics, semantic targets and labels
are never edited. Surface checks remain visible even when policy permits
them. This is an audit pool, not a frozen evaluation dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re

from .analyse_split import GEN, ROOT, read, write, sha, part

spec = importlib.util.spec_from_file_location("train_surface_helpers", GEN / "dataset/apply_fixes.py")
train_surface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_surface)

STRUCTURAL_PART = ("R5:", "reparandum identical", "no CORRECTION", "edge label:", "error:")
AUX = {"am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did"}


def tokens(text):
    # Citations are editorial surface annotations, not SBN number constants.
    text = re.sub(r"\[\d+\]", "", text)
    return re.findall(r"[^\W_]+", part.norm(text).casefold())


def surface_eligible(row):
    """Keep gold content and direction, allowing specified surface variation.

    Tense auxiliaries may be realised or repeated differently; negation is
    always protected. Main verb forms, numbers, names and other content
    words are never substituted here. Formal naturalness/meaning is human
    reviewed, including doubtful auxiliary scope or aspect.
    """
    onset = int(row["repair_onset"])
    gold = tokens(row["nl_clean"])
    left, right = tokens(row["nl_repair"][:onset-2]), tokens(row["nl_repair"][onset:])
    if not right:
        return False, "empty repair text"
    variants = [(gold, left, right, "surface_equivalent")]
    if right[0] in {"a", "an", "the"}:
        variants.append((gold, left, right[1:], "repeated_boundary_article"))
    # Never erase an explicitly realised finite auxiliary on the repair
    # side: that would wrongly rescue e.g. gold "was" -> final "is".
    if row["repair_type"] == "tense" and right[0] not in AUX:
        variants.append((*([w for w in seq if w not in AUX] for seq in (gold, left, right)),
                         "tense_auxiliary_realisation"))
    for target, before, tail, reason in variants:
        if not tail or len(tail) > len(target) or target[-len(tail):] != tail:
            continue
        prefix = target[:-len(tail)]
        if before[:len(prefix)] != prefix or len(before) <= len(prefix):
            continue
        if reason == "repeated_boundary_article" and right[0] not in prefix:
            continue
        return True, reason
    return False, "changed gold content, prefix, or repair direction"


def normalise_surface(row):
    """Reuse train's pure alignment helpers; retain model text on no alignment."""
    gold, original, onset = row["nl_clean"], row["nl_repair"], int(row["repair_onset"])
    # Treat paired single quotes as quotation marks, preserving contractions.
    quote = lambda s: re.sub(r"(?<!\w)'|'(?!\w)", '"', s.translate(str.maketrans("’‘“”", "''\"\"")))
    left, right = quote(original[:onset-2]), quote(original[onset:])
    text, offset = left + ", " + right, len(left) + 2
    start = train_surface.gold_start(gold, right)
    insertion = train_surface.strip_prefix(left, gold[:start]) if start is not None else None
    aligned = insertion is not None
    if aligned:
        insertion = train_surface.gold_apostrophes(insertion, gold)
        tail = gold[start:]
        k, km = train_surface.first_letter(tail), train_surface.first_letter(right)
        if k is not None and km is not None:
            tail = tail[:k] + right[km] + tail[k+1:]
        text = gold[:start] + insertion + ", " + tail
        offset = start + len(insertion) + 2
    else:
        fm, gm = train_surface.final_mark(text), train_surface.final_mark(gold)
        if fm.strip() != gm.strip():
            text = text[:len(text)-len(fm)] + gm.rstrip()
        text = train_surface.gold_apostrophes(text, gold)
    clone = dict(row, nl_repair=text, repair_onset=str(offset))
    # A normalisation is allowed only if it preserves this same content check.
    assert surface_eligible(clone)[0], row["request_id"]
    assert clone["sbn_repair"] == row["sbn_repair"]
    return clone, aligned


def decision(row, policy):
    if row["status"] != "ok":
        return False, row["status"] or "missing status"
    failures = [key for key in policy["hard_checks"] if row.get(key) != "ok"]
    failures += [key for key, value in (("wn_relation", "same_synset"), ("place_inside", "yes"))
                 if row.get(key) == value]
    if any(marker in row["repair_part"] for marker in STRUCTURAL_PART):
        failures.append("structural_repair_part")
    if failures:
        return False, ";".join(failures)
    return surface_eligible(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT / "repair_data_gen/splits/dev")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    policy_path = workspace / "review_policy.json"
    policy = json.loads(policy_path.read_text())
    assert policy["pair_overlap_action"] == "report_only"
    checked = workspace / "analysis/first_pass/all_checked.tsv"
    source_summary = workspace / "analysis/first_pass/summary.json"
    source = json.loads(source_summary.read_text())
    assert not source["integrity_errors"]
    for path, digest in source["input_sha256"].items():
        assert sha(path) == digest, f"analysis input changed: {path}"
    out = workspace / "analysis" / policy["version"]
    out.mkdir(parents=True, exist_ok=True)
    pool, excluded, changes, revived = [], [], [], []
    for row in read(checked):
        eligible, reason = decision(row, policy)
        record = dict(row, review_eligible=str(eligible), review_reason=reason)
        if not eligible:
            excluded.append(record)
            continue
        normalised, aligned = normalise_surface(record)
        normalised.update(nl_repair_model=row["nl_repair"], repair_onset_model=row["repair_onset"],
                          surface_aligned=str(aligned), policy_version=policy["version"])
        pool.append(normalised)
        if row["machine_pass"] != "True":
            revived.append(normalised)
        if normalised["nl_repair"] != row["nl_repair"]:
            changes.append(normalised)
    write(out / "review_pool.tsv", pool)
    write(out / "excluded.tsv", excluded)
    write(out / "surface_changes.tsv", changes)
    write(out / "reclassified.tsv", revived)
    # Directly reuse the existing frontend schema, without rewriting raw
    # check columns to trick the legacy auto_pass predicate.
    from audit import ITEM_FIELDS
    audit_dir = workspace / "audit" / policy["version"]
    audit_dir.mkdir(parents=True, exist_ok=True)
    items_path = audit_dir / "items.tsv"
    items = read(items_path) if items_path.exists() else []
    by_id = {r["request_id"]: r for r in items}
    fresh = []
    for row in pool:
        item = {key: row.get(key, "") for key in ITEM_FIELDS}
        item.update(audit_id="dev_" + hashlib.sha256(row["request_id"].encode()).hexdigest()[:12],
                    source=str((out / "review_pool.tsv").relative_to(ROOT)))
        if row["request_id"] in by_id:
            assert item == by_id[row["request_id"]], "existing audit item changed; keep its labels and version the queue"
        else:
            fresh.append(item)
    assert set(by_id) <= {r["request_id"] for r in pool}, "existing audit item removed"
    random.Random(20261027).shuffle(fresh)
    items += fresh
    if not (audit_dir / "labels.jsonl").exists():
        items.sort(key=lambda r: r["request_id"])
        random.Random(20261027).shuffle(items)
    assert len({r["audit_id"] for r in items}) == len(items)
    write(items_path, items, list(ITEM_FIELDS))
    summary = dict(status="ready for formal human review; not frozen", policy=policy,
                   returned=len(pool)+len(excluded), review_pool=len(pool),
                   excluded=len(excluded), no_site=sum(r["status"] == "no_site" for r in excluded),
                   rejected_model_ok=sum(r["status"] == "ok" for r in excluded),
                   reclassified_from_legacy_failure=len(revived),
                   normalised_surfaces=len(changes), unaligned_surface_for_human_review=sum(r["surface_aligned"] == "False" for r in pool),
                   pair_overlap_exclusions=0, context_near_duplicate_check="deferred",
                   per_type=dict(Counter(r["repair_type"] for r in pool)),
                   input_sha256={str(p): sha(p) for p in (checked, source_summary, policy_path)},
                   code_sha256={str(Path(__file__)): sha(__file__), str(GEN / "dataset/apply_fixes.py"): sha(GEN / "dataset/apply_fixes.py")},
                   audit_items_sha256=sha(items_path))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"input_sha256", "code_sha256", "policy"}}, indent=2))


if __name__ == "__main__":
    main()
