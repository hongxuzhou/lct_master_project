"""Prepare a stable, blinded full test review queue after local recovery."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random

from .analyse_split import ROOT, read, write, sha, pair_key
from audit import ITEM_FIELDS
from sbn_lin import read_split


TEST = ROOT / "repair_data_gen/splits/test"
RECOVERY = TEST / "recovery_v1"
OUT = TEST / "audit/test_full_review_v1"


def main():
    source = TEST / "analysis/review_policy_v1/review_candidates.tsv"
    recovered = RECOVERY / "reconstructed.checked.tsv"
    validation = RECOVERY / "validation.json"
    report = json.loads(validation.read_text())
    for key in ("input_sha256", "output_sha256", "code_sha256"):
        for name, digest in report[key].items():
            if sha(Path(name)) != digest:
                raise ValueError(f"Recovery input or output changed: {name}")
    pool = {row["request_id"]: row for row in read(source)}
    original_ids = set(pool)
    for row in read(recovered):
        if row["policy_eligible"] == "True":
            pool[row["request_id"]] = row
    rows = list(pool.values())
    if len(rows) != len(pool):
        raise ValueError("Duplicate request IDs")
    sites = [(r["doc_id"], r["repair_type"], r["site_index"]) for r in rows]
    if len(sites) != len(set(sites)):
        raise ValueError("Repeated source/type/site")
    source_train = read_split(ROOT / "data/pmb-5.1.0/split/en/train/gold.sbn")
    source_dev = read_split(ROOT / "data/pmb-5.1.0/split/en/dev/standard.sbn")
    for name, docs in (("train", source_train), ("dev", source_dev)):
        ids, texts = {d.doc_id for d in docs}, {d.sentence for d in docs}
        if any(r["doc_id"] in ids or r["nl_clean"] in texts for r in rows):
            raise ValueError(f"Source overlap with {name}")
    pairs = []
    for split, path in (("train", ROOT / "repair_data_gen/rsr_generation/dataset/train_rsr.base.tsv"),
                        ("dev", ROOT / "repair_data_gen/splits/dev/dataset/dev_rsr.base.tsv")):
        references = Counter(pair_key(r) for r in read(path))
        for row in rows:
            key = pair_key(row)
            for direction, candidate in (("same", key), ("reverse", (*key[:2], key[3], key[2]))):
                if references[candidate]:
                    pairs.append(dict(request_id=row["request_id"], comparison_split=split,
                                      direction=direction, reference_count=references[candidate]))
    rows.sort(key=lambda r: r["request_id"])
    random.Random(20261106).shuffle(rows)
    items = []
    for index, row in enumerate(rows):
        row["audit_id"] = f"test{index:04d}"
        row["source"] = "test_recovery_v1" if row.get("recovery_id") else "test_first_attempt"
        items.append({field: row.get(field, "") for field in ITEM_FIELDS})
    OUT.mkdir(parents=True, exist_ok=True)
    items_path = OUT / "items.tsv"
    # Never change a queue underneath previously collected human labels.
    if (OUT / "labels.jsonl").exists():
        if read(items_path) != items:
            raise ValueError("Reviewed queue differs; use a new version")
    else:
        write(items_path, items, ITEM_FIELDS)
    write(OUT / "review_pool.tsv", rows)
    write(OUT / "pair_collisions.tsv", pairs)
    # E017 is explicitly deferred for naturalness, not machine-eligible.
    # Preserve it as a separate supplemental item, never add it to the pool.
    inventory = json.loads((RECOVERY / "inventory.json").read_text())
    flagged = next(i["source_record"] for i in inventory["items"] if i["review_id"] == "E017")
    supplemental = {field: flagged.get(field, "") for field in ITEM_FIELDS}
    supplemental.update(audit_id="deferred_E017", source="deferred_not_machine_eligible")
    supplemental["nl_repair"] = "The rock group Yes comes, the rock group Spock's Beard comes from Japan."
    (OUT / "deferred_naturalness").mkdir(exist_ok=True)
    write(OUT / "deferred_naturalness/items.tsv", [supplemental], ITEM_FIELDS)
    manifest = {
        "status": "full human review pending; not a selected base or final test",
        "rows": len(rows), "original_review_candidates": len(original_ids),
        "new_recovered_candidates": len(set(pool) - original_ids),
        "per_type": dict(Counter(r["repair_type"] for r in rows)),
        "seed": 20261106,
        "supplemental_E017": "separate deferred_naturalness queue; not machine eligible",
        "source_isolation": "zero train/dev source-document and exact-sentence overlap",
        "near_duplicate_check": "not run",
        "pair_overlap": "report only; no exclusions",
        "input_sha256": {str(p): sha(p) for p in (source, recovered, validation)},
        "items_sha256": sha(items_path), "review_pool_sha256": sha(OUT / "review_pool.tsv"),
        "code_sha256": {str(Path(__file__)): sha(Path(__file__))},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
