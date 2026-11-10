"""Inventory unresolved test requests without changing outputs or acceptance."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "repair_data_gen/splits/test"
GROUPS = (
    ("timeout", "T", "analysis/import_validation/missing.tsv"),
    ("no_site", "N", "analysis/review_policy_v1/no_site.tsv"),
    ("excluded", "E", "analysis/review_policy_v1/excluded.tsv"),
    ("surface_hold", "S", "analysis/review_policy_v1/unaligned.tsv"),
)


def main():
    inputs, items, counts = {}, [], {}
    for group, prefix, relative in GROUPS:
        path = TEST / relative
        inputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline="") as handle:
            rows = sorted(csv.DictReader(handle, delimiter="\t"),
                          key=lambda row: (row["doc_id"] if group == "timeout"
                                           else row["request_id"]))
        counts[group] = len(rows)
        for number, row in enumerate(rows, 1):
            items.append({
                "review_id": f"{prefix}{number:03d}",
                "category": group,
                "request_id": row["request_id"],
                "source_table": relative,
                "source_record": row,
            })
    ids = [item["request_id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("An unresolved request occurs in multiple groups")
    output = TEST / "recovery_v1/inventory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "inventory_only; no recovery or acceptance decisions",
        "scope": "29 unsuccessful requests and 10 eligible surface holds",
        "surface_holds_already_in_review_pool": True,
        "recovery_policy": "Keep original records; record approved recovery separately by request_id; validate before human review.",
        "input_sha256": inputs,
        "counts": counts,
        "batches": [
            {"batch_id": f"B{offset // 8 + 1:02d}",
             "review_ids": [item["review_id"] for item in items[offset:offset + 8]]}
            for offset in range(0, len(items), 8)
        ],
        "items": items,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "counts": counts}))


if __name__ == "__main__":
    main()
