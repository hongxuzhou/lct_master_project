"""Create a text-only calibration queue without changing the original review."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random

TEST = Path(__file__).resolve().parents[1] / "splits/test"
SOURCE = TEST / "audit/test_full_review_v1"
OUT = TEST / "audit/test_calibration_v1"
SEED = 20261109
# Matched borderline examples and clearer controls, chosen before rescoring.
GROUPS = {
    "names_and_vocatives": "0004 0038 0048 0235 0400 0419 0466 0480 0495 0499",
    "number_approximations": "0001 0047 0052 0145 0148 0213 0267 0325 0326 0471",
    "forwarding_and_independent_clauses": "0022 0063 0077 0087 0118 0223 0287 0299 0347 0423",
    "lexical_replacement_and_listing": "0033 0041 0068 0103 0130 0207 0294 0435 0450 0478",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    items_path = SOURCE / "items.tsv"
    first_manifest = json.loads((SOURCE / "manifest.json").read_bytes())
    if digest(items_path) != first_manifest["items_sha256"]:
        raise ValueError("Original queue differs from its manifest")
    with items_path.open(newline="", encoding="utf-8") as handle:
        items = {r["audit_id"]: r for r in csv.DictReader(handle, delimiter="\t")}
    selected = [(f"test{i}", group) for group, ids in GROUPS.items() for i in ids.split()]
    if len({i for i, _ in selected}) != len(selected):
        raise ValueError("Repeated original ID")
    rows = [(items[i], group) for i, group in selected]
    random.Random(SEED).shuffle(rows)
    OUT.mkdir()  # Never replace a queue that may already have labels.
    mapping = {}
    with (OUT / "items.tsv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["audit_id", "nl_repair"], delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for n, (row, group) in enumerate(rows, 1):
            new_id = f"testcal{n:03d}"
            writer.writerow(dict(audit_id=new_id, nl_repair=row["nl_repair"]))
            mapping[new_id] = dict(original_id=row["audit_id"], group=group)
    (OUT / "manifest.json").write_text(json.dumps(dict(
        purpose="text-only calibration; targeted comparison, not a representative sample",
        rows=len(rows), groups=dict(Counter(g for _, g in rows)), seed=SEED,
        original_items_sha256=digest(items_path),
        items_sha256=digest(OUT / "items.tsv"), mapping=mapping,
    ), indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} calibration items: {OUT}")


if __name__ == "__main__":
    main()
