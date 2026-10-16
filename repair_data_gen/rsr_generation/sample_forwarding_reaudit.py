"""Fresh forwarding audit on the frozen set (2026-10-13).

In the main-run audit (`audit/train_full`) 11 of 15 forwarding items were
labelled `unsure` for meaning, nearly all over one question: which subject
the predicate in each version points to. That was settled on 2026-10-08
(each version's predicate points to that version's subject; DECISIONS.md),
and reflexives on 2026-10-13, but the labels were not revisited, so the
meaning rate for forwarding rests on 4 items.

This draws 15 forwarding items from `dataset/train_rsr.base.tsv`, as
released (surface fixes applied), leaving out every document already shown
in a forwarding audit. Every item in the set passed all checks in
`assemble.py`, so no check columns are needed. Not blind to type: every
item is forwarding.

    python3 sample_forwarding_reaudit.py
    python3 audit.py serve audit/train_forwarding_reaudit
    python3 audit.py stats audit/train_forwarding_reaudit
"""
import csv
import random
import sys
from pathlib import Path

from audit import ITEM_FIELDS

HERE = Path(__file__).parent
DATASET = HERE / "dataset" / "train_rsr.base.tsv"
SEEN = [HERE / "audit" / d / "items.tsv" for d in ("r2", "train_full")]
OUT = HERE / "audit" / "train_forwarding_reaudit"
N = 15


def read(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    if (OUT / "items.tsv").exists():
        sys.exit(f"{OUT}/items.tsv exists; not redrawing over a started audit")
    seen = {r["doc_id"] for p in SEEN for r in read(p) if r["repair_type"] == "forwarding"}
    pool = sorted((r for r in read(DATASET) if r["repair_type"] == "forwarding"
                   and r["doc_id"] not in seen), key=lambda r: r["item_id"])
    take = random.Random(13).sample(pool, N)

    OUT.mkdir(parents=True)
    with open(OUT / "items.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ITEM_FIELDS, delimiter="\t")
        w.writeheader()
        for k, r in enumerate(take):
            w.writerow({**{f: r.get(f, "") for f in ITEM_FIELDS},
                        "audit_id": f"a{k:04d}", "source": f"{DATASET.name}#{r['item_id']}"})
    print(f"pool {len(pool)} (left out {len(seen)} documents already audited) "
          f"-> {N} items in {OUT}/items.tsv")


if __name__ == "__main__":
    main()
