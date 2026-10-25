"""Build the unified PMB + RSR train Parquet intended for Hugging Face.

Run from the repository root with the project environment:

    /Users/hongxuzhou/miniconda3/envs/repairsbn/bin/python \
        repair_data_gen/training_dataset/build.py

The output contains no Switchboard/NXT data and no TSV source files.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
ROOT = HERE.parents[1]
REPAIR = ROOT / "repair_data_gen"
RSR = REPAIR / "rsr_generation" / "dataset"
sys.path.insert(0, str(REPAIR))
sys.path.insert(0, str(REPAIR / "hard_negatives"))

import holdout  # noqa: E402
from sbn_lin import SPLITS, read_split  # noqa: E402

PMB = ROOT / SPLITS["train"]
RSR_DATA = RSR / "train_rsr.augmented.tsv"
EDITING_MANIFEST = RSR / "train_rsr.editing.manifest.json"
OUT = HERE / "data" / "train.parquet"
MANIFEST = HERE / "manifest.json"

EXPECTED = {
    "pmb_before_holdout": 9552,
    "holdout": 66,
    "pmb_fluent": 9486,
    "rsr_base": 3489,
    "rsr_editing_expression": 1047,
    "total": 14022,
}

SCHEMA = pa.schema([
    ("example_id", pa.string()),
    ("item_id", pa.string()),
    ("doc_id", pa.string()),
    ("variant", pa.string()),
    ("text", pa.string()),
    ("sbn_clean", pa.string()),
    ("sbn_repair", pa.string()),
    ("repair_type", pa.string()),
    ("editing_expression", pa.string()),
    ("ambiguity_type", pa.string()),
    ("repair_onset", pa.int64()),
    ("reparandum_surface", pa.string()),
    ("reparandum_synset", pa.string()),
    ("reparandum_value", pa.string()),
    ("repair_synset", pa.string()),
    ("site_index", pa.int64()),
    ("strategy", pa.string()),
    ("wn_relation", pa.string()),
    ("label_source", pa.string()),
    ("source", pa.string()),
])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nullable(value: str) -> str | None:
    return value if value else None


def nullable_int(value: str) -> int | None:
    return int(value) if value else None


def fluent_rows() -> list[dict]:
    held = holdout.load()
    docs = read_split(PMB)
    if read_split.skipped:
        raise ValueError(f"PMB reader skipped {len(read_split.skipped)} documents")
    if len(docs) != EXPECTED["pmb_before_holdout"]:
        raise ValueError(f"expected 9,552 PMB train documents, found {len(docs)}")
    rows = []
    for doc in docs:
        if doc.doc_id in held:
            continue
        rows.append(dict(
            example_id=f"pmb_train::{doc.doc_id}", item_id=None,
            doc_id=doc.doc_id, variant="pmb_fluent", text=doc.sentence,
            sbn_clean=doc.raw, sbn_repair=doc.raw, repair_type=None,
            editing_expression=None, ambiguity_type=None, repair_onset=None,
            reparandum_surface=None, reparandum_synset=None,
            reparandum_value=None, repair_synset=None, site_index=None,
            strategy=None, wn_relation=None, label_source="pmb_gold",
            source="PMB 5.1.0 English train gold"))
    if len(rows) != EXPECTED["pmb_fluent"]:
        raise ValueError(f"expected 9,486 eligible fluent rows, found {len(rows)}")
    if {r["doc_id"] for r in rows} & held:
        raise ValueError("hard-negative document remains in fluent rows")
    return rows


def rsr_rows() -> list[dict]:
    held = holdout.load()
    with RSR_DATA.open(encoding="utf-8", newline="") as f:
        source = list(csv.DictReader(f, delimiter="\t"))
    rows = []
    for row in source:
        editing = nullable(row["editing_expression"])
        variant = "rsr_editing_expression" if editing else "rsr_base"
        rows.append(dict(
            example_id=row["variant_id"], item_id=row["item_id"],
            doc_id=row["doc_id"], variant=variant, text=row["nl_repair"],
            sbn_clean=row["sbn_clean"], sbn_repair=row["sbn_repair"],
            repair_type=row["repair_type"], editing_expression=editing,
            ambiguity_type=nullable(row["ambiguous"]),
            repair_onset=nullable_int(row["repair_onset"]),
            reparandum_surface=nullable(row["reparandum_surface"]),
            reparandum_synset=nullable(row["reparandum_synset"]),
            reparandum_value=nullable(row["reparandum_value"]),
            repair_synset=nullable(row["repair_synset"]),
            site_index=nullable_int(row["site_index"]),
            strategy=nullable(row["strategy"]),
            wn_relation=nullable(row["wn_relation"]),
            label_source=nullable(row["label_source"]),
            source="synthetic RSR from PMB 5.1.0 English train gold"))
    counts = Counter(r["variant"] for r in rows)
    for variant in ("rsr_base", "rsr_editing_expression"):
        if counts[variant] != EXPECTED[variant]:
            raise ValueError(f"unexpected {variant} count: {counts[variant]}")
    if {r["doc_id"] for r in rows} & held:
        raise ValueError("hard-negative document remains in RSR rows")
    return rows


def main() -> None:
    editing_meta = json.loads(EDITING_MANIFEST.read_text(encoding="utf-8"))
    expected_rsr_sha = next(x["sha256"] for x in editing_meta["outputs"]
                            if x["file"] == RSR_DATA.name)
    if sha256(RSR_DATA) != expected_rsr_sha:
        raise ValueError("augmented RSR file differs from its manifest")

    fluent = fluent_rows()
    rsr = rsr_rows()
    rows = fluent + rsr
    if len(rows) != EXPECTED["total"]:
        raise ValueError(f"expected 14,022 total rows, found {len(rows)}")
    if len({r["example_id"] for r in rows}) != len(rows):
        raise ValueError("example_id is not unique")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, OUT, compression="zstd", row_group_size=10_000,
                   write_statistics=True, write_page_index=True)
    roundtrip = pq.read_table(OUT)
    if not table.equals(roundtrip):
        raise ValueError("Parquet round trip changed the table")

    counts = dict(Counter(rows[i]["variant"] for i in range(len(rows))))
    manifest = {
        "file": str(OUT.relative_to(HERE)),
        "sha256": sha256(OUT),
        "rows": len(rows),
        "columns": SCHEMA.names,
        "variants": counts,
        "distinct_pmb_documents": len({r["doc_id"] for r in fluent}),
        "distinct_rsr_source_documents": len({r["doc_id"] for r in rsr}),
        "hard_negative_holdout": {
            "documents": len(holdout.load()),
            "sha256": holdout.SHA256,
            "overlap": 0,
        },
        "sources": {
            "pmb_train_gold": {"path": str(PMB.relative_to(ROOT)),
                               "sha256": sha256(PMB)},
            "rsr_augmented": {"path": str(RSR_DATA.relative_to(ROOT)),
                              "sha256": sha256(RSR_DATA)},
            "editing_manifest": {"path": str(EDITING_MANIFEST.relative_to(ROOT)),
                                 "sha256": sha256(EDITING_MANIFEST)},
            "builder": {"path": str(SCRIPT.relative_to(ROOT)),
                        "sha256": sha256(SCRIPT)},
        },
        "targets": {
            "C1": "variant == 'pmb_fluent'; target = sbn_clean",
            "C2": "all rows; target = sbn_clean",
            "C3": "all rows; target = sbn_repair",
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"rows": len(rows), "variants": counts,
                      "sha256": manifest["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
