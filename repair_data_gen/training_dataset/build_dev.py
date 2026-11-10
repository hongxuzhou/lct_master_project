"""Assemble standard PMB dev and curated RSR with the existing train schema.

Run from the repository root in the repairsbn environment:

    python -m repair_data_gen.training_dataset.build_dev

This local build does not freeze the split or publish it to Hugging Face.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from repair_data_gen.training_dataset.build import (  # noqa: E402
    SCHEMA, holdout, nullable, nullable_int, read_split, sha256,
)
from repair_data_gen.training_dataset.fluent_groups import accepted_rows, provenance

DEV = ROOT / "repair_data_gen/splits/dev"
PMB = ROOT / "data/pmb-5.1.0/split/en/dev/standard.sbn"
RSR = DEV / "dataset/dev_rsr.augmented.tsv"
OUT = HERE / "data/validation.parquet"
MANIFEST = HERE / "validation.manifest.json"
EXPECTED = {"pmb_fluent": 1194, "rsr_base": 390, "rsr_editing_expression": 122,
            "fluent_split_reference": 40}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    base_manifest = DEV / "manifests/base_build.json"
    marked_manifest = DEV / "manifests/marked_build.json"
    base_meta = json.loads(base_manifest.read_text(encoding="utf-8"))
    marked_meta = json.loads(marked_manifest.read_text(encoding="utf-8"))
    train_manifest = HERE / "manifest.json"
    train_meta = json.loads(train_manifest.read_text(encoding="utf-8"))
    train = HERE / train_meta["file"]
    require(sha256(train) == train_meta["sha256"], "train differs from its manifest")
    require(pq.read_schema(train).equals(SCHEMA, check_metadata=True),
            "train Parquet schema differs from the shared schema")
    require(sha256(ROOT / base_meta["file"]) == base_meta["sha256"],
            "dev base differs from its build manifest")
    for path, digest in marked_meta["input_sha256"].items():
        require(sha256(Path(path)) == digest, f"marked build input changed: {path}")
    for output in marked_meta["outputs"]:
        require(sha256(ROOT / output["file"]) == output["sha256"],
                f"dev component differs from its build manifest: {output['file']}")
    require(any(ROOT / output["file"] == RSR for output in marked_meta["outputs"]),
            "augmented dev RSR is absent from the marked build manifest")

    source_paths = {
        "pmb_dev_standard": PMB,
        "rsr_base": ROOT / base_meta["file"],
        "rsr_augmented": RSR,
        "base_manifest": base_manifest,
        "marked_manifest": marked_manifest,
        "marker_policy": DEV / "marker_policy.json",
        "train_schema_reference": train,
        "train_manifest": train_manifest,
        "hard_negative_holdout": holdout.HOLDOUT,
        "builder": Path(__file__).resolve(),
        "shared_schema_and_helpers": HERE / "build.py",
        "pmb_reader": ROOT / "repair_data_gen/sbn_lin.py",
        "holdout_loader": Path(holdout.__file__).resolve(),
        "fluent_group_acceptance": HERE/'fluent_group.acceptance.json',
        "fluent_group_adapter": HERE/'fluent_groups.py',
        "fluent_group_source": ROOT/'repair_data_gen/fluent_group_synthesis_v1/dev.candidates.jsonl',
    }
    sources = {key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
               for key, path in source_paths.items()}

    docs = read_split(PMB)
    require(not read_split.skipped, "PMB reader skipped dev documents")
    require(len(docs) == EXPECTED["pmb_fluent"], "unexpected fluent dev count")
    by_doc = {doc.doc_id: doc for doc in docs}
    require(len(by_doc) == len(docs), "duplicate PMB dev document IDs")
    fluent = []
    for doc in docs:
        row = dict.fromkeys(SCHEMA.names)
        row.update(example_id=f"pmb_dev::{doc.doc_id}", doc_id=doc.doc_id,
                   variant="pmb_fluent", text=doc.sentence,
                   sbn_clean=doc.raw, sbn_repair=doc.raw,
                   label_source="pmb_gold", source="PMB 5.1.0 English standard dev")
        fluent.append(row)

    with RSR.open(encoding="utf-8", newline="") as handle:
        augmented = list(csv.DictReader(handle, delimiter="\t"))
    rsr = []
    for row in augmented:
        doc = by_doc.get(row["doc_id"])
        require(doc is not None, f"RSR source is outside PMB dev: {row['item_id']}")
        require((row["nl_clean"], row["sbn_clean"]) == (doc.sentence, doc.raw),
                f"RSR clean source differs from PMB dev: {row['item_id']}")
        editing = nullable(row["editing_expression"])
        rsr.append(dict(
            example_id=row["variant_id"], item_id=row["item_id"],
            doc_id=row["doc_id"],
            variant="rsr_editing_expression" if editing else "rsr_base",
            text=row["nl_repair"], sbn_clean=row["sbn_clean"], sbn_repair=row["sbn_repair"],
            repair_type=row["repair_type"], editing_expression=editing,
            ambiguity_type=nullable(row["ambiguous"]),
            repair_onset=nullable_int(row["repair_onset"]),
            reparandum_surface=nullable(row["reparandum_surface"]),
            reparandum_synset=nullable(row["reparandum_synset"]),
            reparandum_value=nullable(row["reparandum_value"]),
            repair_synset=nullable(row["repair_synset"]),
            site_index=nullable_int(row["site_index"]), strategy=nullable(row["strategy"]),
            wn_relation=nullable(row["wn_relation"]), label_source=nullable(row["label_source"]),
            source="synthetic RSR from PMB 5.1.0 English standard dev"))

    rows = [provenance(row) for row in fluent+rsr] + accepted_rows('validation', SCHEMA)
    counts = Counter(row["variant"] for row in rows)
    require(counts == Counter(EXPECTED), f"unexpected variant counts: {dict(counts)}")
    require(len({row["example_id"] for row in rows}) == len(rows), "duplicate example IDs")
    required_fields = ("example_id", "doc_id", "variant", "text", "sbn_clean", "sbn_repair")
    require(all(all(row[field] for field in required_fields) for row in rows),
            "empty required field")
    held = holdout.load()
    require(not {row["doc_id"] for row in rows} & held, "hard-negative overlap in dev")

    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, OUT, compression="zstd", row_group_size=10_000,
                   write_statistics=True, write_page_index=True)
    require(table.equals(pq.read_table(OUT), check_metadata=True),
            "Parquet round trip changed the table")
    for key, path in source_paths.items():
        require(sha256(path) == sources[key]["sha256"], f"source changed during build: {path}")

    manifest = {
        "dataset_version": "fluent-groups-v1", "schema_version": 2,
        "status": "assembled locally; final validation and freezing pending",
        "split": "validation", "pmb_split": "dev_standard",
        "file": str(OUT.relative_to(HERE)), "sha256": sha256(OUT),
        "rows": len(rows), "columns": SCHEMA.names,
        "variants": dict(counts),
        "origins": dict(Counter(r['data_origin'] for r in rows)),
        "fluent_construction_types": dict(Counter(r['construction_type'] for r in rows
                                                 if r['variant']=='fluent_split_reference')),
        "distinct_pmb_documents": len(by_doc),
        "distinct_rsr_source_documents": len({row["doc_id"] for row in rsr}),
        "row_order": "PMB, augmented RSR, then accepted fluent split-reference; no shuffle",
        "targets": {
            "C1": "variant in {'pmb_fluent', 'fluent_split_reference'}; target = sbn_clean",
            "C2": "all rows; target = sbn_clean",
            "C3": "all rows; target = sbn_repair",
        },
        "assembly_checks": {
            "source_component_hashes": "passed", "train_schema_match": True,
            "rsr_clean_source_matches_pmb_dev": len(rsr),
            "unique_example_ids": len(rows), "hard_negative_overlap": 0,
            "parquet_roundtrip": "passed", "inputs_unchanged": True,
        },
        "pending": ["final assembled-split validation", "model-template token-length audit",
                    "freeze", "Hugging Face publication and readback"],
        "sources": sources, "pyarrow_version": pa.__version__,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "file", "rows", "variants", "sha256")},
                     indent=2))


if __name__ == "__main__":
    main()
