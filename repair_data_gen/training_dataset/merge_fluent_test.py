"""Merge the original PMB standard test with the frozen RSR pairs."""
import argparse
from collections import Counter
from pathlib import Path
import re

import pyarrow as pa
import pyarrow.parquet as pq

from .build import HERE, ROOT, SCHEMA, SPLITS, sha256
from .fluent_groups import provenance
from .release_test import read, require, save


def stage(output, paired):
    source = ROOT / SPLITS["test"]
    receipt = read(HERE / "test.publication.json")
    require(sha256(paired) == receipt["verified_files_sha256"]["data/test.parquet"],
            "Input must be the frozen RSR-only release")
    repairs = pq.read_table(paired)
    require(repairs.schema.equals(SCHEMA) and len(repairs) == 1118, "Unexpected RSR input")
    lines = source.read_text(encoding="utf-8").splitlines()
    fluent = []
    # Preserve the source gold even when a downstream graph validator rejects it.
    for i, line in enumerate(lines):
        doc_id = line.strip()
        if not re.fullmatch(r"p\d{2}/d\d{4}", doc_id):
            continue
        sentence, sbn = lines[i+1:i+3]
        require(sentence.strip() and sbn.strip(), f"Empty source record: {doc_id}")
        row = dict.fromkeys(SCHEMA.names)
        row.update(example_id=f"pmb_test::{doc_id}", doc_id=doc_id,
                   text=sentence.strip(), sbn_clean=sbn.strip(), sbn_repair=sbn.strip(),
                   variant="pmb_fluent", label_source="pmb_gold",
                   source="PMB 5.1.0 English standard test")
        fluent.append(provenance(row))
    ids = {r["doc_id"] for r in fluent}
    require(len(ids) == len(fluent) == 1194, "Wrong fluent partition size")
    require(set(repairs["doc_id"].to_pylist()) <= ids, "RSR source outside standard test")
    for split in ("train", "validation"):
        path = HERE / "data" / f"{split}.parquet"
        require(not ids.intersection(pq.read_table(path, columns=["doc_id"])[0].to_pylist()),
                f"Test documents overlap {split}")
    combined = pa.concat_tables([repairs, pa.Table.from_pylist(fluent, schema=SCHEMA)])
    counts = Counter(combined["variant"].to_pylist())
    require(counts == {"pmb_fluent": 1194, "rsr_base": 559, "rsr_editing_expression": 559}
            and len(set(combined["example_id"].to_pylist())) == len(combined),
            "Invalid merged test inventory")
    (output / "data").mkdir(parents=True, exist_ok=False)
    path = output / "data/test.parquet"
    pq.write_table(combined, path, compression="zstd")
    require(pq.read_table(path).equals(combined), "Parquet round-trip mismatch")
    save(output / "test.acceptance.json", dict(
        read(HERE / "test.acceptance.json"), rows=len(combined),
        variant_counts=dict(counts), source_documents=len(ids), rsr_source_documents=505,
        pmb_source=SPLITS["test"], pmb_source_sha256=sha256(source),
        rsr_release_revision=receipt["revision"], rsr_parquet_sha256=sha256(paired),
        parquet_sha256=sha256(path),
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--rsr-parquet", type=Path, required=True)
    args = parser.parse_args()
    stage(args.stage_dir, args.rsr_parquet)
