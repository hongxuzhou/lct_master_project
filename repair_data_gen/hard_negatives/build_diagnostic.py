"""Build the local hard-negative scoring inputs without changing PMB or the hold-out."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

from holdout import HOLDOUT, load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "repair_data_gen"), str(ROOT / "evaluation")]
from build_gold import process_row
from sbn_env import sbn_src_dir
from sbn_lin import parse_sbn_line, read_split

PMB = ROOT / "data/pmb-5.1.0/split/en/train/gold.sbn"
OUT = HERE / "diagnostic_v2"
PATCHES = {
    "p05/d0919": ("sbn_repair", (
        ("CONTINUATION <3 pig", "CONTINUATION <2 pig"),
        ("CONTINUATION <5 cat", "CONTINUATION <4 cat"),
    )),
    "p17/d2285": ("sbn_repair", (("typical.a.01 AttributeOf +2", "typical.a.01 AttributeOf +1"),)),
    "p05/d2605": ("sentence", (("put in on the chair", "put it on the chair"),)),
    "p40/d1979": ("sbn_repair", ((
        "NEGATION <1 entity.n.01 EQU -2 Instrument +1 united_states_dollar.n.01",
        "NEGATION <1 entity.n.01 EQU -2 Asset +1 united_states_dollar.n.01",
    ),)),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    if path.exists():
        require(path.read_bytes() == data, f"Existing export differs: {path}")
    else:
        with path.open("xb") as f:
            f.write(data)


def main():
    held = load()
    docs = {d.doc_id: d for d in read_split(PMB)}
    require(not read_split.skipped and len(docs) == 9552, "Unexpected PMB train inventory")
    with HOLDOUT.open(encoding="utf-8", newline="") as handle:
        inventory = list(csv.DictReader(handle, delimiter="\t"))
    require(len(inventory) == len(held) == 66 and {r["doc_id"] for r in inventory} == held,
            "Unexpected hold-out inventory")
    require(PATCHES.keys() <= held, "Invalid adjudication IDs")

    ready, records = [], {}
    for item in inventory:
        doc_id = item["doc_id"]
        doc = docs[doc_id]
        require(doc.sentence == item["sentence"], f"PMB text differs: {doc_id}")
        row = dict(id=f"hard_negative::{doc_id}", doc_id=doc_id, category=item["label"],
                   sentence=doc.sentence, sbn_repair=doc.raw)
        record = dict(category=item["label"], status="ready", changes=[])
        if doc_id in PATCHES:
            field, changes = PATCHES[doc_id]
            before = row[field]
            for old, new in changes:
                require(row[field].count(old) == 1, f"Patch no longer matches {doc_id}: {old}")
                row[field] = row[field].replace(old, new, 1)
            record["changes"].append(dict(field=field, original=before, evaluation=row[field]))
        if doc_id == "p40/d1979":
            record["adjudication"] = dict(
                reason="The two currencies fill the same role in the transaction. "
                       "Keep the original Asset role for yen and use it for the negated US dollar alternative.",
                references=["https://doi.org/10.1109/ICSC.2011.57", "https://pmb.let.rug.nl/roles.php"],
            )
        graph = parse_sbn_line(row["sbn_repair"])
        require(not any(s.name == "CORRECTION" for s in graph.separators), f"RSR gold: {doc_id}")
        check = process_row(row["sbn_repair"])
        require(check["gold_status"] == "ok" and check["n_repairs"] == 0 and check["penman_clean"]
                and check["penman_repair"] == check["penman_clean"],
                f"Gold cannot be scored as a non-repair: {doc_id}: {check['strip_reason']}")
        ready.append(row)
        records[doc_id] = record

    require(len(ready) == len(held), "Unexpected diagnostic count")
    require(len({r["sentence"] for r in ready}) == len(ready), "Duplicate diagnostic inputs")
    protected = [ROOT / f"repair_data_gen/training_dataset/data/{split}.parquet"
                 for split in ("train", "validation")]
    for path in protected:
        data = pq.read_table(path, columns=["doc_id", "text"])
        require(not held.intersection(data["doc_id"].to_pylist()), f"Hold-out overlap with {path.name}")
        require(not {r["sentence"] for r in ready}.intersection(data["text"].to_pylist()),
                f"Diagnostic text overlap with {path.name}")

    OUT.mkdir(exist_ok=True)
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(ready[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(ready)
    annotation = OUT / "annotations.tsv"
    save(annotation, out.getvalue().encode())
    manifest = dict(
        version="hard_negative_diagnostic_v2",
        source="PMB 5.1.0 English train gold, held out before fine-tuning",
        held_out_documents=len(held),
        scoring_rows=len(ready),
        pending_semantic_review=0,
        scoring_rows_by_category=dict(Counter(r["category"] for r in ready)),
        validation=dict(non_repair_gold_exports=len(held), train_validation_overlap=0),
        input_sha256={str(path.relative_to(ROOT)): sha(path) for path in (HOLDOUT, PMB, *protected)},
        parser_sha256={name: sha(sbn_src_dir() / name) for name in
                       ("sbn_spec.py", "sbn_smatch.py", "graph_base.py", "penman_model.py")},
        annotations_sha256=sha(annotation), records=records,
    )
    save(OUT / "manifest.json", (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode())
    print(f"{len(held)} held out; {len(ready)} scoring rows; no pending semantic review.")


if __name__ == "__main__":
    main()
