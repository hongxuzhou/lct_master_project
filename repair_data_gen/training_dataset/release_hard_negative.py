"""Stage or publish the validated hard-negative diagnostic as its own test config."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import shutil

from datasets import load_dataset
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq

from .build import HERE, ROOT, sha256
from .release_test import REPO, read, require, save

DIAGNOSTIC = ROOT / "repair_data_gen/hard_negatives/diagnostic_v2"
PARQUET = "hard_negative/test.parquet"
MANIFEST = "hard_negative/manifest.json"
DOCS = {"README.md", "SCHEMA.md"}
ALLOWLIST = DOCS | {PARQUET, MANIFEST}
FIELDS = ("example_id", "doc_id", "text", "sbn_clean", "sbn_repair", "category")


def stage_release(stage: Path):
    diagnostic = read(DIAGNOSTIC / "manifest.json")
    source = DIAGNOSTIC / "annotations.tsv"
    require(sha256(source) == diagnostic["annotations_sha256"], "Accepted annotations changed")
    require(diagnostic["scoring_rows"] == diagnostic["held_out_documents"]
            == diagnostic["validation"]["non_repair_gold_exports"] == 66
            and diagnostic["pending_semantic_review"] == 0, "Diagnostic not complete")
    for name, digest in diagnostic["input_sha256"].items():
        require(sha256(ROOT / name) == digest, f"Diagnostic input changed: {name}")
    for name, digest in diagnostic["parser_sha256"].items():
        require(sha256(ROOT / "data/pmb-5.1.0/src/sbn" / name) == digest, f"Reader changed: {name}")
    with source.open(encoding="utf-8", newline="") as handle:
        rows = [dict(example_id=r["id"], doc_id=r["doc_id"], text=r["sentence"],
                     sbn_clean=r["sbn_repair"], sbn_repair=r["sbn_repair"], category=r["category"])
                for r in csv.DictReader(handle, delimiter="\t")]
    require(len(rows) == 66 and set(diagnostic["records"]) == {r["doc_id"] for r in rows},
            "Wrong diagnostic inventory")
    require(all(diagnostic["records"][r["doc_id"]]["status"] == "ready"
                and diagnostic["records"][r["doc_id"]]["category"] == r["category"] for r in rows),
            "Source records do not match")
    for key in ("example_id", "doc_id", "text"):
        require(len({r[key] for r in rows}) == len(rows), f"Duplicate {key}")
    require(Counter(r["category"] for r in rows) == diagnostic["scoring_rows_by_category"],
            "Unexpected category totals")

    baseline = read(HERE / "test.publication.json")
    prior = baseline["verified_files_sha256"]
    for name in ("data/train.parquet", "data/validation.parquet", "data/test.parquet",
                 "later_reference/test.parquet"):
        require(sha256(HERE / name) == prior[name], f"Protected local file changed: {name}")
        keys = ("doc_id", "text") if name.startswith("data/") else ("text",)
        data = pq.read_table(HERE / name, columns=list(keys))
        for key in keys:
            require(not set(data[key].to_pylist()) & {r[key] for r in rows}, f"{key} overlap: {name}")
    configs = DatasetCard.load(HERE / "README.md").data.to_dict()["configs"]
    require(configs == [
        {"config_name": "default", "default": True, "data_files": [
            {"split": s, "path": f"data/{s}.parquet"} for s in ("train", "validation", "test")]},
        {"config_name": "later_reference", "data_files": [{"split": "test", "path": "later_reference/test.parquet"}]},
        {"config_name": "hard_negative", "data_files": [{"split": "test", "path": PARQUET}]},
    ], "Unexpected card configuration")

    stage.mkdir(parents=True, exist_ok=False)
    (stage / "hard_negative").mkdir()
    table = pa.Table.from_pylist(rows, schema=pa.schema([(name, pa.string()) for name in FIELDS]))
    pq.write_table(table, stage / PARQUET, compression="zstd")
    require(pq.read_table(stage / PARQUET).equals(table, check_metadata=True), "Parquet round-trip mismatch")
    local = load_dataset("parquet", data_files={"test": str(stage / PARQUET)}, split="test",
                         cache_dir=str(stage.parent / "datasets-local"))
    require(local.to_list() == rows, "Local Datasets round-trip mismatch")
    for name in DOCS:
        shutil.copyfile(HERE / name, stage / name)
    save(stage / MANIFEST, dict(
        version=diagnostic["version"], config="hard_negative", split="test",
        rows=len(rows), source=diagnostic["source"], categories=diagnostic["scoring_rows_by_category"],
        changes={i:r for i,r in diagnostic["records"].items() if r["changes"]},
        validation=diagnostic["validation"], parser_sha256=diagnostic["parser_sha256"],
        annotations_sha256=diagnostic["annotations_sha256"], parquet_sha256=sha256(stage / PARQUET),
    ))
    save(stage / "stage.manifest.json", dict(
        repo_id=REPO, parent=baseline["revision"], prior_files=prior,
        files={name:sha256(stage / name) for name in sorted(ALLOWLIST)},
    ))
    print(f"Staged {len(rows)} hard negatives in {stage}")


def publish(stage: Path):
    manifest = read(stage / "stage.manifest.json")
    hashes = manifest["files"]
    require(manifest["repo_id"] == REPO and set(hashes) == ALLOWLIST, "Invalid publication allowlist")
    for name, digest in hashes.items():
        require(sha256(stage / name) == digest, f"Staged file changed: {name}")
    for name in DOCS:
        require(sha256(HERE / name) == hashes[name], f"Local documentation changed: {name}")
    for name in ALLOWLIST - DOCS:
        target = HERE / name
        require(not target.exists() or sha256(target) == hashes[name], f"Local export differs: {name}")

    api = HfApi()
    parent, prior = manifest["parent"], manifest["prior_files"]
    cache = stage.parent / "hub"

    def verify(revision, files):
        info = api.dataset_info(REPO, revision=revision)
        require({f.rfilename for f in info.siblings} == set(files), "Remote inventory changed")
        def one(item):
            name, expected = item
            actual = hf_hub_download(REPO, name, repo_type="dataset", revision=revision, cache_dir=cache)
            require(sha256(Path(actual)) == expected, f"Remote hash mismatch: {name}")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(one, files.items()))

    pending_path = stage / "publication.pending.json"
    if pending_path.exists():
        pending = read(pending_path)
        require(pending["files"] == hashes, "Pending revision does not match stage")
        revision = pending["revision"]
    else:
        require(api.dataset_info(REPO).sha == parent, "Remote head changed")
        require(not set(prior) & (ALLOWLIST - DOCS), "Hard-negative output already exists")
        verify(parent, prior)
        commit = api.create_commit(
            REPO, repo_type="dataset", parent_commit=parent,
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=stage / name)
                        for name in sorted(ALLOWLIST)],
            commit_message="Add held-out PMB hard-negative test configuration",
        )
        revision = commit.oid
        save(pending_path, dict(revision=revision, files=hashes))
        print(f"Commit created: {revision}", flush=True)
    verify(revision, prior | hashes)
    data = load_dataset(REPO, "hard_negative", split="test", revision=revision,
                        cache_dir=str(stage.parent / "datasets-remote"))
    require(data.to_list() == pq.read_table(stage / PARQUET).to_pylist(), "HF Datasets readback differs")

    (HERE / "hard_negative").mkdir(exist_ok=True)
    for name in (PARQUET, MANIFEST):
        shutil.copyfile(stage / name, HERE / name)
    url = f"https://huggingface.co/datasets/{REPO}/commit/{revision}"
    save(HERE / "hard_negative.publication.json", dict(
        status="published_and_verified", repo_id=REPO, revision=revision, parent_revision=parent,
        commit_url=url, config="hard_negative", split="test", rows=len(data),
        files_sha256=hashes, verified_files_sha256=prior | hashes,
        existing_non_document_files_unchanged=True, exact_datasets_readback=True,
    ))
    docs = HERE / "documentation.publication.json"
    previous = read(docs)
    save(docs, dict(status="published_and_verified", repo_id=REPO, revision=revision,
                   parent_revision=parent, data_release_revision=previous["data_release_revision"],
                   documentation_sha256={n:hashes[n] for n in sorted(DOCS)}, commit_url=url,
                   verified_files=len(prior | hashes), data_files_unchanged=False,
                   added_configuration="hard_negative",
                   hard_negative_publication="hard_negative.publication.json"))
    print(f"Verified {len(data)} rows via HF Datasets: {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "publish"))
    parser.add_argument("--stage-dir", type=Path, required=True)
    args = parser.parse_args()
    (stage_release if args.action == "stage" else publish)(args.stage_dir.resolve())
