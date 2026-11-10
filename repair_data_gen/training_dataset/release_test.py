"""Stage or publish the accepted PMB RSR positives as default/test."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil

from datasets import load_dataset
from huggingface_hub import CommitOperationAdd, DatasetCard, HfApi, hf_hub_download
import pyarrow as pa
import pyarrow.parquet as pq

from .build import HERE, ROOT, SCHEMA, nullable, nullable_int, sha256
from .fluent_groups import provenance

REPO = "Shrikes/repair-sbn"
TEST = ROOT / "repair_data_gen/splits/test"
ALLOWLIST = {"README.md", "SCHEMA.md", "data/test.parquet", "test.acceptance.json"}
PROTECTED_DATA = ("data/train.parquet", "data/validation.parquet", "later_reference/test.parquet")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_bytes())


def stage_release(stage):
    acceptance_path = TEST / "manifests/acceptance_v1.json"
    acceptance = read(acceptance_path)
    require(acceptance["status"] == "accepted local PMB RSR positive test partition", "Test not accepted")
    for name, meta in acceptance["outputs"].items():
        require(sha256(TEST / "dataset" / name) == meta["sha256"], f"Accepted input changed: {name}")
    source = TEST / "dataset/test_rsr.paired.tsv"
    with source.open(encoding="utf-8", newline="") as handle:
        pairs = list(csv.DictReader(handle, delimiter="\t"))
    rows, variants = [], defaultdict(list)
    for r in pairs:
        editing = nullable(r["editing_expression"])
        row = provenance(dict(
            example_id=r["variant_id"], item_id=r["item_id"], doc_id=r["doc_id"],
            variant="rsr_editing_expression" if editing else "rsr_base", text=r["nl_repair"],
            sbn_clean=r["sbn_clean"], sbn_repair=r["sbn_repair"], repair_type=r["repair_type"],
            editing_expression=editing, ambiguity_type=nullable(r["ambiguous"]),
            repair_onset=nullable_int(r["repair_onset"]), reparandum_surface=nullable(r["reparandum_surface"]),
            reparandum_synset=nullable(r["reparandum_synset"]), reparandum_value=nullable(r["reparandum_value"]),
            repair_synset=nullable(r["repair_synset"]), site_index=nullable_int(r["site_index"]),
            strategy=nullable(r["strategy"]), wn_relation=nullable(r["wn_relation"]),
            label_source=r["label_source"], source="synthetic RSR from PMB 5.1.0 English standard test",
        ))
        require(set(row) == set(SCHEMA.names), "Test row does not match schema")
        variants[row["item_id"]].append(row)
        rows.append(row)
    require(len(rows) == acceptance["paired_rows"] == len({r["example_id"] for r in rows})
            == len({r["text"] for r in rows}), "Incorrect row count or duplicate input")
    require(len(variants) == acceptance["base_items"], "Incorrect base count")
    for item_id, pair in variants.items():
        require(len(pair) == 2 and {r["variant"] for r in pair} == {"rsr_base", "rsr_editing_expression"}
                and all(pair[0][k] == pair[1][k] for k in
                        ("sbn_clean", "sbn_repair", "doc_id", "family_id", "repair_type")),
                f"Broken pair: {item_id}")
    require(Counter(r["repair_type"] for r in rows if r["variant"] == "rsr_base")
            == acceptance["per_type"], "Type counts differ")
    require(Counter(r["editing_expression"] for r in rows if r["editing_expression"])
            == acceptance["marker_counts"], "Marker counts differ")
    baseline = read(HERE / "later_reference.publication.json")
    parent = baseline["revision"]
    old = baseline["verified_files_sha256"]
    for name in PROTECTED_DATA:
        require(sha256(HERE / name) == old[name], f"Protected local file changed: {name}")
    for name in PROTECTED_DATA[:2]:
        table = pq.read_table(HERE / name, columns=["doc_id", "family_id", "example_id", "text"])
        require(pq.read_schema(HERE / name).equals(SCHEMA, check_metadata=True), f"Schema changed: {name}")
        for k in table.column_names:
            require(not ({r[k] for r in rows} & set(table[k].to_pylist())), f"{k} overlap with {name}")
    expected_configs = [
        {"config_name": "default", "default": True, "data_files": [
            {"split": split, "path": f"data/{split}.parquet"} for split in ("train", "validation", "test")]},
        {"config_name": "later_reference", "data_files": [{"split": "test", "path": "later_reference/test.parquet"}]},
    ]
    require(DatasetCard.load(HERE / "README.md").data.to_dict()["configs"] == expected_configs,
            "Unexpected dataset card split mapping")
    stage.mkdir(parents=True, exist_ok=False)
    (stage / "data").mkdir()
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    parquet = stage / "data/test.parquet"
    pq.write_table(table, parquet, compression="zstd")
    require(pq.read_table(parquet).equals(table, check_metadata=True), "Parquet round-trip mismatch")
    local = load_dataset("parquet", data_files={"test": str(parquet)}, split="test",
                         cache_dir=str(stage.parent / "datasets-local"))
    require(local.to_list() == table.to_pylist(), "Local Datasets round-trip mismatch")
    for name in ("README.md", "SCHEMA.md"):
        shutil.copyfile(HERE / name, stage / name)
    save(stage / "test.acceptance.json", dict(
        status="passed", config="default", split="test", file="data/test.parquet",
        base_items=len(variants), rows=len(rows), source_documents=acceptance["source_documents"],
        schema_fields=len(SCHEMA), per_type=acceptance["per_type"], marker_counts=acceptance["marker_counts"],
        marked_naturalness=acceptance["marked_naturalness"]["review"],
        scoring=acceptance["primary_aggregation"], paired_tsv_sha256=sha256(source),
        parquet_sha256=sha256(parquet),
    ))
    save(stage / "stage.manifest.json", dict(
        repo_id=REPO, parent=parent, prior_files=old,
        files={name: sha256(stage / name) for name in sorted(ALLOWLIST)},
    ))
    print(f"Staged {len(rows)} rows in {len(variants)} pairs at {stage}")


def publish(stage):
    manifest = read(stage / "stage.manifest.json")
    require(manifest["repo_id"] == REPO and set(manifest["files"]) == ALLOWLIST,
            "Unexpected repository or upload allowlist")
    for name, digest in manifest["files"].items():
        require(sha256(stage / name) == digest, f"Stage changed: {name}")
    for name in ("README.md", "SCHEMA.md"):
        require(sha256(HERE / name) == manifest["files"][name], f"Local card changed: {name}")
    api = HfApi()
    parent = manifest["parent"]
    cache = stage.parent / "hub"
    def remote(name, revision):
        return Path(hf_hub_download(REPO, name, repo_type="dataset", revision=revision, cache_dir=cache))
    pending = stage / "publication.pending.json"
    if pending.exists():
        record = read(pending)
        require(record["files"] == manifest["files"], "Pending publication does not match stage")
        revision = record["revision"]
    else:
        info = api.dataset_info(REPO)
        require(info.sha == parent and set(manifest["prior_files"]) == {f.rfilename for f in info.siblings},
                "Hub parent or file inventory changed")
        require(not (set(manifest["prior_files"]) & (ALLOWLIST - {"README.md", "SCHEMA.md"})),
                "Test outputs already exist remotely")
        for name, digest in manifest["prior_files"].items():
            require(sha256(remote(name, parent)) == digest, f"Baseline file changed: {name}")
        commit = api.create_commit(
            REPO, repo_type="dataset", parent_commit=parent,
            operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=stage / name)
                        for name in sorted(ALLOWLIST)],
            commit_message="Add paired PMB RSR default test split",
        )
        revision = commit.oid
        save(pending, dict(revision=revision, files=manifest["files"]))
        print(f"Commit created: {revision}", flush=True)
    expected = manifest["prior_files"] | manifest["files"]
    info = api.dataset_info(REPO, revision=revision)
    require({f.rfilename for f in info.siblings} == set(expected), "Unexpected remote file inventory")
    for name, digest in expected.items():
        require(sha256(remote(name, revision)) == digest, f"Remote file mismatch: {name}")
    table = pq.read_table(stage / "data/test.parquet")
    loaded = load_dataset(REPO, "default", split="test", revision=revision,
                          cache_dir=str(stage.parent / "datasets-remote"))
    require(loaded.to_list() == table.to_pylist(), "Remote Datasets readback mismatch")
    published = dict(
        status="published_and_verified", repo_id=REPO, revision=revision, parent_revision=parent,
        config="default", split="test", rows=len(loaded), pairs=len(loaded)//2,
        commit_url=f"https://huggingface.co/datasets/{REPO}/commit/{revision}",
        files_sha256=manifest["files"], verified_files_sha256=expected,
        existing_non_document_files_unchanged=True, exact_datasets_readback=True,
    )
    for name in ("data/test.parquet", "test.acceptance.json"):
        target = HERE / name
        require(not target.exists() or sha256(target) == manifest["files"][name],
                f"Different local file already exists: {name}")
        if not target.exists():
            shutil.copyfile(stage / name, target)
    save(HERE / "test.publication.json", published)
    docs = HERE / "documentation.publication.json"
    old_docs = read(docs)
    save(docs, dict(
        status="published_and_verified", repo_id=REPO, revision=revision, parent_revision=parent,
        data_release_revision=old_docs["data_release_revision"],
        documentation_sha256={name: expected[name] for name in ("README.md", "SCHEMA.md")},
        commit_url=published["commit_url"], verified_files=len(expected),
        data_files_unchanged=False, train_validation_unchanged=True,
        added_split="default/test", test_publication="test.publication.json",
    ))
    print(json.dumps({k:published[k] for k in ("status", "commit_url", "rows", "pairs")}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("stage", "publish"))
    p.add_argument("--stage-dir", type=Path, required=True)
    args = p.parse_args()
    (stage_release if args.action == "stage" else publish)(args.stage_dir.resolve())


if __name__ == "__main__":
    main()
