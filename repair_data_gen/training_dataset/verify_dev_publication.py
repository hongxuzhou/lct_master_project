"""Read back a pinned HF release and record exact local/remote agreement.

    python -m repair_data_gen.training_dataset.verify_dev_publication --revision COMMIT
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

from .build import HERE, sha256


REPO = "Shrikes/repair-sbn"
RELEASE_FILES = ("README.md", "data/validation.parquet", "validation.manifest.json",
                 "validation.acceptance.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    api = HfApi()
    info = api.dataset_info(REPO, revision=args.revision)
    if info.sha != args.revision:
        raise ValueError("pass an immutable full commit hash")
    configs = info.card_data.to_dict()["configs"]
    expected = [{"config_name": "default", "default": True, "data_files": [
        {"split": "train", "path": "data/train.parquet"},
        {"split": "validation", "path": "data/validation.parquet"}]}]
    if configs != expected:
        raise ValueError("remote split mapping differs from the release")
    checked_files, counts, variants = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="repairsbn-hf-publication-readback-") as cache:
        for name in (*RELEASE_FILES, "data/train.parquet", "manifest.json"):
            downloaded = Path(hf_hub_download(REPO, name, repo_type="dataset",
                                             revision=args.revision, cache_dir=cache))
            if downloaded.read_bytes() != (HERE / name).read_bytes():
                raise ValueError(f"remote file differs from local file: {name}")
            checked_files[name] = sha256(downloaded)
        ds = load_dataset(REPO, name="default", revision=args.revision,
                          cache_dir=str(Path(cache) / "datasets"))
        if set(ds) != {"train", "validation"}:
            raise ValueError(f"unexpected remote splits: {list(ds)}")
        if ds["train"].features != ds["validation"].features:
            raise ValueError("remote split schemas differ")
        for split in ("train", "validation"):
            local = pq.read_table(HERE / f"data/{split}.parquet").to_pylist()
            if ds[split].to_list() != local:
                raise ValueError(f"remote datasets readback changed {split}")
            counts[split] = len(ds[split])
            variants[split] = dict(Counter(ds[split]["variant"]))
    receipt = {
        "status": "published and verified", "repo_id": REPO, "revision": info.sha,
        "commit_url": f"https://huggingface.co/datasets/{REPO}/commit/{info.sha}",
        "private": info.private, "rows": counts, "variants": variants,
        "files_sha256": checked_files, "uploaded_files": list(RELEASE_FILES),
        "readback": "exact file bytes and all rows match local; schemas match across splits",
        "train_unchanged": True, "test_published": False,
        "verifier_sha256": sha256(Path(__file__)),
    }
    path = HERE / "validation.publication.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
