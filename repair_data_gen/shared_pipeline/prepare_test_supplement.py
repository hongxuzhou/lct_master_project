"""Prepare the isolated PMB test supplement and its separate review queue."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile

from .select_candidates import (
    BASE_FIELDS, GENERATION, PROJECT_ROOT, common_row, prompt_sha, selection,
)

TEST = PROJECT_ROOT / "repair_data_gen/splits/test"
NAME = "test_supplement_v1"
WORK = TEST / "supplement_v1"
SAMPLE = WORK / f"{NAME}.tsv"
MANIFEST = WORK / "candidate_samples.json"
REMOTE = f"/nethome/honzhou/thesis_zone/{NAME}"
QUOTAS = {"constant_name": 45, "content_word": 60}
SEED = 20261109


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def tsv_bytes(rows: list[dict], fields) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode()


def preserve(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"{path} differs; prepare a new version")
    else:
        path.write_bytes(data)


def build_samples() -> dict:
    first_path = TEST / "manifests/candidate_samples.json"
    first = json.loads(first_path.read_bytes())
    config = TEST / "config.json"
    pmb = PROJECT_ROOT / first["pmb_source"]
    prompt = GENERATION / "prompts/base"
    if sha(config.read_bytes()) != first["config_sha256"]:
        raise ValueError("First-round config changed")
    if sha(pmb.read_bytes()) != first["pmb_sha256"]:
        raise ValueError("PMB test source changed")
    if prompt_sha(prompt) != first["prompt_sha256"]["base"]:
        raise ValueError("Base prompt changed")

    initial = []
    for entry in first["outputs"]:
        path = PROJECT_ROOT / entry["path"]
        if sha(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"First-round sample changed: {path}")
        initial.extend(read(path))
    if len(initial) != first["candidate_counts"]["total_requests"]:
        raise ValueError("First-round request count changed")
    used_ids = {row["doc_id"] for row in initial}
    used_text = {row["nl_clean"] for row in initial}
    docs = selection.read_split(pmb)
    if selection.read_split.skipped or len(docs) != first["pmb_documents"]:
        raise ValueError("Incomplete PMB test read")

    isolation = {}
    for split, entry in first["source_isolation"].items():
        source = PROJECT_ROOT / entry["source"]
        if sha(source.read_bytes()) != entry["sha256"]:
            raise ValueError(f"{split} PMB source changed")
        other = selection.read_split(source)
        if selection.read_split.skipped:
            raise ValueError(f"Incomplete PMB {split} read")
        isolation[split] = ({d.doc_id for d in other}, {d.sentence for d in other})

    eligible = sorted((d for d in docs
                       if d.doc_id not in used_ids | selection.EXAMPLE_DOCS
                       and d.sentence not in used_text
                       and selection.gold_is_sound(d)), key=lambda d: d.doc_id)
    initial_name_ids = {r["doc_id"] for r in initial if r["repair_type"] == "constant_name"}
    name_counts = Counter(value for d in docs if d.doc_id in initial_name_ids
                          for value in {v for _, v in selection.site_name(d)})
    cap = json.loads(config.read_bytes())["candidate_plan"]["constant_value_caps"]["constant_name"]
    rng = random.Random(SEED)
    rows = []
    pool_sizes = {}
    for kind, fn in (("constant_name", selection.site_name),
                     ("content_word", selection.site_content_word)):
        pool = [(d, sorted(set(fn(d)))) for d in eligible if fn(d)]
        pool_sizes[kind] = len(pool)
        rng.shuffle(pool)
        count = 0
        for doc, sites in pool:
            if doc.doc_id in used_ids or doc.sentence in used_text:
                continue
            if any(doc.doc_id in ids or doc.sentence in texts for ids, texts in isolation.values()):
                raise ValueError(f"Source overlaps train or dev: {doc.doc_id}")
            if kind == "constant_name":
                values = {v for _, v in sites}
                # Same eligibility rule as the first round, counting earlier name requests too.
                if cap and all(name_counts[v] >= cap for v in values):
                    continue
                name_counts.update(values)
            rows.append(common_row(doc, kind, sites, f"{NAME}::{kind}::{doc.doc_id}"))
            used_ids.add(doc.doc_id)
            used_text.add(doc.sentence)
            count += 1
            if count == QUOTAS[kind]:
                break
        if count != QUOTAS[kind]:
            raise ValueError(f"Only {count} eligible {kind} requests")

    # Mix the two types so an interrupted job does not return only one type.
    rng.shuffle(rows)
    sample_data = tsv_bytes(rows, BASE_FIELDS)
    manifest = dict(
        status="follow-up generation requests; not accepted test items",
        seed=SEED, first_round_manifest_sha256=sha(first_path.read_bytes()),
        excluded_first_round_documents=len({row["doc_id"] for row in initial}),
        eligible_unused_documents=pool_sizes,
        config_sha256=first["config_sha256"], pmb_sha256=first["pmb_sha256"],
        prompt_sha256={"base": first["prompt_sha256"]["base"]},
        candidate_counts=QUOTAS, source_isolation="unique sources; no first-round/train/dev ID or exact-text overlap",
        builder=rel(Path(__file__)), builder_sha256=sha(Path(__file__).read_bytes()),
        outputs=[dict(path=rel(SAMPLE), rows=len(rows), sha256=sha(sample_data))],
    )
    preserve(SAMPLE, sample_data)
    preserve(MANIFEST, json_bytes(manifest))
    return manifest


def package(output: Path, manifest: dict) -> None:
    files = {
        f"splits/test/samples/{SAMPLE.name}": SAMPLE,
        "splits/test/config.json": TEST / "config.json",
        "splits/test/manifests/followup.json": MANIFEST,
        "splits/test/hpc/preflight.sh": WORK / "hpc/preflight.sh",
        f"splits/test/hpc/run_{NAME}.sub": WORK / f"hpc/run_{NAME}.sub",
    }
    for name in ("run_llm.py", "wn_lookup.py", "hpc/run.sh",
                 *(f"prompts/base/{f}" for f in (
                     "sbn_manual.md", "repair_notation.md", "task.md", "item_template.md"))):
        files[f"rsr_generation/{name}"] = GENERATION / name
    data = {name: path.read_bytes() for name, path in files.items()}
    wrapper = "rsr_generation/hpc/run.sh"
    assignment = b"CODE=$NETHOME/thesis_zone/rsr_generation"
    if data[wrapper].count(assignment) != 1:
        raise ValueError("Unexpected legacy wrapper CODE assignment")
    data[wrapper] = data[wrapper].replace(
        assignment, f"CODE=$NETHOME/thesis_zone/{NAME}/rsr_generation".encode())
    data["splits/test/results/raw/.keep"] = b""
    data["submission_manifest.json"] = json_bytes(dict(
        remote_root=REMOTE, requests=manifest["candidate_counts"],
        source_sha256={rel(path): sha(path.read_bytes()) for path in files.values()},
        packaged_sha256={name: sha(contents) for name, contents in sorted(data.items())},
    ))
    data["files.sha256"] = "".join(f"{sha(contents)}  {name}\n"
                                   for name, contents in sorted(data.items())).encode()
    with output.open("xb") as handle, tarfile.open(fileobj=handle, mode="w:gz") as archive:
        for name, contents in sorted(data.items()):
            info = tarfile.TarInfo(f"{NAME}/{name}")
            info.size = len(contents)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(contents))


def prepare_review(policy_dir: Path) -> None:
    from .analyse_split import write
    from audit import ITEM_FIELDS

    summary = json.loads((policy_dir / "summary.json").read_bytes())
    if summary["returned"] + summary["transport_missing"] != sum(QUOTAS.values()):
        raise ValueError("Machine review does not account for the supplement requests")
    sample = {r["request_id"]: r for r in read(SAMPLE)}
    dispositions = read(policy_dir / "dispositions.tsv")
    if (len({r["request_id"] for r in dispositions}) != len(dispositions)
            or len(dispositions) != summary["returned"]
            or any(r["request_id"] not in sample for r in dispositions)):
        raise ValueError("Machine review dispositions do not match the supplement")
    source = policy_dir / "review_candidates.tsv"
    pool = sorted(read(source), key=lambda r: r["request_id"])
    expected = {r["request_id"] for r in dispositions if r["policy_eligible"] == "True"}
    if (len(pool) != len(expected) or {r["request_id"] for r in pool} != expected
            or len(pool) != summary["policy_eligible"]):
        raise ValueError("Machine-eligible requests do not match dispositions")
    if not pool:
        raise ValueError("No eligible candidates to review")
    for row in pool:
        ref = sample[row["request_id"]]
        if any(ref[k] != row[k] for k in ("doc_id", "repair_type", "nl_clean", "sbn_clean")):
            raise ValueError(f"Unexpected source: {row['request_id']}")
    random.Random(SEED).shuffle(pool)
    for n, row in enumerate(pool, 1):
        row.update(audit_id=f"testsup{n:04d}", source=NAME)
    items = [{key: row.get(key, "") for key in ITEM_FIELDS} for row in pool]
    out = TEST / "audit" / NAME
    out.mkdir()
    write(out / "items.tsv", items, ITEM_FIELDS)
    write(out / "review_pool.tsv", pool)
    (out / "manifest.json").write_bytes(json_bytes(dict(
        status="supplemental blind review; not final test", rows=len(items),
        per_type=dict(Counter(r["repair_type"] for r in pool)),
        sample_sha256=sha(SAMPLE.read_bytes()), source_sha256=sha(source.read_bytes()),
        items_sha256=sha((out / "items.tsv").read_bytes()),
        review_pool_sha256=sha((out / "review_pool.tsv").read_bytes()),
    )))
    print(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path, help="new local tar.gz for manual LST upload")
    group.add_argument("--review-from", type=Path, help="validated machine-policy output directory")
    args = parser.parse_args()
    if args.review_from:
        prepare_review(args.review_from)
    else:
        manifest = build_samples()
        package(args.out, manifest)
        print(json.dumps(dict(archive=str(args.out.resolve()), requests=QUOTAS,
                              sha256=sha(args.out.read_bytes())), indent=2))


if __name__ == "__main__":
    main()
