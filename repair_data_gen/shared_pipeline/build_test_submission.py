"""Package the first PMB test synthesis batch without accessing the cluster."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

from .select_candidates import PROJECT_ROOT, GENERATION, prompt_sha


NAME = "test_synthesis_v1"
REMOTE = f"/nethome/honzhou/thesis_zone/{NAME}"
WORKSPACE = PROJECT_ROOT / "repair_data_gen/splits/test"


def build(output: Path) -> dict:
    files: dict[str, bytes] = {}
    sources = {}

    def add(source: Path, destination: str):
        data = source.read_bytes()
        files[destination] = data
        sources[str(source.relative_to(PROJECT_ROOT))] = hashlib.sha256(data).hexdigest()

    candidate = json.loads((WORKSPACE / "manifests/candidate_samples.json").read_text())
    for entry in candidate["outputs"]:
        path = PROJECT_ROOT / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"candidate sample changed: {path}")
        add(path, "splits/test/samples/" + path.name)
    for version, digest in candidate["prompt_sha256"].items():
        prompt = GENERATION / "prompts" / version
        if prompt_sha(prompt) != digest:
            raise ValueError(f"prompt changed: {version}")
        for name in ("sbn_manual.md", "repair_notation.md", "task.md", "item_template.md"):
            add(prompt / name, f"rsr_generation/prompts/{version}/{name}")
    for name in ("run_llm.py", "wn_lookup.py", "hpc/run.sh"):
        add(GENERATION / name, "rsr_generation/" + name)
    harness = files["rsr_generation/hpc/run.sh"].decode()
    original = "CODE=$NETHOME/thesis_zone/rsr_generation"
    if harness.count(original) != 1:
        raise ValueError("legacy harness CODE assignment changed")
    files["rsr_generation/hpc/run.sh"] = harness.replace(
        original, f"CODE=$NETHOME/thesis_zone/{NAME}/rsr_generation").encode()
    for name in ("config.json", "review_policy.json", "marker_policy.json", "README.md",
                 "manifests/candidate_samples.json"):
        add(WORKSPACE / name, "splits/test/" + name)
    if hashlib.sha256(files["splits/test/config.json"]).hexdigest() != candidate["config_sha256"]:
        raise ValueError("generation config changed")
    for path in sorted((WORKSPACE / "hpc").glob("*")):
        if path.is_file():
            add(path, "splits/test/hpc/" + path.name)
    for directory in ("results", "results/raw"):
        files[f"splits/test/{directory}/.keep"] = b""
    receipt = dict(remote_root=REMOTE, requests=candidate["candidate_counts"],
                   source_sha256=sources,
                   packaged_sha256={p: hashlib.sha256(b).hexdigest() for p, b in sorted(files.items())},
                   harness_change="CODE points to isolated test runtime; all other bytes unchanged")
    files["submission_manifest.json"] = (json.dumps(receipt, indent=2) + "\n").encode()
    files["files.sha256"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in sorted(files.items())).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to replace an already prepared archive.
    with output.open("xb") as target, tarfile.open(fileobj=target, mode="w:gz") as archive:
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(NAME + "/" + name)
            info.size = len(data)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(data))
    return dict(archive=str(output.resolve()), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                files=len(files), remote_root=REMOTE, requests=candidate["candidate_counts"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.out), indent=2))


if __name__ == "__main__":
    main()
