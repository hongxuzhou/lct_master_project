"""Transfer frozen Hábrók adapters to LST through a private model repo."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(rows):
    expected = {(s, c, str(seed)) for s in ("2B", "9B")
                for c in ("C1", "C2", "C3") for seed in (24, 42, 4224)}
    if len(rows) != 18 or {
        (r["model_size"], r["condition"], str(r["seed"])) for r in rows
    } != expected or len({r["run_id"] for r in rows}) != 18:
        raise ValueError("Expected the complete 18-run Hábrók matrix")
    for r in rows:
        run = r["run_id"]
        checkpoint = f"checkpoint-{int(r['best_step'])}"
        if r["cluster"] != "habrok" or not re.fullmatch(r"formal-habrok-[\w-]+", run):
            raise ValueError(f"Invalid run: {run}")
        names = r["files"]
        allowed = {"adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
                   "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json",
                   "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
                   "generation_config.json"}
        if set(names) - allowed or not {
            "adapter_config.json", "adapter_model.safetensors",
            "tokenizer.json", "tokenizer_config.json"
        } <= names.keys():
            raise ValueError(f"Invalid file inventory: {run}")
        yield r, Path(run) / checkpoint


def verify(base, rows, source=False):
    for r, relative in inventory(rows):
        path = Path(r["checkpoint_path"]) if source else base / relative
        files = [(path.parent / "config.json", r["config_sha256"])]
        files.extend((path / name, sha) for name, sha in r["files"].items())
        for file, sha in files:
            if digest(file) != sha:
                raise ValueError(f"Hash mismatch: {file}")
        yield r, relative, files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("upload", "download"))
    p.add_argument("root", type=Path, help="Formal test directory on this cluster")
    p.add_argument("--repo", default="Shrikes/repair-sbn-formal-adapters")
    p.add_argument("--revision", help="Commit SHA printed by upload; required for download")
    p.add_argument("--stage-dir", type=Path, help="Existing scratch directory for temporary upload files")
    p.add_argument("--download-dir", type=Path, help="Shared directory for downloaded weights")
    a = p.parse_args()
    if a.action == "download" and not re.fullmatch(r"[a-f0-9]{40}", a.revision or ""):
        p.error("Download requires the exact upload commit SHA")
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    root = a.root.resolve(strict=True)
    name = "frozen-habrok.json"
    if a.action == "upload":
        rows = json.loads((root / name).read_text())
        stage_dir = a.stage_dir.resolve(strict=True) if a.stage_dir else root
        with tempfile.TemporaryDirectory(prefix="formal-hf-", dir=stage_dir) as tmp:
            stage = Path(tmp)
            for r, relative, files in verify(stage, rows, source=True):
                (stage / relative).mkdir(parents=True)
                for file, _ in files:
                    target = stage / relative / file.name
                    if file.name == "config.json":
                        target = target.parent.parent / file.name
                    shutil.copyfile(file, target)
            list(verify(stage, rows))
            # Keep the Hub manifest portable; preserve the hashes of all files.
            for r, relative in inventory(rows):
                r["checkpoint_path"] = str(relative)
                r.pop("selection_json", None)
            (stage / name).write_text(json.dumps(rows, indent=2) + "\n")
            api.create_repo(a.repo, repo_type="model", private=True, exist_ok=True)
            if not api.repo_info(a.repo, repo_type="model").private:
                p.error("Target model repository is not private")
            commit = api.upload_folder(repo_id=a.repo, repo_type="model", folder_path=stage,
                                       commit_message="Transfer 18 frozen 2B and 9B adapters")
            print(commit.commit_url)
            print(f"Revision: {commit.oid}")
        return

    if not api.repo_info(a.repo, repo_type="model").private:
        p.error("Source model repository is not private")
    destination = a.download_dir.resolve(strict=True) if a.download_dir else root
    target = destination / ("hub-adapters-" + a.revision[:12])
    snapshot_download(a.repo, repo_type="model", revision=a.revision, local_dir=target,
                      allow_patterns=[name, "formal-habrok-*/*", "formal-habrok-*/*/*"])
    rows = json.loads((target / name).read_text())
    for r, relative, _ in verify(target, rows):
        r["checkpoint_path"] = str(target / relative)
    manifest = root / "frozen-habrok-on-lst.json"
    with manifest.open("x") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")
    print(f"Verified 18 adapters: {manifest}")


if __name__ == "__main__":
    main()
