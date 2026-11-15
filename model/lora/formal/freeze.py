"""Verify the selected checkpoints on one cluster and emit their file digests."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def freeze(row):
    path = Path(row["checkpoint_path"])
    root = path.parent
    selection = root / "selection.json"
    selected = read_json(selection)
    config = read_json(root / "config.json")
    summary = read_json(root / "summary.json")
    state = read_json(path / "state.json")
    spec = config["protocol"]["models"][row["model_size"]]
    loss = float(row["best_dev_loss"])
    step = int(row["best_step"])
    if not (
        selection == Path(row["selection_json"])
        and root.name == config["run_id"] == row["run_id"]
        and all(str(config[k]) == row[k] for k in ("model_size", "condition", "seed", "cluster"))
        and (spec["id"], spec["revision"]) == (row["base_model"], row["base_revision"])
        and summary["status"] == "formal_passed"
        and summary["steps"] == int(row["steps"])
        and summary["best_step"] == state["step"] == step
        and selected["best_checkpoint"] == summary["best_checkpoint"] == path.name == f"checkpoint-{step}"
        and selected["best_eval_loss"] == summary["best_eval_loss"] == state["eval_loss"] == loss
        and summary["reload_eval_loss"] == float(row["reload_dev_loss"])
    ):
        raise ValueError(f"Selection differs from the audited W&B run: {root}")
    required = {"adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json", "tokenizer.json"}
    optional = {"chat_template.jinja", "special_tokens_map.json", "added_tokens.json",
                "vocab.json", "merges.txt", "tokenizer.model", "generation_config.json"}
    names = {p.name for p in path.iterdir()} - {"training_state.pt", "state.json", "README.md"}
    if required - names or names - required - optional:
        raise ValueError(f"Unexpected checkpoint inventory at {path}: {sorted(names)}")
    adapter = read_json(path / "adapter_config.json")
    if (adapter["base_model_name_or_path"], adapter["r"], adapter["lora_alpha"]) != (
        spec["id"], config["lora_rank"], config["lora_alpha"]
    ):
        raise ValueError(f"Adapter config mismatch: {path}")
    files = {}
    for name in sorted(names):
        p = path / name
        if not p.is_file() or p.is_symlink() or not p.stat().st_size:
            raise ValueError(f"Invalid checkpoint file: {p}")
        files[name] = digest(p)
    return dict(row, files=files, selection_sha256=digest(selection),
                config_sha256=digest(root / "config.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cluster", choices=("habrok", "lst"))
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    with args.manifest.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream, delimiter="\t") if r["cluster"] == args.cluster]
    sizes = {"2B", "9B"} if args.cluster == "habrok" else {"27B"}
    expected = {(size, cond, str(seed)) for size in sizes for cond in ("C1", "C2", "C3")
                for seed in (24, 42, 4224)}
    if len(rows) != len(expected) or {(r["model_size"], r["condition"], r["seed"]) for r in rows} != expected:
        raise ValueError(f"Incomplete or duplicate run matrix for {args.cluster}")
    print(json.dumps([freeze(row) for row in rows], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
