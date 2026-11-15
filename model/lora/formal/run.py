"""Configure one full-data run, then hand it to the shared trainer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

LORA = Path(__file__).resolve().parents[1]
PROTOCOL = LORA / "protocols/fluent_groups_v1.published.json"
CONDITIONS = ("C1", "C2", "C3")
SETTINGS = {
    "2B": {"C1": (4e-4, 64), "C2": (3e-4, 32), "C3": (3e-4, 64)},
    "9B": {"C1": (4e-4, 64), "C2": (3e-4, 32), "C3": (3e-4, 64)},
    "27B": {"C1": (4e-4, 32), "C2": (3e-4, 32), "C3": (3e-4, 64)},
}


def training_command(cluster, size, condition, seed, output, dataset, cache):
    lr, rank = SETTINGS[size][condition]
    # C1 uses fluent inputs only. C2/C3 read the same full input split,
    # but data.py selects the clean or repair-aware SBN answer, respectively.
    # Keep the tuning schedule and alpha/rank ratio; only remove the train subset.
    return [
        sys.executable, str(LORA / "train.py"), "--protocol", str(PROTOCOL),
        "--purpose", "formal", "--cluster", cluster, "--model-size", size,
        "--condition", condition, "--seed", str(seed), "--max-steps", "0",
        "--learning-rate", str(lr), "--lora-rank", str(rank), "--lora-alpha", str(2 * rank),
        "--micro-batch-size", "1", "--eval-epoch-fraction", "0.3",
        "--generation-examples", "20", "--wandb-mode", "online",
        "--dataset-dir", str(dataset), "--model-cache", str(cache), "--output-dir", str(output),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cluster", choices=("habrok", "lst"))
    p.add_argument("size", choices=SETTINGS)
    p.add_argument("condition", choices=CONDITIONS)
    p.add_argument("seed", type=int)
    p.add_argument("job_id")
    args = p.parse_args()
    import torch
    # Check the allocation before loading a multi-GB base checkpoint.
    if torch.cuda.device_count() != 1:
        raise SystemExit("Expected exactly one visible CUDA GPU")
    gpu = torch.cuda.get_device_properties(0)
    expected, min_gb = ("H100", 79) if args.size == "27B" else ("A100", 39)
    if expected not in gpu.name or gpu.total_memory < min_gb * 10**9:
        raise SystemExit(f"Expected {expected} with at least {min_gb} GB; got {gpu.name}")
    # Keep separate attempts for traceability. A retry cannot overwrite an
    # earlier job's config, predictions or selected adapter.
    name = f"formal-{args.cluster}-{args.size}-{args.condition}-s{args.seed}-{args.job_id}"
    if not all(ch.isascii() and (ch.isalnum() or ch in "_-") for ch in name):
        raise SystemExit("Invalid scheduler job ID")
    output = Path(os.environ["FORMAL_RUN_ROOT"]) / name
    command = training_command(args.cluster, args.size, args.condition, args.seed, output,
                               os.environ["DATASET_DIR"], os.environ["MODEL_CACHE"])
    print(json.dumps({"gpu": gpu.name, "command": command}), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
