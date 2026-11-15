"""Preview or submit formal LoRA jobs; one scheduler job per size/condition/seed."""
import argparse
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import shlex
import subprocess

from run import CONDITIONS, LORA, PROTOCOL, SETTINGS

HERE = Path(__file__).resolve().parent


def check_inputs(cluster, sizes):
    # Ask the same environment file that the worker will use. This only
    # reads paths and small files; it does not import torch or load a model.
    paths = subprocess.check_output([
        "bash", "-c", 'set -eu; source "$1"; printf "%s\\0" "$PYTHON" "$DATASET_DIR" "$MODEL_CACHE"',
        "formal-check", str(LORA / "tuning" / cluster / "environment_v2.sh"),
    ]).decode().split("\0")[:3]
    interpreter, dataset, cache = map(Path, paths)
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError(f"Missing executable interpreter: {interpreter}")
    protocol = json.loads(PROTOCOL.read_text())
    for spec in protocol["dataset"]["files"].values():
        path = dataset / spec["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {path}")
    for size in sizes:
        spec = protocol["models"][size]
        snapshot = cache / ("models--" + spec["id"].replace("/", "--")) / "snapshots" / spec["revision"]
        required = {"config.json", "tokenizer_config.json", "tokenizer.json"}
        index = snapshot / "model.safetensors.index.json"
        if index.is_file():
            required.update(json.loads(index.read_text())["weight_map"].values())
        else:
            required.add("model.safetensors")
        missing = [name for name in sorted(required) if not (snapshot / name).is_file()]
        if missing:
            raise ValueError(f"{size}: incomplete cache at {snapshot}: {', '.join(missing)}")
    print(f"Pinned train/dev and {', '.join(sizes)} cache paths checked.", flush=True)


def scheduler_command(cluster, size, condition, seed, log_dir):
    # Resource requests stay here; the training command is the same on both
    # sites. In particular, the LST H100 profile cannot fall back to an A100.
    if size == "27B" and cluster != "lst":
        raise ValueError("27B requires the LST H100 profile")
    heavy = size == "27B"
    cpus, ram, hours = (8, 128, 72) if heavy else (4, 64, 24 if size == "2B" else 48)
    job = f"qwen35-formal-{size}-{condition}-s{seed}"
    if cluster == "habrok":
        # --wrap invokes the real worker path on shared storage. Slurm copies
        # its own generated script to spool, so it must not infer that path.
        worker = ["bash", str(HERE / "worker.sh"), cluster, size, condition, str(seed)]
        return ["sbatch", "--export=ALL", "--partition=gpu", "--nodes=1", "--ntasks=1",
                "--gpus-per-node=a100:1", f"--cpus-per-task={cpus}", f"--mem={ram}G",
                f"--time={hours}:00:00", f"--job-name={job}",
                f"--output={log_dir}/%x-%j.out", f"--error={log_dir}/%x-%j.err",
                "--wrap", shlex.join(worker)]
    for path in (str(LORA), str(log_dir)):
        if any(ch.isspace() or ch in "'\"$#\\" for ch in path):
            raise ValueError("LST paths must be plain unquoted Condor paths")
    macros = {
        "LORA_DIR": LORA, "LOG_DIR": log_dir, "SIZE": size, "CONDITION": condition,
        "SEED": seed, "CPUS": cpus, "RAM": f"{ram}G", "WALL_SECONDS": hours * 3600,
        "CAPABILITY": "9.0" if heavy else "8.0",
        "MIN_GPU_MB": 79000 if heavy else 39000,
        "MAX_GPU_MB": 85000 if heavy else 41000,
    }
    cmd = ["condor_submit", str(HERE / "lst.sub")]
    for name, value in macros.items():
        cmd.extend(["-append", f"{name} = {value}"])
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", required=True, choices=("habrok", "lst"))
    parser.add_argument("--sizes", nargs="+", choices=SETTINGS,
                        help="Default: Hábrók 2B/9B; LST 27B. Explicit sizes for handover.")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--log-dir", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--submit", action="store_true", help="Check inputs and submit the printed commands")
    action.add_argument("--check", action="store_true", help="Check local cluster inputs without submitting")
    args = parser.parse_args()
    sizes = args.sizes or (("2B", "9B") if args.cluster == "habrok" else ("27B",))
    for values in (sizes, args.conditions, args.seeds):
        if len(values) != len(set(values)):
            parser.error("Duplicate size, condition or seed")
    if any(not 0 <= seed < 2**32-3 for seed in args.seeds):
        parser.error("Seeds must fit the Python and NumPy 32-bit seed range")
    logs = args.log_dir or (HERE / "logs" if args.cluster == "habrok"
                            else Path("/scratch/hongxuzhou/logs"))
    logs = logs.resolve()
    try:
        commands = [scheduler_command(args.cluster, size, condition, seed, logs)
                    for size, condition, seed in product(sizes, args.conditions, args.seeds)]
        if args.check or args.submit:
            check_inputs(args.cluster, sizes)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
    print(f"{len(commands)} job(s)", flush=True)
    if args.submit:
        logs.mkdir(parents=True, exist_ok=True)
        for cmd in commands:
            subprocess.run(cmd, check=True, env=os.environ)


if __name__ == "__main__":
    main()
