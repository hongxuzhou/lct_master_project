"""Preview full test jobs for one frozen cluster manifest."""
import argparse
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cluster", choices=("habrok", "lst"))
    p.add_argument("root", type=Path)
    p.add_argument("--habrok-on-lst", action="store_true")
    p.add_argument("--pilot", action="store_true",
                   help="Preview only C1 seed 24 for each imported model size")
    args = p.parse_args()
    root = args.root.resolve()
    if args.habrok_on_lst and args.cluster != "lst":
        p.error("--habrok-on-lst requires the LST execution profile")
    if args.pilot and not args.habrok_on_lst:
        p.error("--pilot requires --habrok-on-lst")
    origin = "habrok" if args.habrok_on_lst else args.cluster
    manifest = "frozen-habrok-on-lst.json" if args.habrok_on_lst else f"frozen-{origin}.json"
    rows = json.loads((root / manifest).read_text())
    expected = 18 if origin == "habrok" else 9
    if len(rows) != expected or len({r["run_id"] for r in rows}) != expected:
        p.error("Incomplete frozen manifest")
    for r in rows:
        run_id = r["run_id"]
        if r["cluster"] != origin or not re.fullmatch(r"formal-[a-zA-Z0-9-]+", run_id):
            p.error(f"Invalid run: {run_id}")
        if args.pilot and (r["condition"], str(r["seed"])) != ("C1", "24"):
            continue
        if (root / "predictions" / f"{run_id}.tsv").exists():
            if args.habrok_on_lst:
                continue
            p.error(f"Prediction already exists: {run_id}")
        worker = ["bash", str(HERE / "test_worker.sh"), args.cluster, str(root), run_id]
        if args.habrok_on_lst:
            worker.append(manifest)
        log = root / "logs" / run_id
        if args.cluster == "habrok":
            cmd = ["sbatch", "--partition=gpu", "--nodes=1", "--ntasks=1",
                   "--gpus-per-node=a100:1", "--cpus-per-task=4", "--mem=64G",
                   "--time=24:00:00", f"--job-name=test-{r['model_size']}-{r['condition']}-s{r['seed']}",
                   f"--output={log}-%j.out", f"--error={log}-%j.err", "--wrap", shlex.join(worker)]
        else:
            if any(not re.fullmatch(r"[A-Za-z0-9_./-]+", str(v)) for v in [HERE, root, log]):
                p.error("LST needs plain paths")
            settings = [
                "universe = vanilla", "executable = /bin/bash", f"arguments = {' '.join(worker[1:])}",
                f"output = {log}.$(ClusterId).out", f"error = {log}.$(ClusterId).err",
                f"log = {log}.$(ClusterId).log", "request_cpus = 8", "request_memory = 128G",
                "request_gpus = 1", 'requirements = (TARGET.UidDomain == "coli.uni-saarland.de") && (GPUs_Capability == 9.0) && (GPUs_GlobalMemoryMb >= 79000) && (GPUs_GlobalMemoryMb <= 85000)',
                "+MaxWallTime = 86400", "notification = Never", "should_transfer_files = NO",
                "getenv = PYTHON,MODEL_CACHE", "queue 1",
            ]
            if args.habrok_on_lst:
                settings.insert(-1, 'environment = "PATH=/usr/bin:/bin:/usr/sbin:/sbin"')
            print("condor_submit <<'CONDOR_TEST_JOB'\n" + "\n".join(settings) + "\nCONDOR_TEST_JOB")
            continue
        print(shlex.join(cmd))


if __name__ == "__main__":
    main()
