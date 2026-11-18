#!/usr/bin/env python3
"""Score imported formal runs, preserving existing predictions and scores."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

HERE = Path(__file__).resolve().parent
BATCH = HERE / "runs/formal-test-20261113"
AUDIT = HERE / "runs/gold-audit-20261115"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_slice(prediction, slice_name):
    started = time.monotonic()
    size, condition, seed = prediction.relative_to(BATCH).parts[:3]
    output = prediction.parent / "scores-scoped-v2"
    output.mkdir(exist_ok=True)
    reference = AUDIT / "references" / f"{slice_name}.{condition}.tsv"
    refs = pd.read_csv(reference, sep="\t", keep_default_na=False)
    raw = pd.read_csv(prediction, sep="\t", keep_default_na=False)
    if raw.duplicated(["example_id", "condition"]).any() or set(raw.condition) != {condition}:
        raise ValueError(f"Invalid prediction keys/condition: {prediction}")
    keys = set(zip(refs.example_id, refs.condition))
    selected = raw[[key in keys for key in zip(raw.example_id, raw.condition)]]
    selected_path = output / f"{slice_name}.predictions.tsv"
    if not selected_path.exists():
        selected.to_csv(selected_path, sep="\t", index=False, mode="x")
    else:
        existing = pd.read_csv(selected_path, sep="\t", keep_default_na=False)
        pd.testing.assert_frame_equal(existing, selected.reset_index(drop=True))
    basic = output / f"{slice_name}.basic.jsonl"
    checks = output / f"{slice_name}.checks.json"

    def run(script, args, stage):
        # Each new invocation has its own log, including failed invocations.
        log = output / f"{slice_name}.{stage}.{time.time_ns()}.log"
        with log.open("x") as handle:
            subprocess.run([sys.executable, str(HERE / script), *map(str, args)],
                           stdout=handle, stderr=subprocess.STDOUT, check=True, cwd=HERE.parent)

    if not basic.exists():
        run("score_basic.py", ["--references", reference, "--predictions", selected_path,
                               "--output", basic], "basic")
    rows = [json.loads(line) for line in basic.read_text().splitlines()]
    from score_basic import require_panel
    require_panel(rows, refs.to_dict("records"))
    if any(r["score_version"] != "scoped_sbn_v2" for r in rows):
        raise ValueError(f"Wrong scoring version: {basic}")
    predicted = dict(zip(selected.example_id, selected.pred_sbn))
    if any(r["pred_sbn"] != predicted.get(r["example_id"]) for r in rows):
        raise ValueError(f"Scored prediction mismatch: {basic}")
    errors = [r for r in rows if r["score_status"] not in {"scored", "incomplete"}]
    if errors:
        raise ValueError(f"{basic}: {len(errors)} gold/scorer errors; inspect preserved output")
    if not checks.exists() and (slice_name != "core_rsr" or condition == "C3"):
        args = ["--basic", basic, "--output", checks]
        if slice_name == "core_rsr":
            script = "check_core_repair.py"
            args += ["--gold", AUDIT / "retained/data/test.parquet"]
        elif slice_name == "later_reference":
            script = "check_later_reference.py"
            args += ["--references", reference, "--gold", AUDIT / "retained/later_reference/test.parquet"]
        else:
            script = "check_hard_negatives.py"
            args += ["--references", reference]
        run(script, args, "checks")
    if checks.exists():
        checked = json.loads(checks.read_text())["items"]
        require_panel(checked, refs.to_dict("records"))
    evidence = dict(size=size, condition=condition, seed=seed, slice=slice_name,
                    expected=len(refs), selected_predictions=len(selected),
                    pred_status=dict(Counter(r["pred_status"] for r in rows)),
                    score_status=dict(Counter(r["score_status"] for r in rows)),
                    score_version="scoped_sbn_v2", elapsed_seconds=time.monotonic()-started,
                    sha256={str(p.relative_to(HERE)): digest(p) for p in
                            [prediction, reference, selected_path, basic] + ([checks] if checks.exists() else [])})
    verification = output / f"{slice_name}.verification.json"
    if not verification.exists():
        with verification.open("x") as handle:
            json.dump(evidence, handle, indent=2)
    return {k: evidence[k] for k in ("size", "condition", "seed", "slice", "score_status", "elapsed_seconds")}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workers", type=int, default=6)
    cli.add_argument("--sizes", nargs="+", choices=("2B", "9B", "27B"),
                     default=["2B", "9B", "27B"])
    args = cli.parse_args()
    tasks = []
    for size in args.sizes:
        for condition in ("C1", "C2", "C3"):
            for seed in (24, 42, 4224):
                prediction = BATCH / size / condition / f"seed{seed}" / "predictions.tsv"
                if not prediction.is_file():
                    raise FileNotFoundError(prediction)
                slices = ["pmb_fluent", "hard_negative"]
                if condition != "C1":
                    slices += ["core_rsr", "later_reference"]
                tasks.extend((prediction, s) for s in slices)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_slice, *task): task for task in tasks}
        for future in as_completed(futures):
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as exc:
                failure = f"{futures[future]}: {exc}"
                failures.append(failure)
                print(failure, file=sys.stderr, flush=True)
    if failures:
        raise RuntimeError("\n".join(failures))
    print(f"Verified {len(tasks)} run/slice panels", flush=True)


if __name__ == "__main__":
    main()
