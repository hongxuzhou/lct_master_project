#!/usr/bin/env python3
"""LST-only canary: observe real kernel calls and verify online metric readback."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time


class StepLogger:
    """Merge train/eval records sharing an optimizer step before committing."""

    def __init__(self, run):
        self.run = run

    def __getattr__(self, name):
        return getattr(self.run, name)

    def log(self, values, step=None):
        self.run.log(values, step=step, commit=False)


class KernelProbe:
    """Observe Python entries during the first real training window only."""

    def __init__(self):
        self.calls = {"causal_conv1d": 0, "fla_gated_delta": 0}
        self.sources = {}

    def profile(self, frame, event, arg):
        if event != "call":
            return
        module = frame.f_globals.get("__name__", "")
        name = frame.f_code.co_name
        kind = None
        if module.startswith("causal_conv1d") and name == "causal_conv1d_fn":
            kind = "causal_conv1d"
        elif module.startswith("fla.") and name == "chunk_gated_delta_rule":
            kind = "fla_gated_delta"
        if kind:
            self.calls[kind] += 1
            self.sources[kind] = {"module": module, "function": name,
                                  "file": frame.f_code.co_filename}


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


def check_remote(wandb, config, run_id, expected_train, expected_eval):
    """Read the server, not local W&B spool files; allow brief sync latency."""
    last = {}
    for attempt in range(6):
        try:
            api = wandb.Api(overrides={"base_url": config["base_url"]}, timeout=30)
            remote = api.run(f"{config['entity']}/{config['project']}/{run_id}")
            train = {int(r["_step"]): r["train_loss"] for r in remote.scan_history(
                keys=["_step", "train_loss"])}
            dev = {int(r["_step"]): r["eval_loss"] for r in remote.scan_history(
                keys=["_step", "eval_loss"])}
            values_match = all(k in train and math.isclose(v, train[k], rel_tol=1e-6)
                               for k, v in expected_train.items()) and all(
                k in dev and math.isclose(v, dev[k], rel_tol=1e-6)
                for k, v in expected_eval.items())
            last = {"state": remote.state, "train_steps": sorted(train),
                    "eval_steps": sorted(dev), "values_match_local": values_match,
                    "summary_status": remote.summary.get("status")}
            if (remote.state.lower() == "finished" and values_match
                    and last["summary_status"] == "development_passed"):
                return {"passed": True, **last}
        except Exception as error:
            last = {"error_type": type(error).__name__}
        if attempt < 5:
            time.sleep(10)
    return {"passed": False, **last}


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import train

    args = train.parse_args()
    if (args.model_size, args.condition, args.purpose, args.max_steps,
            args.eval_every, args.generation_examples) != ("2B", "C3", "canary", 30, 10, 20):
        raise ValueError("This diagnostic requires 2B C3, 30 steps, eval every 10, 20 generations")
    output = args.output_dir
    if output.exists():
        raise FileExistsError(output)
    diagnostics = output.parent / (output.name + "-diagnostics")
    diagnostics.mkdir(parents=True, exist_ok=False)
    report_path = diagnostics / "acceptance.json"
    report = {"training_passed": False, "kernels_passed": False,
              "wandb_passed": False, "output_dir": str(output),
              "note": "Kernel profiling affects the first step timing; not a speed benchmark."}
    config = train.read_protocol(args.protocol)["wandb"]
    run = None
    wandb = None
    try:
        import wandb
        run = wandb.init(
            entity=config["entity"], project=config["project"],
            name=output.name, id=output.name, resume="never", mode="online",
            group="lst-kernels-wandb-canary", job_type="canary",
            tags=["lst", "2B", "C3", "canary", "kernel-check"],
            dir=str(diagnostics), settings=wandb.Settings(
                base_url=config["base_url"], disable_git=True, disable_code=True,
                console="off", init_timeout=45, finish_timeout=60))
        if run is None or run.settings.mode != "online":
            raise RuntimeError("W&B did not create an online run")
        report["wandb_url"] = run.url
        print("W&B online run:", run.url, flush=True)
    except Exception as error:
        report["wandb_init_error_type"] = type(error).__name__
        print("W&B initialization failed; continuing local training:", type(error).__name__, flush=True)
        if run is not None:
            run.finish(exit_code=1)
        run = None
    write_report(report_path, report)

    original = train.train_window
    probe = KernelProbe()
    observed = False

    def train_window(*a, **kw):
        nonlocal observed
        if observed:
            return original(*a, **kw)
        observed = True
        previous = sys.getprofile()
        sys.setprofile(probe.profile)
        try:
            return original(*a, **kw)
        finally:
            sys.setprofile(previous)
            report["kernel_calls_first_training_window"] = probe.calls
            report["kernel_sources"] = probe.sources
            report["kernels_passed"] = all(probe.calls.values())
            write_report(report_path, report)
            print("Kernel call evidence:", json.dumps(probe.calls), flush=True)

    train.train_window = train_window
    # Disable the trainer's second initialization if the explicit online init failed.
    args.wandb_mode = "online" if run is not None else "disabled"
    try:
        train.run_training(args, wandb_run=StepLogger(run) if run is not None else None)
        summary = json.loads((output / "summary.json").read_text())
        events = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        expected_train = {r["step"]: r["train_loss"] for r in events if r["event"] == "train"}
        expected_eval = {r["step"]: r["eval_loss"] for r in events if r["event"] == "eval"}
        report["training_passed"] = (summary["status"] == "development_passed"
            and summary["steps"] == 30 and summary["generation_examples"] == 20
            and set(expected_train) == set(range(1, 31)) and set(expected_eval) == {10, 20, 30})
        report["local_train_steps"] = sorted(expected_train)
        report["local_eval_steps"] = sorted(expected_eval)
        report["wandb_logging_errors"] = [r["event"] for r in events if r["event"].startswith("wandb_")]
        write_report(report_path, report)
        if run is not None:
            run.summary.update({"kernel_probe_passed": report["kernels_passed"]})
            run.finish(exit_code=0)
            run = None
            remote = check_remote(wandb, config, output.name, expected_train, expected_eval)
            report["wandb_readback"] = remote
            report["wandb_passed"] = remote["passed"] and not report["wandb_logging_errors"]
    except Exception as error:
        report["diagnostic_error_type"] = type(error).__name__
        raise
    finally:
        train.train_window = original
        try:
            if run is not None:
                run.finish(exit_code=1)
        finally:
            write_report(report_path, report)
            print("Acceptance report:", report_path, flush=True)
            print(json.dumps(report, indent=2), flush=True)
    return 0 if all(report[k] for k in ("training_passed", "kernels_passed", "wandb_passed")) else 2


if __name__ == "__main__":
    sys.exit(main())
