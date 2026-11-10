"""Join downloaded canary diagnostics to accepted dev gold; never alter inputs."""
import argparse
from collections import Counter
import contextlib
import io
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repair_data_gen"))
from validate_with_pmb_parser import check


def graph_pass(result, text):
    keys = ["parse", "dag", "index", "magnitude", "edges"]
    if "CORRECTION" in text.split() or "CONJUNCTION" in text.split():
        keys.append("boxes")
    return all(result[k] == "ok" for k in keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    gold = {r["example_id"]: r for r in pq.read_table(ROOT / "repair_data_gen/training_dataset/data/validation.parquet").to_pylist()}
    reports, details = {}, []
    for condition in ("C1", "C2", "C3"):
        directory = args.run_root / condition
        config = json.loads((directory / "config.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        generations = json.loads((directory / "dev_generations.json").read_text())
        audit = json.loads((directory / "data_audit.json").read_text())
        metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        train = [m for m in metrics if m["event"] == "train"]
        evaluations = [m for m in metrics if m["event"] == "eval"]
        start = next(m for m in metrics if m["event"] == "run_start")
        steps_per_epoch = math.ceil(audit["train"]["rows"] / start["effective_batch"])
        warmup = math.ceil(start["full_budget_steps"] * config["protocol"]["candidate"]["warmup_ratio"])
        records = []
        for generated in generations:
            row = gold[generated["example_id"]]
            target = row["sbn_repair" if condition == "C3" else "sbn_clean"]
            ids = generated["token_ids"]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                graph = check(generated["text"])
                gold_graph = check(target)
            if not graph_pass(gold_graph, target):
                raise ValueError(f"Gold diagnostic baseline failed: {row['example_id']}")
            record = {**generated, "condition": condition, "input": row["text"], "target": target,
                      "target_text_tokens": len(tokenizer(target, add_special_tokens=False)["input_ids"]),
                      "eos_anywhere": tokenizer.eos_token_id in ids,
                      "exact_target": generated["text"] == target,
                      "correction_mentions": generated["text"].split().count("CORRECTION"),
                      "conjunction_mentions": generated["text"].split().count("CONJUNCTION"),
                      "time_pointer_mentions": generated["text"].count("Time +"),
                      "repeated_now_nodes": generated["text"].count("time.n.08 EQU now"),
                      "graph_checks": graph,
                      "graph_checks_pass": graph_pass(graph, generated["text"])}
            records.append(record)
        details.extend(records)
        failures = [r for r in records if not r["ended_with_eos"]]
        reports[condition] = {
            "run_id": config["run_id"], "summary": summary,
            "dev_curve": [{k: e[k] for k in ("step", "eval_loss", "eval_target_tokens")} for e in evaluations],
            "steps_per_epoch": steps_per_epoch, "warmup_steps": warmup,
            "training_rows_seen": len(train) * start["effective_batch"],
            "epoch_fraction_seen": len(train) / steps_per_epoch,
            "last_logged_next_lr": train[-1]["learning_rate"],
            "train_step_seconds_median": statistics.median(m["step_seconds"] for m in train[1:]),
            "train_step_seconds_total": sum(m["step_seconds"] for m in train),
            "generation_tokens": sum(r["tokens"] for r in records),
            "unterminated_tokens": sum(r["tokens"] for r in failures),
            "eos_by_variant": {v: {"rows": sum(r["variant"] == v for r in records),
                                  "eos": sum(r["variant"] == v and r["ended_with_eos"] for r in records)}
                               for v in sorted({r["variant"] for r in records})},
            "failures": [{k:r[k] for k in ("example_id", "variant", "tokens", "target_text_tokens", "eos_anywhere", "time_pointer_mentions", "repeated_now_nodes")} for r in failures],
            "graph_checks_pass": sum(r["graph_checks_pass"] for r in records),
            "exact_targets": sum(r["exact_target"] for r in records),
            "correction_outputs": sum(r["correction_mentions"] > 0 for r in records),
            "conjunction_outputs": sum(r["conjunction_mentions"] > 0 for r in records),
        }
    output = args.run_root / "analysis"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    (output / "examples.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for axis, (condition, report) in zip(axes, reports.items()):
        axis.plot([r["step"] for r in report["dev_curve"]], [r["eval_loss"] for r in report["dev_curve"]], marker="o")
        axis.set(title=condition, xlabel="Optimizer step", xticks=[10, 20, 30])
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Assistant-target dev loss")
    fig.suptitle("2B canary: within-run learning curves (different target tasks)")
    fig.tight_layout()
    fig.savefig(output / "dev_loss.png", dpi=160)
    plt.close(fig)
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
