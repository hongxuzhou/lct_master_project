"""Audit the assembled dev Parquet without changing data or freezing it.

Run in repairsbn with pinned tokenizer-only snapshots prepared by hf download:
    python -m repair_data_gen.training_dataset.validate_dev --tokenizer-root PATH
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from repair_data_gen.training_dataset.build import SCHEMA, holdout, read_split, sha256  # noqa: E402

sys.path.insert(0, str(ROOT / "repair_data_gen/rsr_generation"))
from validate_with_pmb_parser import check as graph_check  # noqa: E402
from check_withdraw import diff as withdraw_diff  # noqa: E402
from check_untouched import missing_edges  # noqa: E402
from analyse_run import new_concepts_exist  # noqa: E402

DEV = ROOT / "repair_data_gen/splits/dev"
PARQUET = HERE / "data/validation.parquet"
REPORT_DIR = DEV / "analysis/unified_validation"
INSTRUCTION = "Please parse the following sentence to discourse representation structure:"
EXPECTED = {"pmb_fluent": 1194, "rsr_base": 390, "rsr_editing_expression": 122}
TYPE_BASE = {"constant_name": 60, "constant_number": 60, "constant_temporal": 47,
             "content_word": 60, "edge_label": 43, "forwarding": 60, "tense": 60}
STRING_SOURCE = {
    "example_id": "variant_id", "item_id": "item_id", "doc_id": "doc_id",
    "text": "nl_repair", "sbn_clean": "sbn_clean", "sbn_repair": "sbn_repair",
    "repair_type": "repair_type", "editing_expression": "editing_expression",
    "ambiguity_type": "ambiguous", "reparandum_surface": "reparandum_surface",
    "reparandum_synset": "reparandum_synset", "reparandum_value": "reparandum_value",
    "repair_synset": "repair_synset", "strategy": "strategy", "wn_relation": "wn_relation",
    "label_source": "label_source",
}


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def stats(values):
    values = sorted(values)
    return {"min": values[0], "p50": values[math.ceil(len(values) * .5) - 1],
            "p95": values[math.ceil(len(values) * .95) - 1],
            "p99": values[math.ceil(len(values) * .99) - 1], "max": values[-1]}


@lru_cache(maxsize=None)
def checked_graph(sbn):
    return graph_check(sbn)


def main():
    if (HERE/'fluent_group.acceptance.json').exists():
        from repair_data_gen.training_dataset.validate_integrated import main as validate_current
        return validate_current()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    args = parser.parse_args()
    errors, checks, row_checks = [], {}, []
    inputs = {}

    def track(path):
        path = Path(path)
        inputs[str(path)] = sha256(path)
        return path

    def load_json(path):
        return json.loads(track(path).read_text(encoding="utf-8"))

    def require(condition, name, detail=None):
        if not condition:
            errors.append({"check": name, "detail": detail})

    track(Path(__file__))
    track(ROOT / "documentation/model_experiment_pipeline.md")
    track(ROOT / "colloquium_prep/data_prep_scripts/lora_data_prep/lora_data_gen.py")
    build = load_json(HERE / "validation.manifest.json")
    for entry in build["sources"].values():
        path = ROOT / entry["path"]
        require(sha256(track(path)) == entry["sha256"], "assembly_source_hash", entry["path"])
    require(sha256(track(PARQUET)) == build["sha256"], "validation_hash")
    for filename in ("base_build.json", "marked_build.json"):
        metadata = load_json(DEV / "manifests" / filename)
        for field in ("input_sha256", "code_sha256"):
            for path, expected in metadata[field].items():
                require(sha256(track(path)) == expected, "component_provenance_hash", path)

    table = pq.read_table(PARQUET)
    rows = table.to_pylist()
    require(table.schema.equals(SCHEMA, check_metadata=True), "shared_schema")
    require(Counter(row["variant"] for row in rows) == Counter(EXPECTED), "variant_counts")
    require(len({row["example_id"] for row in rows}) == len(rows), "unique_example_ids")
    required = ("example_id", "doc_id", "variant", "text", "sbn_clean", "sbn_repair", "source")
    require(all(all(row[field] for field in required) for row in rows), "required_fields")
    require(all(value is None or not isinstance(value, str) or value != ""
                for row in rows for value in row.values()), "empty_strings_are_null")
    checks["schema_and_counts"] = {"rows": len(rows), "columns": table.num_columns,
                                   "variants": dict(Counter(r["variant"] for r in rows)),
                                   "null_counts": {name: table[name].null_count for name in table.column_names}}

    fluent = [row for row in rows if row["variant"] == "pmb_fluent"]
    rsr = [row for row in rows if row["variant"] != "pmb_fluent"]
    docs = read_split(track(ROOT / "data/pmb-5.1.0/split/en/dev/standard.sbn"))
    require(not read_split.skipped and len(docs) == 1194, "complete_pmb_dev")
    require([(r["doc_id"], r["text"], r["sbn_clean"], r["sbn_repair"]) for r in fluent]
            == [(d.doc_id, d.sentence, d.raw, d.raw) for d in docs], "fluent_exact_source_and_order")
    repair_only = set(SCHEMA.names) - set(required) - {"label_source"}
    require(all(all(r[field] is None for field in repair_only) for r in fluent), "fluent_null_metadata")
    require(all(r["example_id"] == f"pmb_dev::{r['doc_id']}" and r["label_source"] == "pmb_gold"
                and r["source"] == "PMB 5.1.0 English standard dev" for r in fluent), "fluent_identity")

    augmented = read_tsv(track(DEV / "dataset/dev_rsr.augmented.tsv"))
    require(len(augmented) == len(rsr), "rsr_source_count")
    by_source = {r["variant_id"]: r for r in augmented}
    for row, source in zip(rsr, augmented):
        for out_key, in_key in STRING_SOURCE.items():
            require(row[out_key] == (source[in_key] or None), "rsr_exact_field", [row["example_id"], out_key])
        for key in ("site_index", "repair_onset"):
            require(row[key] == (int(source[key]) if source[key] else None),
                    "rsr_integer_field", [row["example_id"], key])
        expected_variant = "rsr_editing_expression" if source["editing_expression"] else "rsr_base"
        require(row["variant"] == expected_variant
                and row["source"] == "synthetic RSR from PMB 5.1.0 English standard dev",
                "rsr_variant_and_source", row["example_id"])
    require([r["example_id"] for r in rows] == [f"pmb_dev::{d.doc_id}" for d in docs]
            + [r["variant_id"] for r in augmented], "complete_row_order")

    bases = {r["item_id"]: r for r in rsr if r["variant"] == "rsr_base"}
    marked = [r for r in rsr if r["variant"] == "rsr_editing_expression"]
    policy = load_json(DEV / "marker_policy.json")
    require(len(bases) == 390 and len(marked) == 122, "base_marked_counts")
    require(Counter(r["repair_type"] for r in bases.values()) == Counter(TYPE_BASE), "base_type_counts")
    require(len({(r["doc_id"], r["repair_type"], r["site_index"]) for r in bases.values()}) == 390,
            "unique_base_sites")
    require({r["item_id"] for r in marked} == {r["item_id"] for r in bases.values() if r["ambiguity_type"]},
            "marked_covers_exact_ambiguity_subset")
    for row in marked:
        base = bases[row["item_id"]]
        source = by_source[row["example_id"]]
        start, end = int(source["editing_start"]), row["repair_onset"]
        require(start == base["repair_onset"] and int(source["editing_end"]) == start + len(row["editing_expression"]),
                "marker_offsets", row["example_id"])
        require(row["text"][start:end] == row["editing_expression"] + ", "
                and row["text"][:start] + row["text"][end:] == base["text"],
                "exact_marker_removal", row["example_id"])
        require(all(row[k] == base[k] for k in ("doc_id", "repair_type", "site_index", "sbn_clean", "sbn_repair")),
                "paired_identity_and_targets", row["example_id"])
    require(Counter(r["editing_expression"] for r in marked)
            == Counter({"I mean": 31, "no": 31, "no wait": 30, "actually": 30}), "marker_counts")
    require(not {r["editing_expression"] for r in marked} & set(policy["held_out_markers"]), "held_out_marker")
    require(all(r["sbn_clean"] != r["sbn_repair"] for r in rsr), "repair_targets_differ")
    checks["source_and_pairs"] = {"fluent_exact": len(fluent), "rsr_exact": len(rsr),
                                   "base_items": len(bases), "marked_pairs": len(marked),
                                   "marker_counts": dict(Counter(r["editing_expression"] for r in marked))}

    train = pq.read_table(HERE / "data/train.parquet").to_pylist()
    gold_train = read_split(track(ROOT / "data/pmb-5.1.0/split/en/train/gold.sbn"))
    require(not read_split.skipped and len(gold_train) == 9552, "complete_train_reference")
    dev_ids, train_ids = {r["doc_id"] for r in rows}, {d.doc_id for d in gold_train}
    checks["isolation"] = {
        "train_document_overlap": len(dev_ids & train_ids),
        "train_example_id_overlap": len({r["example_id"] for r in rows} & {r["example_id"] for r in train}),
        "hard_negative_overlap": len(dev_ids & holdout.load()),
        "exact_input_overlap_with_train": len({r["text"] for r in rows} & {r["text"] for r in train}),
        "source_surface_overlap_with_train_gold": len({d.sentence for d in docs} & {d.sentence for d in gold_train}),
        "duplicate_dev_inputs": len(rows) - len({r["text"] for r in rows}),
    }
    for key, count in checks["isolation"].items():
        require(count == 0, key, count)
    checks["condition_views"] = {
        "C1_rows": len(fluent), "C2_rows": len(rows), "C3_rows": len(rows),
        "C2_input_digest": digest([(r["example_id"], r["text"]) for r in rows]),
        "C3_input_digest": digest([(r["example_id"], r["text"]) for r in rows]),
        "target_fields": {"C1": "sbn_clean", "C2": "sbn_clean", "C3": "sbn_repair"},
    }

    semantic_counts = Counter()
    for row in rows:
        record = {"example_id": row["example_id"], "variant": row["variant"], "graphs": {}}
        for target in ("sbn_clean", "sbn_repair"):
            result = checked_graph(row[target])
            keys = ("parse", "dag", "index", "magnitude", "edges")
            if target == "sbn_repair" and row["variant"] != "pmb_fluent":
                keys += ("boxes",)
            record["graphs"][target] = {key: result[key] for key in keys}
            for key in keys:
                require(result[key] == "ok", "graph_" + key, [row["example_id"], target, result["detail"]])
            semantic_counts[target] += 1
        if row["variant"] != "pmb_fluent":
            missing, added = withdraw_diff(row)
            untouched = missing_edges(row)
            wn, missing_wn = new_concepts_exist(row["sbn_clean"], row["sbn_repair"])
            require(not missing and not added, "withdraw_to_gold", row["example_id"])
            require(not untouched, "untouched_gold_edges", row["example_id"])
            require(wn == "ok", "new_wordnet_concepts", [row["example_id"], missing_wn])
            record.update(withdraw_to_gold=not missing and not added, untouched=not untouched, wordnet=wn)
            semantic_counts["rsr_withdraw_untouched_wordnet"] += 1
        row_checks.append(record)
    checks["semantics"] = dict(semantic_counts)

    from datasets import load_dataset
    with tempfile.TemporaryDirectory(prefix="repairsbn-dev-acceptance-") as cache:
        loaded = load_dataset("parquet", data_files={"train": str(HERE / "data/train.parquet"),
                                                    "validation": str(PARQUET)}, cache_dir=cache)
        require(loaded["train"].features == loaded["validation"].features, "datasets_feature_match")
        require(loaded["validation"].to_list() == rows, "datasets_exact_readback")
    checks["local_datasets_readback"] = "passed"

    print(f"Data checks completed; {len(errors)} errors. Checking pinned tokenizers.", flush=True)
    from transformers import AutoTokenizer
    revisions = load_json(ROOT / "model/smoke_test/revisions.json")["models"]
    token_reports, token_rows = {}, []
    for size, entry in revisions.items():
        directory = args.tokenizer_root / size
        tokenizer_files = {p.name: sha256(track(p)) for p in sorted(directory.iterdir()) if p.is_file()}
        require("tokenizer.json" in tokenizer_files and "chat_template.jinja" in tokenizer_files,
                "complete_tokenizer_files", size)
        # hf download stores the resolved commit for each local file.
        for name in tokenizer_files:
            metadata_path = track(directory / ".cache/huggingface/download" / (name + ".metadata"))
            require(metadata_path.read_text().splitlines()[0] == entry["revision"],
                    "tokenizer_pinned_revision", [size, name])
        tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, use_fast=True)
        by_condition = {"C1": [], "C2": [], "C3": []}
        model_records = []
        for row in rows:
            messages = [{"role": "user", "content": INSTRUCTION + "\n" + row["text"]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            for condition, field in (("C2", "sbn_clean"), ("C3", "sbn_repair")):
                full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": row[field]}],
                                                     tokenize=False, add_generation_prompt=False, enable_thinking=False)
                ids = tokenizer(full, add_special_tokens=False)["input_ids"]
                prefix_ok = full.startswith(prompt) and ids[:len(prompt_ids)] == prompt_ids
                supervised = ids[len(prompt_ids):]
                eos_ok = tokenizer.eos_token_id in supervised
                require(prefix_ok, "exact_prompt_mask_prefix", [size, condition, row["example_id"]])
                require(bool(supervised) and row[field] in full[len(prompt):],
                        "nonempty_complete_assistant_target", [size, condition, row["example_id"]])
                require(eos_ok, "assistant_eos_supervised", [size, condition, row["example_id"]])
                result = {"model_size": size, "condition": condition, "example_id": row["example_id"],
                          "variant": row["variant"], "prompt_tokens": len(prompt_ids),
                          "sequence_tokens": len(ids), "supervised_tokens": len(supervised),
                          "truncated_at_1024": len(ids) > 1024,
                          "target_exceeds_512": len(supervised) > 512,
                          "prompt_mask_prefix_ok": prefix_ok, "supervised_eos": eos_ok}
                by_condition[condition].append(result)
                if condition == "C2" and row["variant"] == "pmb_fluent":
                    by_condition["C1"].append(result)
                token_rows.append(result)
                model_records.append(result)
        conditions = {}
        for condition, records in by_condition.items():
            conditions[condition] = {
                "rows": len(records),
                "prompt_tokens": stats([r["prompt_tokens"] for r in records]),
                "sequence_tokens": stats([r["sequence_tokens"] for r in records]),
                "supervised_tokens": stats([r["supervised_tokens"] for r in records]),
                "truncated_at_1024": sum(r["truncated_at_1024"] for r in records),
                "target_exceeds_512": sum(r["target_exceeds_512"] for r in records),
                "longest_example_ids": [r["example_id"] for r in records
                                        if r["sequence_tokens"] == max(x["sequence_tokens"] for x in records)],
            }
            require(conditions[condition]["truncated_at_1024"] == 0, "no_training_truncation", [size, condition])
            require(conditions[condition]["target_exceeds_512"] == 0, "generation_budget", [size, condition])
        token_reports[size] = {"model_id": entry["id"], "revision": entry["revision"],
                               "tokenizer_files": tokenizer_files, "conditions": conditions,
                               "lengths_digest": digest([(r["example_id"], r["condition"], r["prompt_tokens"],
                                                         r["sequence_tokens"], r["supervised_tokens"])
                                                        for r in model_records])}
        print(f"{size}: max C2/C3 tokens = {conditions['C2']['sequence_tokens']['max']}/"
              f"{conditions['C3']['sequence_tokens']['max']}", flush=True)

    previous = load_json(DEV / "analysis/base_validation/summary.json")
    for path, expected in list(inputs.items()):
        require(sha256(Path(path)) == expected, "input_unchanged_during_validation", path)
    report = {
        "status": "passed" if not errors else "failed", "errors": errors,
        "validation_sha256": sha256(PARQUET), "checks": checks,
        "tokenization": {"instruction": INSTRUCTION, "user_content": "instruction + '\\n' + text",
                         "instruction_source": "documentation/model_experiment_pipeline.md, section 3.2",
                         "user_content_source": "colloquium_prep/data_prep_scripts/lora_data_prep/lora_data_gen.py",
                         "enable_thinking": False, "training_max_length": 1024,
                         "generation_max_new_tokens": 512, "loss": "assistant suffix only, including EOS",
                         "template_status": "formal-design audit format; formal trainer integration pending",
                         "models": token_reports},
        "notes": {"known_surface_holds": previous["legacy_surface_holds"],
                  "lexical_pair_overlap_policy": "report_only; not an exclusion criterion",
                  "context_near_duplicate_checks": "not part of this acceptance; previously deferred",
                  "freeze": "not performed", "hub_publication": "not performed"},
        "versions": {name: version(name) for name in ("pyarrow", "datasets", "transformers", "tokenizers")},
        "input_sha256": inputs,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, records in (("rows.jsonl", row_checks), ("token_lengths.jsonl", token_rows)):
        (REPORT_DIR / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    report["outputs"] = {name: sha256(REPORT_DIR / name) for name in ("rows.jsonl", "token_lengths.jsonl")}
    (REPORT_DIR / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": errors, "report": str(REPORT_DIR / "summary.json")}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
