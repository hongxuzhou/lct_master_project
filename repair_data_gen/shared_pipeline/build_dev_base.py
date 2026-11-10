"""Export and independently validate the confirmed dev base, before markers.

Run with the repairsbn environment: python -m repair_data_gen.shared_pipeline.build_dev_base
Raw runs, audit labels, curation decisions and train data are read-only.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re

from .analyse_split import ROOT, GEN, read, sha, write
from .prepare_dev_review import decision, normalise_surface
from .select_candidates import prompt_sha
from .split_config import load_config
import analyse_run
import check_repair_part
import check_untouched
import check_withdraw
from audit import ITEM_FIELDS, read_labels
from check_place_r5 import pin as place_pin
from check_temporal_extra import pin as temporal_pin
from run_llm import result_row
from sbn_lin import read_split
from validate_with_pmb_parser import check


WORKSPACE = ROOT / "repair_data_gen/splits/dev"
EXPECTED = dict(content_word=60, constant_name=60, constant_number=60,
                constant_temporal=47, tense=60, edge_label=43, forwarding=60)
SEMANTIC_FIELDS = ("nl_clean", "sbn_clean", "sbn_repair", "site_index", "repair_type")
BASE_FIELDS = (
    "item_id", "repair_type", "doc_id", "nl_clean", "nl_repair", "repair_onset",
    "sbn_clean", "sbn_repair", "reparandum_surface", "reparandum_synset",
    "reparandum_value", "repair_synset", "site_index", "devices", "strategy",
    "wn_relation", "ambiguous", "interregnum", "label_source", "audit_ref",
    "source_file", "nl_repair_model", "repair_onset_model", "fixes", "aligned",
    "request_id", "audit_id", "stage", "raw_file", "alternative_readings",
    "marker_priority",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(rows, key):
    require(all(r.get(key) for r in rows), f"missing {key}")
    require(len({r[key] for r in rows}) == len(rows), f"duplicate {key}")
    return {r[key]: r for r in rows}


def boundary(row):
    text, onset = row["nl_repair"], int(row["repair_onset"])
    require(2 <= onset < len(text) and text[onset-2:onset] == ", ",
            f"{row['request_id']}: invalid marker insertion boundary")
    require(len(text.encode("utf-16-le")) // 2 == len(text),
            f"{row['request_id']}: Python/frontend character offsets disagree")
    require(row["interregnum"] == "none", f"{row['request_id']}: not an unmarked base")


def main():
    config_path = WORKSPACE / "config.json"
    config = load_config(config_path, check_inputs=True)
    selection_path = WORKSPACE / "manifests/base_selection.json"
    selection = json.loads(selection_path.read_text())
    inputs = {}

    def track(path):
        inputs[str(path)] = sha(path)

    def verify_hashes(mapping):
        for path, digest in mapping.items():
            require(sha(path) == digest, f"input changed: {path}")
            inputs[str(path)] = digest

    verify_hashes(selection["input_sha256"])
    track(selection_path)
    require(selection["base_items"] == 390 and selection["per_type"] == EXPECTED,
            "selection count differs from the confirmed design")
    selected = unique(selection["items"], "audit_id")
    unique(selection["items"], "request_id")
    require(len(selected) == 390, "selection does not contain 390 items")
    items_path = WORKSPACE / "audit/dev_review_v1/items.tsv"
    items = unique(read(items_path), "audit_id")
    labels = read_labels(items_path.parent)
    curation_path = items_path.parent / "curation.jsonl"
    curation = {r["audit_id"]: r for r in map(json.loads, curation_path.read_text().splitlines())}
    require({k for k, r in curation.items() if r.get("selection_status") == "selected"}
            == set(selected), "curation and selection disagree")
    pool_path = WORKSPACE / "analysis/dev_review_v1/review_pool.tsv"
    pool = unique(read(pool_path), "request_id")
    track(pool_path)
    policy_path = WORKSPACE / "review_policy.json"
    policy = json.loads(policy_path.read_text())
    track(policy_path)
    marker_policy_path = WORKSPACE / "marker_policy.json"
    marker_policy = json.loads(marker_policy_path.read_text()) if marker_policy_path.exists() else None
    if marker_policy:
        track(marker_policy_path)
    checked = {}
    for folder in ("first_pass_rechecked", "followup"):
        directory = WORKSPACE / "analysis" / folder
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text())
        require(not summary["integrity_errors"], f"integrity errors in {folder}")
        verify_hashes(summary["input_sha256"])
        source_path = directory / "all_checked.tsv"
        track(source_path)
        track(summary_path)
        for request, row in unique(read(source_path), "request_id").items():
            require(request not in checked, f"duplicate checked request: {request}")
            checked[request] = row
    gold = {d.doc_id: d for d in read_split(config.pmb_source)}
    require(len(gold) == 1194 and not read_split.skipped, "unexpected PMB dev inventory")
    track(config.pmb_source)
    train_path = ROOT / "data/pmb-5.1.0/split/en/train/gold.sbn"
    train = read_split(train_path)
    require(not read_split.skipped, "PMB train reader skipped documents")
    train_ids, train_text = {d.doc_id for d in train}, {d.sentence for d in train}
    track(train_path)
    train_base = GEN / "dataset/train_rsr.base.tsv"
    train_sha = sha(train_base)
    track(train_base)
    from .analyse_split import holdout
    held = holdout.load()
    original_manifest_path = WORKSPACE / "manifests/candidate_samples.json"
    original_manifest = json.loads(original_manifest_path.read_text())
    stage_data = {}
    output, diagnostics, errors = [], [], []
    for audit_id, chosen in selected.items():
        request = chosen["request_id"]
        item, row, raw_checked, review = items[audit_id], pool[request], checked[request], curation[audit_id]
        for field in ("request_id", "doc_id", "repair_type"):
            require(chosen[field] == item[field] == review[field], f"{request}: {field} identity mismatch")
        require(labels[audit_id]["meaning"] == labels[audit_id]["natural"] == "yes",
                f"{request}: missing confirmed human acceptance")
        require(review["decision"] == "keep" and review.get("edits") is None,
                f"{request}: unresolved edit in selected base")
        for field in ITEM_FIELDS:
            if field not in {"audit_id", "source"}:
                require(item[field] == row[field], f"{request}: frontend/pool mismatch: {field}")
        for field in SEMANTIC_FIELDS:
            require(row[field] == raw_checked[field], f"{request}: changed semantic/source field: {field}")
        reproduced, _ = normalise_surface(raw_checked)
        require((reproduced["nl_repair"], reproduced["repair_onset"]) ==
                (row["nl_repair"], row["repair_onset"]), f"{request}: surface normalization not reproducible")
        boundary(row)
        doc = gold[row["doc_id"]]
        require((row["nl_clean"], row["sbn_clean"]) == (doc.sentence, doc.raw), f"{request}: PMB source differs")
        require(row["doc_id"] not in train_ids | held and row["nl_clean"] not in train_text,
                f"{request}: training-source overlap")
        stage = row["stage"]
        if stage not in stage_data:
            sample_path = WORKSPACE / "samples" / (stage + ".tsv")
            sample = unique(read(sample_path), "request_id")
            results = list((WORKSPACE / "results").glob(f"*.{stage}.tsv"))
            require(len(results) == 1, f"{stage}: ambiguous result file")
            result_path = results[0]
            raw_dir = result_path.parent / "raw" / result_path.stem
            metas = list(raw_dir.glob("run_meta_*.json"))
            require(bool(metas), f"{stage}: missing run metadata")
            for path in metas:
                meta = json.loads(path.read_text())
                prompt = Path(meta["docs"]).name
                require(meta["sample_sha256"] == sha(sample_path) and
                        meta["config_sha256"] == sha(config_path) and
                        meta["prompt_sha256"] == original_manifest["prompt_sha256"][prompt] ==
                        prompt_sha(GEN / "prompts" / prompt), f"{stage}: metadata hash mismatch")
                track(path)
            track(sample_path)
            track(result_path)
            stage_data[stage] = sample, unique(read(result_path), "request_id"), raw_dir
        samples, originals, raw_dir = stage_data[stage]
        sample, original = samples[request], originals[request]
        for field in ("doc_id", "repair_type", "nl_clean", "sbn_clean"):
            require(original[field] == sample[field] == row[field], f"{request}: run/sample/source mismatch: {field}")
        stem = re.sub(r"[^\w.-]+", "_", request).strip("_") + "_" + hashlib.sha256(request.encode()).hexdigest()[:12]
        raw_path = raw_dir / (stem + ".json")
        rebuilt = result_row(sample, original["model"], raw_path.read_text(), float(original["duration_s"]), raw_path)
        require(all(str(v) == original.get(k, "") for k, v in rebuilt.items() if k != "raw_file"),
                f"{request}: raw response does not reconstruct the TSV")
        for field in ("nl_repair", "repair_onset", "sbn_repair", "reparandum_synset", "reparandum_value"):
            model_field = field + "_model" if field in {"nl_repair", "repair_onset"} else field
            require(original[field] == row[model_field], f"{request}: raw/normalised lineage mismatch: {field}")
        track(raw_path)
        fresh = dict(row)
        fresh.update(check(row["sbn_repair"]))
        fresh["reply"] = original["parse"]
        fresh["wn_ok"], fresh["wn_missing"] = analyse_run.new_concepts_exist(row["sbn_clean"], row["sbn_repair"])
        fresh["wn_relation"] = analyse_run.wn_relation(row)
        fresh["place_inside"] = analyse_run.place_inside(row)
        fresh["site_ok"] = analyse_run.site_ok(row)
        fresh["repair_part"] = ";".join(check_repair_part.problems(row)) or "ok"
        lost = check_untouched.missing_edges(row)
        fresh["untouched"] = "ok" if not lost else str(lost)
        missing, added = check_withdraw.diff(row)
        fresh["withdraw"] = "ok" if not missing and not added else str((missing, added))
        fresh["pin"] = ((temporal_pin if row["repair_type"] == "constant_temporal" else place_pin)(row, sample)
                        if "pinned_site" in sample else "ok")
        fresh["declared_site"] = "ok" if doc.concepts[int(row["site_index"])].synset == row["repair_synset"] else "mismatch"
        eligible, reason = decision(fresh, policy)
        # The legacy checker also reports marker and boundary errors. Only
        # its two gold-alignment diagnostics may be rescued by surface policy.
        allowed_surface_issues = {
            "rest of the sentence changed: repair part is not the end of the gold sentence",
            "gold words not kept in order",
        }
        if fresh["repair_part"] != "ok":
            unexpected = set(fresh["repair_part"].split(";")) - allowed_surface_issues
            if unexpected:
                eligible, reason = False, ";".join(sorted(unexpected))
        fresh["validated_review_eligible"], fresh["validation_reason"] = str(eligible), reason
        diagnostics.append(fresh)
        if not eligible:
            errors.append(dict(request_id=request, audit_id=audit_id, reason=reason))
        machine_ambiguity = analyse_run.ambiguous(row)
        alternatives = review.get("alternative_readings", [])
        ambiguous = "|".join(sorted(set(alternatives) | ({machine_ambiguity} if machine_ambiguity else set())))
        base = dict(row, item_id=config.item_id_prefix + "_" + audit_id.removeprefix("dev_"),
                    audit_id=audit_id, audit_ref=audit_id, source_file=str(result_path.relative_to(ROOT)),
                    raw_file=str(raw_path.relative_to(ROOT)), label_source="human_review_and_user_confirmed_curation",
                    ambiguous=ambiguous, alternative_readings="|".join(alternatives),
                    marker_priority="" if review.get("marker_priority") is None else str(review["marker_priority"]),
                    fixes="approved_surface_normalisation" if row["nl_repair"] != original["nl_repair"] else "",
                    aligned=row["surface_aligned"])
        output.append({key: base[key] for key in BASE_FIELDS})
    require(Counter(r["repair_type"] for r in output) == EXPECTED, "export per-type counts differ")
    unique(output, "item_id")
    unique(output, "request_id")
    require(len({r["nl_repair"] for r in output}) == len(output), "duplicate repair inputs")
    require(len({(r['doc_id'], r['repair_type'], r['site_index']) for r in output}) == len(output),
            "duplicate source/type/site")
    analysis_dir = WORKSPACE / "analysis/base_validation"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write(analysis_dir / "checked.tsv", diagnostics)
    # Exercise insertion/removal on every possible seen marker without
    # assigning any marker or writing a marked dataset.
    marker_helper = GEN / "dataset/insert_editing_expressions.py"
    spec = importlib.util.spec_from_file_location("train_marker_helpers", marker_helper)
    marker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(marker_module)
    dry_runs = 0
    if not errors:
        for row in output:
            for marker in config.marker_policy.markers:
                marked = marker_module.derive(row, marker)
                start, end = int(row["repair_onset"]), int(marked["repair_onset"])
                require(marked["nl_repair"][start:end] == marker + ", ", "marker insertion differs")
                require(marked["nl_repair"][:start] + marked["nl_repair"][end:] == row["nl_repair"],
                        "marker removal does not recover base")
                require(all(marked[k] == row[k] for k in ("sbn_clean", "sbn_repair")), "marker changed target")
                dry_runs += 1
    code_paths = [Path(__file__), Path(__file__).with_name("prepare_dev_review.py"),
                  Path(__file__).with_name("analyse_split.py"),
                  Path(__file__).with_name("select_candidates.py"),
                  Path(__file__).with_name("split_config.py"), marker_helper,
                  GEN / "dataset/apply_fixes.py", GEN / "analyse_run.py", GEN / "audit.py",
                  GEN / "check_repair_part.py", GEN / "check_untouched.py", GEN / "check_withdraw.py",
                  GEN / "check_place_r5.py", GEN / "check_temporal_extra.py", GEN / "select_docs.py",
                  GEN / "wn_lookup.py", GEN / "run_llm.py", GEN.parent / "sbn_lin.py",
                  GEN.parent / "validate_with_pmb_parser.py"]
    report = dict(status="passed" if not errors else "failed", selected=390, errors=errors,
                  per_type=dict(Counter(r["repair_type"] for r in output)),
                  distinct_documents=len({r["doc_id"] for r in output}),
                  exact_duplicate_inputs=0, duplicate_source_type_sites=0,
                  training_source_overlap=0, hard_negative_overlap=0,
                  check_counts={key: dict(Counter(r[key] for r in diagnostics)) for key in policy["hard_checks"]},
                  legacy_surface_holds=[dict(request_id=r["request_id"], repair_part=r["repair_part"])
                                        for r in diagnostics if r["repair_part"] != "ok"],
                  known_ambiguous_items=sum(bool(r["ambiguous"]) for r in output),
                  confirmed_marker_priority_candidates=sum(r["marker_priority"] == "True" for r in output),
                  marker_boundary_valid=390, raw_response_reconstruction_valid=390,
                  normalisation_reproduced=390, source_semantic_targets_unchanged=390,
                  marker_roundtrip_dry_runs=dry_runs, marked_variants_written=0,
                  python=platform.python_version(), input_sha256=inputs,
                  code_sha256={str(p): sha(p) for p in code_paths})
    report_path = analysis_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    require(not errors, f"base validation failed; see {report_path}")
    output.sort(key=lambda r: (r["repair_type"], r["request_id"]))
    dataset_dir = WORKSPACE / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    base_path = dataset_dir / "dev_rsr.base.tsv"
    write(base_path, output, BASE_FIELDS)
    require(read(base_path) == output, "TSV round trip changed data")
    require(sha(train_base) == train_sha, "frozen train changed")
    for path, digest in inputs.items():
        require(sha(path) == digest, f"input changed during build: {path}")
    manifest = dict(status="validated base; editing expressions not yet built", rows=390,
                    per_type=report["per_type"], file=str(base_path.relative_to(ROOT)), sha256=sha(base_path),
                    selection_manifest_sha256=sha(selection_path), validation_report_sha256=sha(report_path),
                    marker_quota=marker_policy["target_marked_items"] if marker_policy else 117, input_sha256=inputs,
                    code_sha256=report["code_sha256"])
    (WORKSPACE / "manifests/base_build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k not in {"input_sha256", "code_sha256"}}, indent=2))
    print(f"Built {base_path.relative_to(ROOT)}: {sha(base_path)}")


if __name__ == "__main__":
    main()
