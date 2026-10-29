"""Rebuild machine diagnostics without changing synthesis outputs or labels.

Run with the repairsbn Python environment from the repository root:
    python -m repair_data_gen.shared_pipeline.analyse_split --config PATH

Pair keys are diagnostic candidates, not a frozen exclusion policy. No row
is accepted for evaluation by this script; human review remains separate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, InvalidOperation
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

from .split_config import load_config

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "repair_data_gen/rsr_generation"
sys.path.insert(0, str(GEN))
sys.path.insert(0, str(GEN.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from .select_candidates import prompt_sha  # noqa: E402
from audit import CHECKS, auto_pass  # noqa: E402
import check_repair_part as part  # noqa: E402
from check_place_r5 import pin as place_pin  # noqa: E402
from check_temporal_extra import pin as temporal_pin  # noqa: E402
from sbn_lin import parse_sbn_line  # noqa: E402
from select_docs import read_split, TEMPORAL_ROLES, GOLD, holdout  # noqa: E402
from wn_lookup import _exact  # noqa: E402
from run_llm import result_row  # noqa: E402


def read(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normal_value(value):
    value = value.strip().strip('"').casefold()
    try:
        return str(Decimal(value).normalize())
    except InvalidOperation:
        return value


def synset(value):
    lemma, pos, num = value.rsplit(".", 2)
    found = _exact(lemma, pos, num)
    return found.name() if found else value


def node_signature(concept):
    constants = sorted((r, normal_value(v)) for r, v in concept.roles
                       if not re.fullmatch(r"[+\-<>]\d+", v))
    return json.dumps([synset(concept.synset), constants], ensure_ascii=False)


def pair_key(row):
    """A directed, type-specific diagnostic key derived from actual boxes.

    Lexical pairs use canonical WordNet synsets; constants include the role;
    tense and edge pairs retain the event synset. Forwarding retains subject
    constants and its predicate, rather than equating all person subjects.
    """
    doc = parse_sbn_line(row["sbn_repair"])
    rep, fix = part.boxes(doc)
    gold = parse_sbn_line(row["sbn_clean"])
    site = gold.concepts[int(row["site_index"])]
    kind = row["repair_type"]
    context = ""
    if kind == "content_word":
        return kind, "synset", synset(row["reparandum_synset"]), synset(site.synset)
    if kind in {"constant_name", "constant_number", "constant_temporal"}:
        roles = ({"Name"} if kind == "constant_name" else
                 TEMPORAL_ROLES if kind == "constant_temporal" else
                 {r for r, v in site.roles if re.fullmatch(r"-?\d+(?:\.\d+)?", v)})
        matches = [(r, v) for c in rep for r, v in c.roles
                   if r in roles and normal_value(v) == normal_value(row["reparandum_value"])]
        matches = sorted(set(matches))
        if len(matches) != 1:
            raise ValueError(f"expected one reparandum constant, got {matches}")
        role, value = matches[0]
        targets = [v for r, v in site.roles if r == role]
        if len(targets) != 1:
            raise ValueError("ambiguous gold constant")
        return kind, role, normal_value(value), normal_value(targets[0])
    if kind == "tense":
        def tense(cs):
            # The tense node may be shared outside this box; later clauses
            # can also have unrelated tense nodes inside the repair box.
            verbs = [c for c in cs if c.synset == site.synset]
            if not verbs:
                raise ValueError("no corresponding tense verb")
            verb = verbs[0]
            times = [doc.concepts[verb.pos + int(v)] for r, v in verb.roles
                     if r == "Time" and re.fullmatch(r"[+-]\d+", v)]
            local = [c for c in times if c.box == verb.box and abs(c.pos - verb.pos) == 1]
            times = local or times
            values = sorted({r + " " + v for c in times if c.synset == "time.n.08"
                             for r, v in c.roles if r in {"EQU", "TPR", "TSU"} and v == "now"})
            if len(values) != 1:
                raise ValueError(f"ambiguous tense {values}")
            return values[0]
        return kind, synset(site.synset), tense(rep), tense(fix)
    if kind == "edge_label":
        a, b = part.place(doc, rep), part.place(doc, fix)
        if row["reparandum_value"] in {"Source", "Destination", "Location"}:
            def role(cs):
                values = sorted({r for c in cs for r, _ in c.roles
                                 if r in {"Source", "Destination", "Location"}})
                if len(values) != 1:
                    raise ValueError(f"ambiguous edge role {values}")
                return values[0]
            a, b = role(rep), role(fix)
        if a is None or b is None:
            raise ValueError("unresolved spatial operator")
        return kind, synset(site.synset), a, b
    if kind == "forwarding":
        subjects = [c for c in rep if c.synset == row["reparandum_synset"]]
        if len(subjects) != 1:
            raise ValueError("ambiguous forwarding subject")
        verbs = [c.synset for c in rep if c.pos_tag == "v"]
        if len(verbs) != 1:
            raise ValueError("ambiguous forwarding predicate")
        context = synset(verbs[0])
        return kind, context, node_signature(subjects[0]), node_signature(site)
    raise ValueError(kind)


def quote_diagnostic(row):
    """Test quote-only differences on copies; never change generated text."""
    def clean(s):
        s = s.translate(str.maketrans("’‘“”", "''\"\""))
        return re.sub(r"(?<!\w)'|'(?!\w)", "", s).replace('"', '')
    clone = dict(row)
    onset = int(row["repair_onset"])
    left, right = clean(row["nl_repair"][:onset - 2]), clean(row["nl_repair"][onset:])
    clone.update(nl_clean=clean(row["nl_clean"]), nl_repair=left + ", " + right,
                 reparandum_surface=clean(row["reparandum_surface"]), repair_onset=str(len(left) + 2))
    return not part.problems(clone)


def failure_flags(row):
    flags = [k for k in CHECKS + ("wn_ok", "untouched", "repair_part", "withdraw", "site_ok")
             if row.get(k) != "ok"]
    flags += [k for k, v in (("place_inside", "yes"), ("wn_relation", "same_synset"))
              if row.get(k) == v]
    if row.get("pin", "ok") != "ok":
        flags.append("pin")
    if row.get("declared_site", "ok") != "ok":
        flags.append("declared_site")
    return flags


def exact_surface(row):
    """Raw surface matches PMB outside insertion (first-letter case allowed)."""
    gold, text = row["nl_clean"], row["nl_repair"]
    onset = int(row["repair_onset"])
    tail = text[onset:]
    cut = len(gold) - len(tail)
    return (bool(tail) and cut >= 0 and gold[cut:cut+1].lower() == tail[:1].lower()
            and gold[cut+1:] == tail[1:] and text[:cut] == gold[:cut])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--logs", type=Path, help="downloaded HTCondor logs (read only)")
    args = parser.parse_args()
    config = load_config(args.config, check_inputs=True)
    out = args.out or config.workspace / "analysis/first_pass"
    if out.resolve().is_relative_to((GEN / "dataset").resolve()):
        raise ValueError("cannot write analysis into frozen train dataset")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = config.workspace / "manifests/candidate_samples.json"
    manifest = json.loads(manifest_path.read_text())
    gold = {d.doc_id: d for d in read_split(config.pmb_source)}
    integrity, stages, all_rows, missing, inputs = [], [], [], [], {}
    log_evidence = []
    pair_sources = config.pair_exclusion_sources
    train_rows = [r for path in pair_sources for r in read(path)]
    train_ids = {r["doc_id"] for r in train_rows}
    held_out = holdout.load()
    fluent_train = [d for d in read_split(GOLD) if d.doc_id not in held_out]
    if set(gold) & {d.doc_id for d in fluent_train}:
        integrity.append("PMB source documents overlap training fluent sources")
    train_sentences = defaultdict(list)
    for doc in fluent_train:
        train_sentences[doc.sentence].append(doc.doc_id)
    clean_overlaps = [dict(dev_doc_id=d.doc_id, train_doc_ids=";".join(train_sentences[d.sentence]),
                           nl_clean=d.sentence) for d in gold.values() if d.sentence in train_sentences]
    inputs[str(args.config)] = sha(args.config)
    inputs[str(config.pmb_source)] = sha(config.pmb_source)
    inputs[str(GOLD)] = sha(GOLD)
    if sha(args.config) != manifest["config_sha256"]:
        integrity.append("config differs from candidate manifest")
    if sha(config.pmb_source) != manifest["pmb_sha256"]:
        integrity.append("PMB differs from candidate manifest")
    for entry in manifest["outputs"]:
        sample_path = ROOT / entry["path"]
        samples = read(sample_path)
        tag = sample_path.stem
        results = list((config.workspace / "results").glob(f"*.{tag}.tsv"))
        if len(results) != 1:
            raise ValueError(f"expected one result for {tag}, found {results}")
        result = results[0]
        originals = {r["request_id"]: r for r in read(result)}
        raw_dir = result.parent / "raw" / result.stem
        metas = sorted(raw_dir.glob("run_meta_*.json"))
        if not metas:
            integrity.append(f"{tag}: missing run metadata")
        for path in metas:
            inputs[str(path)] = sha(path)
            meta = json.loads(path.read_text())
            prompt_name = Path(meta["docs"]).name
            current_prompt = prompt_sha(GEN / "prompts" / prompt_name)
            if (meta["sample_sha256"] != sha(sample_path)
                    or meta["config_sha256"] != sha(args.config)
                    or meta["prompt_sha256"] != manifest["prompt_sha256"][prompt_name]
                    or current_prompt != meta["prompt_sha256"]):
                integrity.append(f"{tag}: run metadata hashes differ")
        inputs[str(sample_path)] = sha(sample_path)
        inputs[str(result)] = sha(result)
        if sha(sample_path) != entry["sha256"]:
            integrity.append(f"{tag}: sample hash mismatch")
        sample_by_id = {r["request_id"]: r for r in samples}
        if len(sample_by_id) != len(samples):
            integrity.append(f"{tag}: duplicate sample request ID")
        checked_path = out / f"{tag}.checked.tsv"
        proc = subprocess.run([sys.executable, str(GEN / "analyse_run.py"), str(result),
                               "--out", str(checked_path)], text=True, capture_output=True)
        (out / f"{tag}.checks.txt").write_text(proc.stdout + proc.stderr)
        proc.check_returncode()
        rows = read(checked_path)
        counts = Counter(r["request_id"] for r in rows)
        if any(n != 1 for n in counts.values()):
            integrity.append(f"{tag}: duplicate result request ID")
        missing.extend({**s, "stage": tag} for k, s in sample_by_id.items() if k not in counts)
        for row in rows:
            request = row["request_id"]
            sample = sample_by_id[request]
            stem = re.sub(r"[^\w.-]+", "_", request).strip("_")
            stem += "_" + hashlib.sha256(request.encode()).hexdigest()[:12]
            raw_path = raw_dir / (stem + ".json")
            if not raw_path.exists():
                integrity.append(f"{request}: missing raw response")
            else:
                inputs[str(raw_path)] = sha(raw_path)
                original = originals[request]
                rebuilt = result_row(sample, original["model"], raw_path.read_text(),
                                     float(original["duration_s"]), raw_path)
                differs = [k for k, v in rebuilt.items()
                           if k != "raw_file" and str(v) != original.get(k, "")]
                if differs:
                    integrity.append(f"{request}: raw-to-TSV mismatch: {differs}")
            row["stage"] = tag
            for field in ("doc_id", "repair_type", "nl_clean", "sbn_clean"):
                if row[field] != sample[field]:
                    integrity.append(f"{request}: source mismatch: {field}")
            source = gold.get(row["doc_id"])
            if source is None or source.sentence != row["nl_clean"] or source.raw != row["sbn_clean"]:
                integrity.append(f"{request}: not identical to PMB source")
            if row["doc_id"] in train_ids:
                integrity.append(f"{request}: source document in frozen train RSR")
            if "pinned_site" in sample:
                row["pin"] = (temporal_pin if row["repair_type"] == "constant_temporal" else place_pin)(row, sample)
            else:
                row["pin"] = "ok" if row["status"] == "ok" else "-"
            row["declared_site"] = "-"
            if row["status"] == "ok":
                cs = parse_sbn_line(row["sbn_clean"]).concepts
                i = int(row["site_index"])
                row["declared_site"] = ("ok" if 0 <= i < len(cs) and cs[i].synset == row["repair_synset"]
                                        else "site_index/synset mismatch")
            row["machine_pass"] = str(auto_pass(row) and row["pin"] == "ok" and row["declared_site"] == "ok")
            row["exact_surface"] = str(exact_surface(row)) if row["status"] == "ok" else "-"
        stages.append(dict(stage=tag, requests=len(samples), returned=len(rows),
                           missing=len(samples)-len(rows)))
        all_rows.extend(rows)
        if args.logs:
            logs = sorted(args.logs.glob(tag + ".*.log"))
            stdout = sorted(args.logs.glob(tag + ".*.out"))
            for path in logs + stdout:
                inputs[str(path)] = sha(path)
            normal = len(logs) == 1 and "with exit-code 0" in logs[0].read_text()
            failures_log = [line for path in stdout for line in path.read_text().splitlines() if " FAILED: " in line]
            missing_docs = {r["doc_id"] for r in missing if r["stage"] == tag}
            failed_docs = {line.split(" FAILED:")[0].split()[-1] for line in failures_log}
            if not normal or missing_docs != failed_docs:
                integrity.append(f"{tag}: scheduler or missing-request log mismatch")
            log_evidence.append(dict(stage=tag, normal_exit_zero=normal, failures=failures_log))
    passed = [r for r in all_rows if r["machine_pass"] == "True"]
    failures = []
    for row in all_rows:
        if row["status"] != "ok" or row["machine_pass"] == "True":
            continue
        flags = failure_flags(row)
        quote_only = flags == ["repair_part"] and quote_diagnostic(row)
        failures.append({**row, "failure_checks": ";".join(flags),
                         "quote_normalisation_only": str(quote_only),
                         "disposition": "hold_quote_policy" if quote_only else "exclude_under_current_checks"})
    collisions, unresolved = [], []
    reference = defaultdict(list)
    for row in train_rows:
        try:
            reference[pair_key(row)].append(row["item_id"])
        except (ValueError, IndexError, TypeError) as error:
            unresolved.append(dict(split="train", item_id=row["item_id"], error=str(error)))
    for row in passed:
        try:
            key = pair_key(row)
            reverse = (*key[:2], key[3], key[2])
            row["pair_key"] = json.dumps(key, ensure_ascii=False)
            for direction, candidate in (("same_direction", key), ("reverse_direction", reverse)):
                if candidate in reference:
                    collisions.append(dict(request_id=row["request_id"], repair_type=row["repair_type"],
                                           direction=direction, key=json.dumps(key, ensure_ascii=False),
                                           train_item_ids=";".join(reference[candidate])))
        except (ValueError, IndexError, TypeError) as error:
            unresolved.append(dict(split=config.split, item_id=row["request_id"], error=str(error)))
    duplicates = []
    for name, key in (("same_input", lambda r: r["nl_repair"]),
                      ("same_source_type_site", lambda r: (r["doc_id"], r["repair_type"], r["site_index"]))):
        groups = defaultdict(list)
        for row in passed:
            groups[key(row)].append(row["request_id"])
        duplicates += [dict(check=name, request_ids=";".join(ids)) for ids in groups.values() if len(ids)>1]
    write(out / "all_checked.tsv", all_rows)
    write(out / "machine_pass.tsv", passed)
    write(out / "failure_review.tsv", failures)
    write(out / "surface_review.tsv", [r for r in passed if r["exact_surface"] == "False"])
    write(out / "missing.tsv", missing)
    write(out / "pair_collisions.tsv", collisions, ["request_id", "repair_type", "direction", "key", "train_item_ids"])
    write(out / "pair_unresolved.tsv", unresolved, ["split", "item_id", "error"])
    write(out / "duplicates.tsv", duplicates, ["check", "request_ids"])
    write(out / "clean_text_overlap.tsv", clean_overlaps, ["dev_doc_id", "train_doc_ids", "nl_clean"])
    for path in pair_sources:
        inputs[str(path)] = sha(path)
    inputs[str(manifest_path)] = sha(manifest_path)
    codes = [Path(__file__), Path(__file__).with_name("split_config.py"),
             Path(__file__).with_name("select_candidates.py"),
             GEN / "analyse_run.py", GEN / "audit.py", GEN / "check_place_r5.py",
             GEN / "check_temporal_extra.py", GEN / "check_repair_part.py", GEN / "check_untouched.py",
             GEN / "check_withdraw.py", GEN / "select_docs.py", GEN / "wn_lookup.py",
             GEN.parent / "sbn_lin.py", GEN.parent / "validate_with_pmb_parser.py"]
    summary = dict(status="machine diagnostics only; not frozen or human accepted", stages=stages,
                   requests=sum(x["requests"] for x in stages), returned=len(all_rows),
                   model_ok=sum(r["status"] == "ok" for r in all_rows),
                   no_site=sum(r["status"] == "no_site" for r in all_rows),
                   missing=len(missing), machine_pass=len(passed), failed=len(failures),
                   machine_pass_exact_surface=sum(r["exact_surface"] == "True" for r in passed),
                   machine_pass_surface_review=sum(r["exact_surface"] == "False" for r in passed),
                   quote_policy_holds=sum(r["quote_normalisation_only"] == "True" for r in failures),
                   integrity_errors=integrity, duplicates=duplicates,
                   fluent_source_document_overlap=0 if not set(gold) & {d.doc_id for d in fluent_train} else 1,
                   fluent_clean_text_overlap=len(clean_overlaps),
                   scheduler_evidence=log_evidence,
                   pair_policy="diagnostic only; exclusion scope and direction are not frozen",
                   pair_unresolved=dict(Counter(r["split"] for r in unresolved)),
                   pair_collision_items_by_type=dict(Counter(r["repair_type"] for r in passed
                       if r["request_id"] in {c["request_id"] for c in collisions})),
                   per_type={t: dict(returned=sum(r["repair_type"] == t for r in all_rows),
                                     machine_pass=sum(r["repair_type"] == t for r in passed),
                                     ambiguous=dict(Counter(r["ambiguous"] for r in passed
                                                            if r["repair_type"] == t and r["ambiguous"])))
                             for t in config.base_cap_per_type},
                   runtime=dict(python=platform.python_version(), executable=sys.executable,
                                nltk=version("nltk"), networkx=version("networkx")),
                   input_sha256=inputs, code_sha256={str(p): sha(p) for p in codes})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"input_sha256", "code_sha256"}}, indent=2))
    if integrity:
        raise SystemExit("source integrity failures; do not use these candidates")


if __name__ == "__main__":
    main()
