"""Build reproducible PMB-gold candidate samples for RSR synthesis.

This selects fluent NL--SBN bases only. It does not call a model and does not
create final RSR data. The dev and initial test plans write four cluster inputs:

- base prompt: content word, name, number, tense and forwarding;
- temporal pass 1: every eligible document, with its first site pinned;
- edge direction: every eligible direction document;
- edge place: every place-only document, with its first site pinned.

A later temporal pass depends on generated reparanda and is not included
in the initial test batch. Every row has a stable ``request_id`` for resume, raw filenames,
checks and audits.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
REPAIR_ROOT = HERE.parent
PROJECT_ROOT = REPAIR_ROOT.parent
GENERATION = REPAIR_ROOT / "rsr_generation"
sys.path.insert(0, str(GENERATION))
sys.path.insert(0, str(REPAIR_ROOT))

import select_docs as selection  # noqa: E402
try:
    from .split_config import RepairSplitConfig, load_config
except ImportError:  # Direct script execution.
    from split_config import RepairSplitConfig, load_config


BASE_TYPES = (
    "content_word", "constant_name", "constant_number", "tense", "forwarding"
)
BASE_FIELDS = (
    "request_id", "doc_id", "repair_type", "nl_clean", "sbn_clean",
    "n_concepts", "n_tokens_nl", "eligible_sites",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_sha(path: Path) -> str:
    system_files = ("sbn_manual.md", "repair_notation.md", "task.md")
    system_prompt = "\n\n---\n\n".join(
        (path / name).read_text(encoding="utf-8") for name in system_files
    )
    digest = hashlib.sha256(system_prompt.encode("utf-8"))
    digest.update((path / "item_template.md").read_bytes())
    return digest.hexdigest()


def common_row(doc, repair_type: str, sites, request_id: str) -> dict[str, str | int]:
    return {
        "request_id": request_id,
        "doc_id": doc.doc_id,
        "repair_type": repair_type,
        "nl_clean": doc.sentence,
        "sbn_clean": doc.raw,
        "n_concepts": len(doc.concepts),
        "n_tokens_nl": len(doc.sentence.split()),
        "eligible_sites": "; ".join(f"{pos}:{site}" for pos, site in sites),
    }


def write_tsv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {"path": str(path.relative_to(PROJECT_ROOT)), "rows": len(rows),
            "sha256": sha256(path)}


def select_base(config: RepairSplitConfig, docs: list) -> tuple[list[dict], dict[str, int]]:
    functions: dict[str, Callable] = {
        "content_word": selection.site_content_word,
        "constant_name": selection.site_name,
        "constant_number": selection.site_number,
        "tense": selection.site_tense,
        "forwarding": selection.site_forwarding,
    }
    eligible = {
        repair_type: [(doc, sorted(set(fn(doc)))) for doc in docs if fn(doc)]
        for repair_type, fn in functions.items()
    }
    quotas = config.candidate_plan["base_per_type"]
    value_caps = config.candidate_plan["constant_value_caps"]
    rng = random.Random(config.seed)
    used: set[str] = set()
    rows: list[dict] = []
    selected: dict[str, int] = {}
    # Rarest pool first preserves scarce number candidates before common types.
    for repair_type in sorted(BASE_TYPES, key=lambda name: len(eligible[name])):
        candidates = eligible[repair_type][:]
        rng.shuffle(candidates)
        per_value: collections.Counter[str] = collections.Counter()
        count = 0
        for doc, sites in candidates:
            if count >= quotas[repair_type]:
                break
            if doc.doc_id in used:
                continue
            if repair_type.startswith("constant_"):
                cap = value_caps[repair_type]
                values = {site for _, site in sites}
                if cap and all(per_value[value] >= cap for value in values):
                    continue
                per_value.update(values)
            request_id = f"{config.split}_base::{repair_type}::{doc.doc_id}"
            rows.append(common_row(doc, repair_type, sites, request_id))
            used.add(doc.doc_id)
            count += 1
        if count != quotas[repair_type]:
            raise ValueError(
                f"only selected {count} of {quotas[repair_type]} candidates for {repair_type}"
            )
        selected[repair_type] = count
    rows.sort(key=lambda row: (selection.TYPES.index(row["repair_type"]), row["doc_id"]))
    return rows, {name: len(eligible[name]) for name in BASE_TYPES} | {
        f"selected:{name}": selected[name] for name in BASE_TYPES
    }


def temporal_rows(docs: list, split: str = "dev") -> list[dict]:
    rows = []
    for doc in docs:
        sites = sorted(set(selection.site_temporal(doc)))
        if not sites:
            continue
        pos, role_value = sites[0]
        request_id = f"{split}_temporal_p1::constant_temporal::{doc.doc_id}::site_{pos}"
        row = common_row(doc, "constant_temporal", sites, request_id)
        row.update(
            pinned_site=f"{pos} {doc.concepts[pos].synset} {role_value}",
            avoid_value="none",
            kind="first_pass",
        )
        rows.append(row)
    return sorted(rows, key=lambda row: row["doc_id"])


def direction_rows(docs: list, split: str = "dev") -> list[dict]:
    rows = []
    for doc in docs:
        sites = sorted(set(selection.site_direction(doc)))
        if sites:
            request_id = f"{split}_edge_direction::edge_label::{doc.doc_id}"
            rows.append(common_row(doc, "edge_label", sites, request_id))
    return sorted(rows, key=lambda row: row["doc_id"])


def place_rows(docs: list, split: str = "dev") -> list[dict]:
    rows = []
    for doc in docs:
        sites = sorted(set(selection.site_place_only(doc)))
        if not sites:
            continue
        pos, site = sites[0]
        request_id = f"{split}_edge_place::edge_label::{doc.doc_id}::site_{pos}"
        row = common_row(doc, "edge_label", sites, request_id)
        form = "plain" if " plain:" in site else "operator"
        verb = "be" if site.startswith("be.v.03 ") else "other"
        row.update(pinned_site=f"{pos} {site}", form=form, verb=verb)
        rows.append(row)
    return sorted(rows, key=lambda row: row["doc_id"])


def build(config: RepairSplitConfig) -> dict:
    docs_all = selection.read_split(config.pmb_source)
    skipped = list(selection.read_split.skipped)
    if skipped:
        raise ValueError(f"PMB reader skipped {len(skipped)} documents: {skipped[:5]}")
    if config.split in {"dev", "test"} and len(docs_all) != 1194:
        raise ValueError(f"expected 1,194 PMB {config.split} documents, found {len(docs_all)}")
    broken = [doc.doc_id for doc in docs_all if not selection.gold_is_sound(doc)]
    if broken and config.split != "test":
        raise ValueError(f"gold structural checks failed for {len(broken)} documents: {broken[:5]}")
    # Test follows the train source rule: record unsound gold and do not
    # synthesise from it. The unchanged fluent evaluation split is separate.
    docs = [doc for doc in docs_all
            if doc.doc_id not in selection.EXAMPLE_DOCS and doc.doc_id not in broken]

    isolation = None
    if config.split == "test":
        comparisons = {
            "train": selection.GOLD,
            "dev": PROJECT_ROOT / "data/pmb-5.1.0/split/en/dev/standard.sbn",
        }
        isolation = {}
        for name, path in comparisons.items():
            other = selection.read_split(path)
            if selection.read_split.skipped:
                raise ValueError(f"reader skipped {name} isolation inputs")
            ids = {d.doc_id for d in other}
            texts = {d.sentence for d in other}
            id_overlap = sorted(d.doc_id for d in docs_all if d.doc_id in ids)
            text_overlap = sorted(d.doc_id for d in docs_all if d.sentence in texts)
            if id_overlap or text_overlap:
                raise ValueError(f"test/{name} source overlap: ids={id_overlap}, text={text_overlap}")
            isolation[name] = dict(source=str(path.relative_to(PROJECT_ROOT)),
                                   sha256=sha256(path), document_overlap=0, exact_text_overlap=0)

    base, base_stats = select_base(config, docs)
    temporal = temporal_rows(docs, config.split)
    direction = direction_rows(docs, config.split)
    place = place_rows(docs, config.split)
    groups = [base, temporal, direction, place]
    request_ids = [row["request_id"] for rows in groups for row in rows]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("candidate request_id values are not unique")

    samples = config.workspace / "samples"
    outputs = [
        write_tsv(samples / f"{config.split}_base.tsv", base, BASE_FIELDS),
        write_tsv(
            samples / f"{config.split}_temporal_p1.tsv", temporal,
            BASE_FIELDS + ("pinned_site", "avoid_value", "kind"),
        ),
        write_tsv(samples / f"{config.split}_edge_direction.tsv", direction, BASE_FIELDS),
        write_tsv(
            samples / f"{config.split}_edge_place.tsv", place,
            BASE_FIELDS + ("pinned_site", "form", "verb"),
        ),
    ]
    prompt_root = REPAIR_ROOT / "rsr_generation" / "prompts"
    prompt_hashes = {
        version: prompt_sha(prompt_root / version)
        for version in sorted(set(config.prompt_versions.values()))
    }
    manifest = {
        "schema_version": 1,
        "status": "working candidates; not final RSR data",
        "split": config.split,
        "seed": config.seed,
        "config": str(config.path.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256(config.path),
        "builder": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "builder_sha256": sha256(Path(__file__)),
        "pmb_source": str(config.pmb_source.relative_to(PROJECT_ROOT)),
        "pmb_sha256": sha256(config.pmb_source),
        "pmb_documents": len(docs_all),
        "prompt_sha256": prompt_hashes,
        "selection": {
            "base": "fixed quotas, rarest type first, no source document reused across base types",
            "temporal_p1": "all eligible documents; first sorted site pinned",
            "temporal_p2": "deferred until pass-1 generated reparanda are checked",
            "edge_direction": "all eligible direction documents",
            "edge_place": "all place-only documents; first sorted site pinned",
        },
        "base_stats": base_stats,
        "candidate_counts": {
            "base": len(base),
            "temporal_p1": len(temporal),
            "edge_direction": len(direction),
            "edge_place": len(place),
            "total_requests": len(request_ids),
        },
        "outputs": outputs,
    }
    manifest_path = config.workspace / "manifests" / "candidate_samples.json"
    if config.split == "test":
        manifest["excluded_unsound_gold"] = broken
        manifest["source_isolation"] = isolation
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_config(args.config, check_inputs=True)
    if config.status != "working":
        raise ValueError(f"candidate selection requires working status, got {config.status!r}")
    manifest = build(config)
    print(json.dumps({"candidate_counts": manifest["candidate_counts"],
                      "outputs": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
