#!/usr/bin/env python3
"""
Extract and classify "be" samples from a PMB gold.sbn file.

Data format (repeating 4-line blocks, blocks separated by a blank line):
    line 1: ID            e.g.  p50/d0779
    line 2: sentence       e.g.  The movie starts at ten o'clock.
    line 3: SBN annotation e.g.  movie.n.01 start.v.01 Theme -1 ...

Two tasks:
  1. Extract every sample whose *sentence* contains a finite/non-finite form
     of the copula/auxiliary "be":  be / am / is / are / being / was / were.
  2. Classify the extracted samples by whether the *SBN annotation* contains a
     "be" node.  In SBN a concept is written  lemma.pos.sense  (e.g. be.v.01),
     so a "be" node is any token of the form  be.<pos>.<sense>.  This matches
     the data-driven "be." heuristic observed on a small sample.

Outputs (written next to this script, under ./output/):
  be_samples.jsonl        all extracted samples + classification fields
  with_be_node.jsonl      samples whose SBN has a be node
  without_be_node.jsonl   samples whose SBN has no be node
  SUMMARY.md              human-readable counts / distributions / heuristic eval
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# --- configuration ----------------------------------------------------------

SRC = Path(
    "/Users/hongxuzhou/Documents/GitHub/lct_master_project/"
    "data/pmb-5.1.0/split/en/train/gold.sbn"
)
OUT_DIR = Path(__file__).resolve().parent / "output"

# Step 1: surface forms of "be" we extract on (whole-word, case-insensitive).
BE_FORMS = ["be", "am", "is", "are", "being", "was", "were"]
BE_FORM_RE = re.compile(r"\b(" + "|".join(BE_FORMS) + r")\b", re.IGNORECASE)

# Supplementary be-related surface material we *record* but do not gate on:
# the past participle "been" and the copula contractions 'm / 's / 're.
# (kept separate because 's is ambiguous with the possessive clitic.)
EXTRA_BE_RE = re.compile(r"\b(been)\b|'(m|s|re)\b", re.IGNORECASE)

# Step 2: an SBN "be" node = a synset token whose lemma is exactly "be".
BE_NODE_RE = re.compile(r"\bbe\.[a-z]\.[0-9]+\b")

ID_RE = re.compile(r"^p\d+/d\d+$")


# --- parsing ----------------------------------------------------------------

def parse_blocks(path: Path):
    """Yield (sample_id, sentence, sbn) triples from a gold.sbn file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    buf: list[str] = []
    for raw in lines:
        if raw.strip() == "":
            if buf:
                yield _emit(buf)
                buf = []
        else:
            buf.append(raw)
    if buf:
        yield _emit(buf)


def _emit(buf: list[str]):
    if len(buf) != 3:
        raise ValueError(f"Expected 3-line block, got {len(buf)}: {buf!r}")
    sample_id, sentence, sbn = buf[0].strip(), buf[1].strip(), buf[2].strip()
    if not ID_RE.match(sample_id):
        raise ValueError(f"Line 1 is not a valid ID: {sample_id!r}")
    return sample_id, sentence, sbn


# --- main -------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    records: list[dict] = []

    for sample_id, sentence, sbn in parse_blocks(SRC):
        total += 1

        be_forms = [m.lower() for m in BE_FORM_RE.findall(sentence)]
        if not be_forms:  # Step 1 filter
            continue

        be_node_senses = BE_NODE_RE.findall(sbn)
        extra = [m for grp in EXTRA_BE_RE.findall(sentence) for m in grp if m]

        records.append(
            {
                "id": sample_id,
                "sentence": sentence,
                "sbn": sbn,
                "be_forms": be_forms,                       # ordered, with dups
                "be_forms_set": sorted(set(be_forms)),
                "extra_be_surface": extra,                  # been / 'm / 's / 're
                "has_be_node": bool(be_node_senses),        # Step 2 classification
                "be_node_senses": be_node_senses,
            }
        )

    with_node = [r for r in records if r["has_be_node"]]
    without_node = [r for r in records if not r["has_be_node"]]

    _write_jsonl(OUT_DIR / "be_samples.jsonl", records)
    _write_jsonl(OUT_DIR / "with_be_node.jsonl", with_node)
    _write_jsonl(OUT_DIR / "without_be_node.jsonl", without_node)

    _write_summary(OUT_DIR / "SUMMARY.md", total, records, with_node, without_node)

    print(f"Parsed samples           : {total}")
    print(f"Samples with a be-form   : {len(records)}")
    print(f"  -> with SBN be node    : {len(with_node)}")
    print(f"  -> without SBN be node : {len(without_node)}")
    print(f"Output written to        : {OUT_DIR}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_summary(path, total, records, with_node, without_node) -> None:
    n = len(records)
    pct = lambda x: f"{(100 * x / n):.1f}%" if n else "n/a"

    # be-form -> (with_node, without_node) contingency, counting each sample once
    # per distinct form it contains.
    form_node = Counter()
    form_total = Counter()
    for r in records:
        for form in r["be_forms_set"]:
            form_total[form] += 1
            if r["has_be_node"]:
                form_node[form] += 1

    sense_dist = Counter()
    for r in with_node:
        sense_dist.update(r["be_node_senses"])

    lines = []
    lines.append("# `be` samples — extraction & be-node classification\n")
    lines.append(f"- Source: `{SRC}`")
    lines.append(f"- Total samples parsed: **{total}**")
    lines.append(
        f"- Samples containing a be-form ({'/'.join(BE_FORMS)}): "
        f"**{n}** ({(100*n/total):.1f}% of corpus)\n"
    )
    lines.append("## Step 2 — classification by SBN be-node\n")
    lines.append("| class | count | share of be-form samples |")
    lines.append("|---|---:|---:|")
    lines.append(f"| has `be.*` node | {len(with_node)} | {pct(len(with_node))} |")
    lines.append(f"| no be node | {len(without_node)} | {pct(len(without_node))} |\n")

    lines.append("## be-node sense distribution (within `has_be_node`)\n")
    lines.append("| sense | count |")
    lines.append("|---|---:|")
    for sense, c in sense_dist.most_common():
        lines.append(f"| `{sense}` | {c} |")
    lines.append("")

    lines.append("## Surface be-form vs. be-node presence\n")
    lines.append(
        "Per distinct surface form, how often samples containing it also carry "
        "an SBN be-node. High rates support the data-driven mapping.\n"
    )
    lines.append("| be-form | samples | with be-node | rate |")
    lines.append("|---|---:|---:|---:|")
    for form in BE_FORMS:
        t = form_total.get(form, 0)
        w = form_node.get(form, 0)
        rate = f"{(100*w/t):.1f}%" if t else "n/a"
        lines.append(f"| {form} | {t} | {w} | {rate} |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
