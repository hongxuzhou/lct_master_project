#!/usr/bin/env python3
"""
Analyse what distinguishes be-form samples that receive an SBN `be` node from
those that do not, to surface the (unstated) annotation guideline behind it.

Reads be_samples.jsonl produced by extract_be_samples.py and cross-tabulates
structural features against `has_be_node`.

Working hypothesis (to be tested):
    A `be` node is introduced iff the copula has no *other* lexical predicate
    to carry the proposition:
      * complement is NOMINAL (identity / class membership / role) -> be node
      * complement is LOCATIVE or EXISTENTIAL (there is/are, where, from) -> be node
      * complement is a predicate ADJECTIVE -> NO be node (adj is the predicate)
      * `be` is an AUXILIARY (passive / progressive) -> NO be node (lexical V is)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
SAMPLES = OUT_DIR / "be_samples.jsonl"

# --- SBN-side feature regexes -----------------------------------------------
ADJ_SYNSET = re.compile(r"\b[a-z_]+\.a\.[0-9]+\b")          # predicate adjective
ADV_SYNSET = re.compile(r"\b[a-z_]+\.r\.[0-9]+\b")          # predicate adverb (away/back)
NONBE_VERB = re.compile(r"\b(?!be\.)[a-z_]+\.v\.[0-9]+\b")  # lexical verb (not be)
BE_SENSE = re.compile(r"\bbe\.[a-z]\.[0-9]+\b")
ROLE_WORDS = ["Co-Theme", "Role", "AttributeOf", "Attribute", "Location",
              "Source", "Result", "Agent", "Patient", "Experiencer",
              "Destination", "InstanceOf", "Goal"]

# --- sentence-side feature regexes ------------------------------------------
BE_FORM_TOKEN = r"(?:be|am|is|are|being|was|were)"
EXISTENTIAL = re.compile(r"\bthere\s+(?:is|are|was|were|'s|'re)\b", re.I)
WH_LOCATIVE = re.compile(r"\bwhere\s+(?:is|are|was|were)\b", re.I)
PROGRESSIVE = re.compile(rf"\b{BE_FORM_TOKEN}\b\s+(?:\w+\s+)?\w+ing\b", re.I)
PASSIVE_BY = re.compile(rf"\b{BE_FORM_TOKEN}\b\s+\w+(?:ed|en)\b.*\bby\b", re.I)


def features(r: dict) -> dict:
    sbn, sent = r["sbn"], r["sentence"]
    f = {
        "has_be_node": r["has_be_node"],
        "sbn_has_adj": bool(ADJ_SYNSET.search(sbn)),
        "sbn_has_adv_pred": bool(ADV_SYNSET.search(sbn)) and "AttributeOf" in sbn,
        "sbn_has_nonbe_verb": bool(NONBE_VERB.search(sbn)),
        "sbn_has_co_theme": "Co-Theme" in sbn,
        "sbn_has_attributeof": "AttributeOf" in sbn,
        "sbn_has_role": bool(re.search(r"\bRole\b", sbn)),
        "snt_existential": bool(EXISTENTIAL.search(sent)),
        "snt_wh_locative": bool(WH_LOCATIVE.search(sent)),
        "snt_progressive": bool(PROGRESSIVE.search(sent)),
        "snt_passive_by": bool(PASSIVE_BY.search(sent)),
    }
    return f


COMPARATIVE = re.compile(r"\b(?:more|less|as)\.r\.[0-9]+\b")


def derived_complement_type(f: dict, sbn: str) -> str:
    """Coarse construction label, predicted from features *other* than the
    be-node itself.  The guiding question: is there another word available to
    serve as the predicate?

      1. a lexical (non-be) verb is present  -> be is an AUXILIARY
      2. else an adjective synset is present -> predicate ADJECTIVE
                                                (comparative if more/less/as.r)
      3. else nothing else can predicate     -> NOMINAL / LOCATIVE / EXISTENTIAL
    """
    if f["sbn_has_nonbe_verb"]:
        return "auxiliary (lexical verb present)"
    if COMPARATIVE.search(sbn):
        # more/less/as.r heads the comparison; Co-Theme is the than-standard,
        # the predicate is still the adjective -> no be node expected.
        return "adjectival — comparative"
    if f["sbn_has_co_theme"]:
        # nominal / equative predicate (an attributive adj may still modify the
        # predicate noun, but the *head* of the predicate is nominal).
        return "nominal / equative (predicate noun)"
    if f["sbn_has_adj"]:
        return "adjectival — predicate adjective"
    if f["sbn_has_adv_pred"]:
        # predicate adverb: away / back / on_board (.r) carrying AttributeOf,
        # patterns exactly like a predicate adjective -> no be node.
        return "adverbial — predicate adverb"
    return "locative / existential / measure (no other predicate)"


def rate(num: int, den: int) -> str:
    return f"{(100*num/den):.1f}%" if den else "n/a"


def main() -> None:
    rows = [json.loads(l) for l in SAMPLES.open()]
    feats = [features(r) for r in rows]
    for r, f in zip(rows, feats):
        f["complement_type"] = derived_complement_type(f, r["sbn"])

    with_ = [f for f in feats if f["has_be_node"]]
    without = [f for f in feats if not f["has_be_node"]]
    nw, nwo = len(with_), len(without)

    L: list[str] = []
    L.append("# What predicts an SBN `be` node? — pattern analysis\n")
    L.append(f"- be-form samples analysed: **{len(feats)}** "
             f"(with be node: **{nw}**, without: **{nwo}**)\n")

    # ---- feature contingency -------------------------------------------------
    L.append("## Feature presence by class\n")
    L.append("Each cell = share of that class whose sample has the feature.\n")
    L.append("| feature | with be-node | without be-node |")
    L.append("|---|---:|---:|")
    feat_keys = [k for k in feats[0] if k.startswith(("sbn_", "snt_"))]
    for k in feat_keys:
        w = sum(f[k] for f in with_)
        wo = sum(f[k] for f in without)
        L.append(f"| `{k}` | {rate(w, nw)} | {rate(wo, nwo)} |")
    L.append("")

    # ---- complement-type breakdown ------------------------------------------
    L.append("## Coarse construction type vs. be-node\n")
    L.append("| construction type | n | with be-node | be-node rate |")
    L.append("|---|---:|---:|---:|")
    ctypes = Counter(f["complement_type"] for f in feats)
    for ct, n in ctypes.most_common():
        w = sum(1 for f in feats if f["complement_type"] == ct and f["has_be_node"])
        L.append(f"| {ct} | {n} | {w} | {rate(w, n)} |")
    L.append("")

    # ---- rule accuracy (confusion matrix) -----------------------------------
    # Rule: predict a be node iff the predicate is NOMINAL or LOCATIVE/
    # EXISTENTIAL/MEASURE, i.e. there is no other word to carry the predicate.
    def predict(f):
        return f["complement_type"].startswith(("nominal", "locative"))

    tp = sum(1 for f in feats if predict(f) and f["has_be_node"])
    fp = sum(1 for f in feats if predict(f) and not f["has_be_node"])
    fn = sum(1 for f in feats if not predict(f) and f["has_be_node"])
    tn = sum(1 for f in feats if not predict(f) and not f["has_be_node"])
    acc = (tp + tn) / len(feats)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    L.append("## Rule accuracy\n")
    L.append("**Rule:** a `be` node appears iff the predicate is nominal "
             "(predicate noun / identity) or locative / existential / measure "
             "— i.e. there is no adjective, adverb or lexical verb to carry the "
             "predicate.\n")
    L.append("| | actually has be-node | actually none |")
    L.append("|---|---:|---:|")
    L.append(f"| predicted be-node | {tp} | {fp} |")
    L.append(f"| predicted none | {fn} | {tn} |")
    L.append("")
    L.append(f"- accuracy: **{acc:.1%}**  ·  precision: **{prec:.1%}**  ·  "
             f"recall: **{rec:.1%}**\n")

    # ---- be-sense -> construction mapping (within with-node) ----------------
    L.append("## be-sense ↔ construction (within `has_be_node`)\n")
    L.append("| be sense | top construction types (count) |")
    L.append("|---|---|")
    by_sense: dict[str, Counter] = {}
    for r, f in zip(rows, feats):
        if not f["has_be_node"]:
            continue
        for s in set(BE_SENSE.findall(r["sbn"])):
            by_sense.setdefault(s, Counter())[f["complement_type"]] += 1
    for s in sorted(by_sense):
        top = ", ".join(f"{ct} ({c})" for ct, c in by_sense[s].most_common(3))
        L.append(f"| `{s}` | {top} |")
    L.append("")

    # ---- counter-examples: same-construction disagreements ------------------
    L.append("## Diagnostic residue\n")
    adj_with = [r for r, f in zip(rows, feats)
                if f["complement_type"] == "adjectival — predicate adjective"
                and f["has_be_node"]]
    nom_without = [r for r, f in zip(rows, feats)
                   if f["complement_type"].startswith(("nominal", "locative"))
                   and not f["has_be_node"]]
    L.append(f"- adjectival-predicate samples that *do* carry a be node: "
             f"**{len(adj_with)}** (expected ~0 under the hypothesis)")
    L.append(f"- nominal/locative/existential samples *without* a be node: "
             f"**{len(nom_without)}**\n")
    if adj_with:
        L.append("Examples (adjective complement yet be-node present):\n")
        for r in adj_with[:8]:
            L.append(f"- `[{r['id']}]` {r['sentence']}")
            L.append(f"    - `{r['sbn']}`")
        L.append("")

    report = OUT_DIR / "PATTERN_ANALYSIS.md"
    report.write_text("\n".join(L), encoding="utf-8")

    # console digest
    print(f"with be-node={nw}  without={nwo}")
    print("\nconstruction type -> be-node rate:")
    for ct, n in ctypes.most_common():
        w = sum(1 for f in feats if f["complement_type"] == ct and f["has_be_node"])
        print(f"  {ct:48s} n={n:4d}  be-node={rate(w,n)}")
    print(f"\nadj-complement WITH be-node : {len(adj_with)}")
    print(f"nominal WITHOUT be-node     : {len(nom_without)}")
    print(f"\nreport -> {report}")


if __name__ == "__main__":
    main()
