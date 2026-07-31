"""
End-to-end generator: gold (nl, sbn) -> (nl_repair, sbn_repair, sbn_clean).

Pipeline per gold document:
  1. spaCy-tag the sentence; align each lexical SBN concept to a token by
     lemma + coarse POS.
  2. pick a WordNet neighbour of the aligned concept as the reparandum, and
     inflect it to the token's PTB tag.
  3. splice CORRECTION/CONJUNCTION into the SBN (repair_transform).
  4. splice the reparandum + interregnum into the sentence.

Reparandum span.  We keep the *content* part of the reparandum to exactly one
word, because only that word has a concept in the graph.  Function words
immediately in front of it (determiner, preposition, auxiliary, possessive)
are repeated for free -- they carry no SBN concept, so LARD's "degree 2/3"
replacements stay expressible.  A degree>1 replacement over *content* words is
not expressible by a single-concept splice and is not generated.

Run:
    python3 generate_repairs.py --out repairs_train.tsv [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import spacy

from inflect_en import inflect, match_case
from repair_transform import build_repair
from sbn_lin import SYNSET_PATTERN, read_split
from wn_candidates import candidates

ROOT = Path(__file__).resolve().parents[1]

# Structural scaffolding, not an uttered content word.
NON_LEXICAL = {
    "time.n.08", "entity.n.01", "person.n.01", "male.n.02", "female.n.02",
    "quantity.n.01", "location.n.01", "event.n.01", "thing.n.12",
}
WN2UD = {"n": "NOUN", "v": "VERB", "a": "ADJ", "r": "ADV"}

INTERREGNA = [
    "I mean", "well", "actually", "no", "in fact", "sorry", "or rather",
    "I mean to say", "no wait", "that is",
]
# Function words that may be repeated inside the reparandum for free.
FREE_POS = {"DET", "ADP", "AUX", "PART"}


def align(doc_sbn, spacy_doc):
    """concept position -> spaCy token index."""
    used: set[int] = set()
    out: dict[int, int] = {}
    for c in doc_sbn.concepts:
        if c.pos_tag not in WN2UD or c.synset in NON_LEXICAL:
            continue
        m = SYNSET_PATTERN.match(c.synset)
        lemma = m.group(1).replace("_", " ").lower()
        head = lemma.split()[0]
        want_pos = WN2UD[c.pos_tag]
        best = None
        for t in spacy_doc:
            if t.i in used or t.is_punct:
                continue
            same_lemma = t.lemma_.lower() == head or t.text.lower() == head
            if not same_lemma:
                continue
            score = (t.pos_ == want_pos) or (want_pos == "ADJ" and t.pos_ in ("ADJ", "VERB"))
            if best is None or (score and not best[1]):
                best = (t.i, bool(score))
            if score:
                break
        if best is not None:
            out[c.pos] = best[0]
            used.add(best[0])
    return out


def reparandum_span(spacy_doc, tok_i: int, max_free: int = 2) -> int:
    """Start index of the repeated span: tok_i, extended left over free words.

    Clitics ("'s", "n't") are excluded: they are attached to the token before
    them, so repeating them alone produces "Let 's sit down, 's sit ...".
    """
    start = tok_i
    for _ in range(max_free):
        p = start - 1
        if p < 0:
            break
        t = spacy_doc[p]
        if t.pos_ not in FREE_POS or t.text.startswith("'") or t.text == "n't":
            break
        if p > 0 and not spacy_doc[p - 1].whitespace_:
            break                       # glued to the previous token
        start = p
    return start


def make_nl(spacy_doc, tok_i: int, new_word: str, interregnum: str | None,
            repeat_free: bool) -> str:
    """Splice the reparandum in, preserving the original whitespace."""
    tok = spacy_doc[tok_i]
    start = reparandum_span(spacy_doc, tok_i) if repeat_free else tok_i

    prefix = "".join(t.text_with_ws for t in spacy_doc[:start])
    repeated = "".join(t.text_with_ws for t in spacy_doc[start:tok_i])
    repair = "".join(t.text_with_ws for t in spacy_doc[start:tok_i]) + tok.text
    tail = "".join(t.text_with_ws for t in spacy_doc[tok_i + 1:])

    reparandum = repeated + match_case(tok.text, new_word)
    if start == 0:
        reparandum = match_case(spacy_doc[0].text, reparandum)
        repair = match_case(spacy_doc[0].text, repair)

    mid = f", {interregnum}, " if interregnum else ", "
    out = prefix + _fix_article(reparandum) + mid + repair
    if tail:
        out += tok.whitespace_ + tail if tok.whitespace_ else tail
    return _fix_article(out.strip())


_ARTICLE = re.compile(r"\b([Aa]n?)(\s+)([A-Za-z])")


def _fix_article(text: str) -> str:
    """Re-agree a/an: the substituted reparandum may change the initial sound."""
    def sub(m):
        art, ws, first = m.groups()
        want = "an" if first.lower() in "aeiou" else "a"
        if art[0].isupper():
            want = want.capitalize()
        return want + ws + first
    return _ARTICLE.sub(sub, text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-doc", type=int, default=1,
                    help="max repair samples per gold document")
    ap.add_argument("--interregnum-rate", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gold = ROOT / f"data/pmb-5.1.0/split/en/{args.split}/gold.sbn"
    docs = read_split(gold)
    if args.limit:
        docs = docs[: args.limit]
    nlp = spacy.load("en_core_web_sm")

    rows = []
    stats = {"docs": 0, "no_site": 0, "samples": 0, "low_conf": 0}
    texts = [d.sentence for d in docs]
    for doc_sbn, sdoc in zip(docs, nlp.pipe(texts, batch_size=256)):
        stats["docs"] += 1
        amap = align(doc_sbn, sdoc)
        made = 0
        for cpos in sorted(amap, key=lambda p: rng.random()):
            if made >= args.per_doc:
                break
            c = doc_sbn.concepts[cpos]
            cands = candidates(c.synset)
            if not cands:
                continue
            res = build_repair(doc_sbn, cpos, cands[0])
            if not res.ok:
                continue
            tok = sdoc[amap[cpos]]
            m = SYNSET_PATTERN.match(cands[0])
            surface, conf = inflect(m.group(1).replace("_", " "), tok.tag_,
                                    c.pos_tag)
            if not conf:
                stats["low_conf"] += 1
            inter = (rng.choice(INTERREGNA)
                     if rng.random() < args.interregnum_rate else None)
            nl = make_nl(sdoc, amap[cpos], surface, inter,
                         repeat_free=rng.random() < 0.5)
            sbn_rep = build_repair(doc_sbn, cpos, cands[0],
                                   interregnum=inter).sbn
            rows.append({
                "doc_id": doc_sbn.doc_id,
                "nl_clean": doc_sbn.sentence,
                "nl_repair": nl,
                "sbn_clean": doc_sbn.raw,
                "sbn_repair": sbn_rep,
                "repair_pos": c.pos_tag,
                "reparandum_synset": cands[0],
                "repair_synset": c.synset,
                "reparandum_surface": surface,
                "interregnum": inter or "",
                "strategy": res.strategy.value,
                "devices": "|".join(d.value for d in res.devices),
                "dropped_roles": "|".join(res.dropped_roles),
                "max_abs_index": res.max_abs_index,
                "separator_in_tail": int(res.separator_in_tail),
                "inflection_confident": int(conf),
            })
            made += 1
            stats["samples"] += 1
        if made == 0:
            stats["no_site"] += 1

    print(f"docs {stats['docs']}, samples {stats['samples']}, "
          f"docs with no usable site {stats['no_site']} "
          f"({stats['no_site']/stats['docs']:.1%}), "
          f"low-confidence inflections {stats['low_conf']}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out}")
    else:
        for r in rows[:15]:
            print()
            print(f"  {r['doc_id']}  [{r['repair_pos']}] "
                  f"{r['reparandum_synset']} -> {r['repair_synset']}  "
                  f"({r['strategy']}, {r['devices']})")
            print(f"    clean : {r['nl_clean']}")
            print(f"    repair: {r['nl_repair']}")
            print(f"    sbn   : {r['sbn_repair']}")


if __name__ == "__main__":
    main()
