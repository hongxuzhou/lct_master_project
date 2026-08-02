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
# Roles whose value is a surface string rather than an index.
NAME_ROLES = {"Name", "Title"}

# Unambiguous correction markers only.  Reformulation markers ("that is",
# "or rather", "in fact") signal "let me put that another way", where both
# formulations stand and the second glosses the first -- the opposite of what
# CORRECTION asserts, and with a close pair they yield apposition ("We added
# something original, that is, new").  "well" is a hesitation as often as a
# correction, and "I mean to say" is nobody's idiom.  See DESIGN.md §1 step 5.
INTERREGNA = ["I mean", "no", "no wait", "sorry", "actually"]
# Function words that may be repeated inside the reparandum for free.
FREE_POS = {"DET", "ADP", "AUX", "PART"}


def _name_constants(concept) -> list[str]:
    """Surface strings a concept carries in its own constant-valued roles.

    PMB gives a proper name the synset of its *hypernym* and puts the name in a
    constant: `state.n.04 Name "Japan"`.  Lemma matching therefore looks for
    `state` in a sentence that only contains `Japan`, and gives up -- which is
    the dominant alignment failure class (FINDINGS.md §5.3).  The quoted string
    *is* the surface form, so it aligns exactly.
    """
    out = []
    for role, val in concept.roles:
        if role not in NAME_ROLES:
            continue
        v = val.strip('"')
        if v and v != "?":
            out.append(v)
    return out


def align(doc_sbn, spacy_doc):
    """Tie SBN concepts to surface tokens.

    Returns (lexical, named), both concept position -> spaCy token index.

    The two are kept apart because they feed different mutation operators.
    A concept carrying `Name "Japan"` is aligned to the token `Japan`, but its
    *synset* is `state.n.04` -- so substituting the synset would put the word
    "state" in the sentence, not another proper name.  Those sites belong to
    the constant operator (DESIGN.md §2); handing them to WordNet substitution
    produces "Adversary, I mean, Hitler assumed power in 1933."
    """
    used: set[int] = set()
    named: dict[int, int] = {}
    out: dict[int, int] = {}

    # Pass 1: concepts whose surface form is written into a constant.  Run
    # first so these tokens are claimed before lemma matching, which cannot
    # reach them at all and might otherwise mis-assign the same token.
    for c in doc_sbn.concepts:
        for name in _name_constants(c):
            head = name.split()[0].lower()
            for t in spacy_doc:
                if t.i in used or t.is_punct:
                    continue
                if t.text.lower() == head:
                    named[c.pos] = t.i
                    used.add(t.i)
                    break
            if c.pos in named:
                break

    # Pass 2: lexical concepts, by lemma + coarse POS.
    for c in doc_sbn.concepts:
        if c.pos_tag not in WN2UD or c.synset in NON_LEXICAL:
            continue
        # A concept that names itself with a constant is not substitutable
        # even when the constant could not be located in the sentence.
        if _name_constants(c):
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
    return out, named


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
        # The reparandum is now sentence-initial and takes the capital.  The
        # repair no longer is, so it must give it back -- unless the word is
        # capitalised for its own reasons.  Capitalising both produced
        # "The convention, I mean, The peace treaty will be signed tomorrow."
        reparandum = match_case(spacy_doc[0].text, reparandum)
        first = spacy_doc[0]
        if first.pos_ != "PROPN" and first.text != "I" and repair[:1].isupper():
            repair = repair[0].lower() + repair[1:]

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


def load_pool(path: str) -> dict[tuple[str, str], list[str]]:
    """Scored pool from `build_pool.py` -> site -> surviving candidates.

    Two filters are applied here, in the layer order of DESIGN.md §5: rows
    whose splice is illegal are dropped first, then rows the NLI lower bound
    calls synonyms.  What remains keeps the file's order, which is the pool
    builder's commonness ordering -- a provisional tie-break, not a judgement
    about which candidate is better (see `wn_candidates._commonness`).
    """
    from nli_filter import SYNONYMY_REJECT_ABOVE

    out: dict[tuple[str, str], list[str]] = {}
    kept = dropped_illegal = dropped_similar = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["legal"] != "1":
                dropped_illegal += 1
                continue
            syn = row.get("synonymy", "")
            if syn != "" and float(syn) > SYNONYMY_REJECT_ABOVE:
                dropped_similar += 1
                continue
            out.setdefault((row["doc_id"], row["concept_pos"]), []).append(
                row["reparandum_synset"])
            kept += 1
    print(f"pool {path}: {kept} candidates over {len(out)} sites "
          f"(dropped {dropped_illegal} illegal, {dropped_similar} too similar)")
    return out


def pick(doc_id: str, cpos: int, synset: str,
         pool: dict | None) -> str | None:
    """The reparandum to use at this site."""
    if pool is None:
        # No pool: fall back to the raw candidate list. This is unfiltered --
        # the first entry is merely the most ordinary word, which is how
        # near-synonyms used to reach the corpus. Use --pool for real output.
        cands = candidates(synset)
        return cands[0] if cands else None
    rows = pool.get((doc_id, str(cpos)))
    return rows[0] if rows else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--pool", default=None,
                    help="scored pool from build_pool.py; without it, "
                         "candidate selection is unfiltered")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-doc", type=int, default=1,
                    help="max repair samples per gold document")
    # Base (unmarked) repair is the layer that must be covered first: a marker
    # like "I mean" lifts detection to 100%, so a corpus that is mostly marked
    # is mostly teaching the easy case.  This default inverts the trial run's
    # 0.7.  It is a stopgap -- DESIGN.md §1 step 5 emits base and marked as
    # separate paired samples, which needs the schema change of §6.
    ap.add_argument("--interregnum-rate", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pool = load_pool(args.pool) if args.pool else None
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
        # `namemap` is where the constant operator will attach; the lexical
        # substitution below must not touch those sites.
        amap, _namemap = align(doc_sbn, sdoc)
        made = 0
        for cpos in sorted(amap, key=lambda p: rng.random()):
            if made >= args.per_doc:
                break
            c = doc_sbn.concepts[cpos]
            cand = pick(doc_sbn.doc_id, cpos, c.synset, pool)
            if cand is None:
                continue
            res = build_repair(doc_sbn, cpos, cand)
            if not res.ok:
                continue
            tok = sdoc[amap[cpos]]
            m = SYNSET_PATTERN.match(cand)
            surface, conf = inflect(m.group(1).replace("_", " "), tok.tag_,
                                    c.pos_tag)
            if not conf:
                stats["low_conf"] += 1
            inter = (rng.choice(INTERREGNA)
                     if rng.random() < args.interregnum_rate else None)
            nl = make_nl(sdoc, amap[cpos], surface, inter,
                         repeat_free=rng.random() < 0.5)
            sbn_rep = build_repair(doc_sbn, cpos, cand,
                                   interregnum=inter).sbn
            rows.append({
                "doc_id": doc_sbn.doc_id,
                "nl_clean": doc_sbn.sentence,
                "nl_repair": nl,
                "sbn_clean": doc_sbn.raw,
                "sbn_repair": sbn_rep,
                "repair_pos": c.pos_tag,
                "reparandum_synset": cand,
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
