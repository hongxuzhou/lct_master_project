"""
Reparandum candidate generation from WordNet, with an SBN-writability filter.

LARD picks the reparandum as a co-hyponym of the repair word (same hypernym).
Two SBN-specific constraints are added here:

  * the candidate must be writable as an SBN synset id: SBNSpec.SYNSET_PATTERN
    accepts only n|v|a|r|x, so WordNet *satellite* adjectives (`.s.`) are
    excluded outright -- and PMB gold contains zero of them;
  * adjectives have no hypernyms at all in WordNet, so the co-hyponym recipe
    silently returns nothing for them.  We fall back to
    similar_to / also_see / antonym, then re-apply the writability filter.
"""
from __future__ import annotations

from functools import lru_cache

from nltk.corpus import wordnet as wn

from sbn_lin import SYNSET_PATTERN

WRITABLE_POS = {"n", "v", "a", "r"}


def _writable(name: str) -> bool:
    m = SYNSET_PATTERN.match(name)
    return bool(m) and m.group(2) in WRITABLE_POS


def _plausibility(s, ref, rank: int) -> tuple:
    """Sort key: a good reparandum is a *common, single-word* near-miss.

    LARD re-ranks candidates with a sentence encoder.  Without one, these three
    cheap WordNet signals remove most of the damage (particle verbs like
    `grind_out.v.01` standing in for `bear.v.02`, obscure senses, and
    multiword terms that no speaker would half-utter):
      * tight relations (verb group, similarity cluster) before the loose
        co-hyponym pool -- WordNet verb troponymy is shallow, so co-hyponyms
        of a verb are often absurdly distant,
      * single-word lemmas first,
      * low sense number  (sense 1 is the frequent reading),
      * more lemmas in the synset = more common concept.
    """
    name = s.name()
    lemma = name.rsplit(".", 2)[0]
    multiword = "_" in lemma
    try:
        sense_no = int(name.rsplit(".", 1)[1])
    except ValueError:
        sense_no = 99
    try:
        sim = s.path_similarity(ref) or 0.0
    except Exception:
        sim = 0.0
    return (multiword, rank, sense_no > 3, round(-sim, 3), sense_no,
            -len(s.lemmas()), name)


@lru_cache(maxsize=None)
def candidates(synset_id: str, limit: int = 10) -> tuple[str, ...]:
    """Reparandum pool for a gold synset id, ordered best-first."""
    m = SYNSET_PATTERN.match(synset_id)
    if not m:
        return ()
    lemma, pos, num = m.groups()
    try:
        ss = wn.synset(f"{lemma}.{pos}.{num}")
    except Exception:
        return ()

    seen: dict[str, tuple] = {}

    def add(s, rank: int) -> None:
        n = s.name()
        if s != ss and _writable(n) and n not in seen:
            seen[n] = (s, rank)

    # rank 0: tight relations.  verb groups are near-synonymous senses of the
    # same verb; the adjective similarity cluster plays the same role.
    if pos == "v":
        for g in ss.verb_groups():
            add(g, 0)
    if pos == "a":
        for sim in ss.similar_tos():
            add(sim, 0)
            for sib in sim.similar_tos():
                add(sib, 0)
        for also in ss.also_sees():
            add(also, 0)
        for l in ss.lemmas():
            for a in l.antonyms():
                add(a.synset(), 1)
    # rank 2: co-hyponyms (LARD's relation).  Only nouns and verbs have
    # hypernyms at all, so this is empty for adjectives.
    for hyper in ss.hypernyms():
        for hypo in hyper.hyponyms():
            add(hypo, 2)
    ranked = sorted(seen.values(), key=lambda sr: _plausibility(sr[0], ss, sr[1]))
    return tuple(s.name() for s, _ in ranked[:limit])


@lru_cache(maxsize=None)
def raw_pool_size(synset_id: str) -> tuple[int, int]:
    """(candidates before writability filter, after) -- for diagnostics."""
    m = SYNSET_PATTERN.match(synset_id)
    if not m:
        return (0, 0)
    lemma, pos, num = m.groups()
    try:
        ss = wn.synset(f"{lemma}.{pos}.{num}")
    except Exception:
        return (0, 0)
    raw: set[str] = set()
    for hyper in ss.hypernyms():
        for hypo in hyper.hyponyms():
            raw.add(hypo.name())
    if pos == "v":
        raw |= {g.name() for g in ss.verb_groups()}
    if pos == "a":
        for sim in ss.similar_tos():
            raw.add(sim.name())
            raw |= {s.name() for s in sim.similar_tos()}
        raw |= {a.name() for a in ss.also_sees()}
        for l in ss.lemmas():
            raw |= {a.synset().name() for a in l.antonyms()}
    raw.discard(ss.name())
    return (len(raw), sum(1 for n in raw if _writable(n)))
