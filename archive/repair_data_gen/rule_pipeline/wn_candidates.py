"""Reparandum candidate pool: a contrast set for a gold concept.

A believable reparandum is a *sibling in a contrast set* -- the same slot, an
incompatible value (`banana_bread` / `cherry_pie`, `girl` / `boy`, `beat` /
`feed`).  Not a synonym: correcting a word into its synonym would be pedantic,
not a slip.  Not an arbitrary distant word either.

No single lexical resource supplies that relation for all three parts of
speech, so the pool is assembled differently for each (FINDINGS.md §6.5):

  noun       WordNet co-hyponyms.  The noun hierarchy is deep, so siblings land
             at a useful distance.
  verb       VerbNet class-mates, unioned with WordNet co-hyponyms.  WordNet
             alone fails in both directions here: verb groups are defined as
             near-synonymous senses of the same verb, and the verb hierarchy is
             too shallow for co-hyponyms to be sensible on their own
             (`write` -> `rhyme`).  VerbNet classes share an argument frame
             while differing in meaning, which is exactly the missing middle
             (`see` -> `hear`, `eat` -> `drink`).
  adjective  FrameNet frame-mates, unioned with WordNet antonyms.  An
             adjective's frame is an attribute dimension (`Size`, `Age`,
             `Chemical-sense_description`) and its adjective lexical units are
             the values on it: `spicy` -> sour / sweet / salty / bland.
             WordNet offers adjectives no contrast relation at all -- see
             `_adjective_contrast`.

The three are the same idea reached by three routes.  A contrast set is "one
slot, different values"; what differs is how the slot is defined -- by a
hierarchy for nouns, an argument frame for verbs, an attribute dimension for
adjectives.  None of the three is a *similarity* relation, which is why
similarity-ranked candidates were the wrong thing all along.

Two relations are rejected outright because WordNet itself defines them as
near-synonymy: **verb groups** and the **`similar_to` cluster** of the concept
being repaired.  They previously ranked first, and produced 23% of verb samples
and 45% of adjective samples.

This module deliberately does **not** decide which candidate is used.  Ordering
here is by how ordinary a word is, which is orthogonal to the reparandum/repair
distance that governs believability; that distance is the filter's business
(DESIGN.md §5.2).  Ranking by relation tightness and taking the top candidate
is what drove generation into the near-synonym end in the first place.

SBN-specific constraint throughout: a candidate must be writable as an SBN
synset id.  `SBNSpec.SYNSET_PATTERN` accepts only n|v|a|r|x, so WordNet
*satellite* adjectives (`.s.`) are excluded -- and PMB gold contains zero.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from nltk.corpus import wordnet as wn

from sbn_lin import SYNSET_PATTERN

WRITABLE_POS = {"n", "v", "a", "r"}

_VERBNET_HINT = (
    "VerbNet is required for verb reparandum candidates and is not installed. "
    "Run:  python3 -c \"import nltk; nltk.download('verbnet')\"\n"
    "Without it the verb pool collapses onto WordNet co-hyponyms, which "
    "FINDINGS.md §6.5 measures as unusable -- so this fails loudly rather "
    "than silently degrading the corpus."
)

_FRAMENET_HINT = (
    "FrameNet is required for adjective reparandum candidates and is not "
    "installed. Run:  python3 -c \"import nltk; nltk.download('framenet_v17')\"\n"
    "Without it the adjective pool falls back to antonyms alone: 40.1% "
    "coverage with a pool of 1, against 71.3% with a median pool of 6. "
    "That is a silent coverage loss, so this fails loudly instead."
)


def _writable(name: str) -> bool:
    m = SYNSET_PATTERN.match(name)
    return bool(m) and m.group(2) in WRITABLE_POS


def _lemma(name: str) -> str:
    return name.rsplit(".", 2)[0].lower()


# --------------------------------------------------------------------------
# VerbNet
# --------------------------------------------------------------------------
def _norm_key(k: str) -> str:
    """VerbNet writes `give%2:40:03`; NLTK's lemma table wants `give%2:40:03::`."""
    return k if k.endswith("::") else k + "::"


@lru_cache(maxsize=1)
def _verbnet_index() -> tuple[dict, dict]:
    """(sense key -> {class id}, class id -> {sense key}).

    Classes are collapsed to their top two levels (`give-13.1-1` -> `give-13.1`)
    so that a singleton subclass inherits its siblings; `eat-39.1-1` on its own
    has one member and would yield no candidates.
    """
    try:
        from nltk.corpus import verbnet as vn
        vn.classids()
    except Exception as exc:                          # not downloaded
        raise RuntimeError(_VERBNET_HINT) from exc

    key2cls: dict[str, set] = defaultdict(set)
    cls2keys: dict[str, set] = defaultdict(set)
    for cid in vn.classids():
        parts = cid.split("-")
        top = "-".join(parts[:2])
        for member in vn.vnclass(cid).findall("MEMBERS/MEMBER"):
            for key in (member.attrib.get("wn") or "").split():
                key = _norm_key(key)
                key2cls[key].add(top)
                cls2keys[top].add(key)
    return key2cls, cls2keys


def _verbnet_mates(ss) -> set:
    key2cls, cls2keys = _verbnet_index()
    classes: set[str] = set()
    for lemma in ss.lemmas():
        try:
            classes |= key2cls.get(lemma.key(), set())
        except Exception:
            continue
    out = set()
    for cid in classes:
        for key in cls2keys[cid]:
            try:
                mate = wn.lemma_from_key(key).synset()
            except Exception:
                continue
            if mate != ss and mate.pos() == "v":
                out.add(mate)
    return out


# --------------------------------------------------------------------------
# WordNet relations
# --------------------------------------------------------------------------
def _cohyponyms(ss) -> set:
    out = set()
    for hyper in ss.hypernyms():
        out |= set(hyper.hyponyms())
    out.discard(ss)
    return out


@lru_cache(maxsize=1)
def _framenet_index() -> tuple[dict, dict]:
    """(adjective lemma -> {frame}, frame -> {adjective lemma})."""
    try:
        from nltk.corpus import framenet as fnet
        fnet.frames()
    except Exception as exc:
        raise RuntimeError(_FRAMENET_HINT) from exc

    lem2frames: dict[str, set] = defaultdict(set)
    frame2adj: dict[str, set] = defaultdict(set)
    for lu in fnet.lus():
        if not lu.name.endswith(".a"):
            continue
        lem = lu.name[:-2].lower()
        lem2frames[lem].add(lu.frame.name)
        frame2adj[lu.frame.name].add(lem)
    return lem2frames, frame2adj


def _frame_mates(ss) -> set:
    """Adjectives sharing a FrameNet frame with this one.

    An adjective's frame *is* an attribute dimension -- `Size`, `Age`,
    `Temperature`, `Chemical-sense_description` -- and its adjective lexical
    units are the values on that dimension.  That is exactly the contrast set
    this module is after: `spicy` -> sour / sweet / salty / bland, `happy` ->
    angry / dejected / elated.

    Antonymy, by comparison, returns only the two poles of such a dimension.
    """
    lem2frames, frame2adj = _framenet_index()
    lem = _lemma(ss.name()).replace("_", " ")
    mates: set[str] = set()
    for frame in lem2frames.get(lem, ()):
        mates |= frame2adj[frame]
    mates.discard(lem)

    out = set()
    for m in mates:
        # FrameNet gives a lemma; SBN needs a synset.  Take the first head
        # (`.a.`) sense.  This is an approximation -- PMB's annotators chose
        # senses by hand -- and its error rate has not been measured.
        for s in wn.synsets(m.replace(" ", "_"), pos="a"):
            if s.pos() == "a":
                out.add(s)
                break
    out.discard(ss)
    return out


def _antonyms(ss) -> set:
    out = set()
    for lemma in ss.lemmas():
        for anto in lemma.antonyms():
            if anto.synset().pos() == "a":
                out.add(anto.synset())
    out.discard(ss)
    return out


def _adjective_contrast(ss) -> set:
    """FrameNet frame-mates, unioned with antonyms.

    WordNet alone cannot do adjectives.  SBN writes only WordNet *head*
    adjectives (`.a.`) -- `SYNSET_PATTERN` rejects satellites (`.s.`), and gold
    contains zero of them -- and heads relate to heads by just two relations,
    neither of which yields a contrast set:

      similar_to  returns satellites only; measured as 0 writable candidates
                  for every adjective tested.
      also_see    returns writable heads, but near-synonyms (`new` -> fresh /
                  modern / current), the relation this module rejects.

    That leaves antonymy, which covers 40.1% of gold adjectives with a pool of
    exactly 1.  FrameNet raises this to 71.3% with a median pool of 6, because
    it supplies the whole dimension rather than its endpoints.

    A residual limitation: frame-mates are alternatives on *one* dimension.
    A cross-dimension repair -- "the curry is too greasy, spicy", swapping
    texture for taste -- is not reachable this way, and would need contextual
    generation over the modified noun instead.
    """
    return _frame_mates(ss) | _antonyms(ss)


# --------------------------------------------------------------------------
# pool
# --------------------------------------------------------------------------
def _pool(ss) -> set:
    pos = ss.pos()
    if pos == "v":
        return _verbnet_mates(ss) | _cohyponyms(ss)
    if pos in ("a", "s"):
        return _adjective_contrast(ss)
    return _cohyponyms(ss)


def _commonness(s) -> tuple:
    """Ordering key: prefer ordinary, everyday words.

    Deliberately contains no distance term.  Speakers slip into common words,
    not obscure ones, so this is a real property of a believable reparandum --
    and it is orthogonal to the reparandum/repair distance, which the filter
    decides.  `name` breaks ties so the order is deterministic.
    """
    name = s.name()
    return (
        "_" in _lemma(name),          # multiword lemmas last
        int(name.rsplit(".", 1)[1]) if name.rsplit(".", 1)[1].isdigit() else 99,
        -len(s.lemmas()),             # more lemmas = more common concept
        name,
    )


@lru_cache(maxsize=None)
def candidates(synset_id: str, limit: int = 20) -> tuple[str, ...]:
    """Reparandum pool for a gold synset id.

    The order is a provisional convenience, not a selection: callers that need
    one candidate should take it from the filter, not from position 0.
    """
    m = SYNSET_PATTERN.match(synset_id)
    if not m:
        return ()
    lemma, pos, num = m.groups()
    try:
        ss = wn.synset(f"{lemma}.{pos}.{num}")
    except Exception:
        return ()

    # Every surface form the repair can be written as, not just the one in the
    # synset id.  `visualize.v.01` lists `see` among its lemmas, so it and
    # `see.v.01` can surface identically even though the ids differ -- the same
    # unlearnable sample as a literal same-lemma pair.
    ref_lemmas = {l.name().lower() for l in ss.lemmas()}

    keep = []
    for s in _pool(ss):
        name = s.name()
        if s == ss or not _writable(name):
            continue
        # Any shared lemma means the two synsets have a surface form in common,
        # so the reparandum and repair could be written the same way
        # ("He ran, actually, ran five miles").
        if {l.name().lower() for l in s.lemmas()} & ref_lemmas:
            continue
        keep.append(s)
    keep.sort(key=_commonness)
    return tuple(s.name() for s in keep[:limit])


@lru_cache(maxsize=None)
def raw_pool_size(synset_id: str) -> tuple[int, int]:
    """(candidates before the writability filter, after) -- for diagnostics."""
    m = SYNSET_PATTERN.match(synset_id)
    if not m:
        return (0, 0)
    lemma, pos, num = m.groups()
    try:
        ss = wn.synset(f"{lemma}.{pos}.{num}")
    except Exception:
        return (0, 0)
    raw = {s.name() for s in _pool(ss)}
    raw.discard(ss.name())
    return (len(raw), sum(1 for n in raw if _writable(n)))
