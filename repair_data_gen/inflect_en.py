"""
Minimal English surface realiser: lemma + PTB tag -> inflected form.

The reparandum is a WordNet synset, i.e. a *lemma*; the word it replaces in the
sentence is inflected ("hunts", "coughed", "strangers").  Producing "The cat
chase, actually, hunts the rat" would put a disfluency in the wrong place, so
the reparandum has to carry the same inflection as the repair.

Irregular forms come from WordNet's own exception lists (the `.exc` files
NLTK exposes as `wn._exception_map`), inverted.  Everything else is regular
orthography.  `confident=False` marks forms we guessed.
"""
from __future__ import annotations

import re
from functools import lru_cache

from nltk.corpus import wordnet as wn

VOWELS = "aeiou"


@lru_cache(maxsize=1)
def _inverse_exceptions() -> dict[tuple[str, str], list[str]]:
    wn.ensure_loaded()
    out: dict[tuple[str, str], list[str]] = {}
    for pos, table in wn._exception_map.items():
        for inflected, bases in table.items():
            for base in bases:
                out.setdefault((base, pos), []).append(inflected)
    return out


def _es_form(w: str) -> str:
    if re.search(r"(s|sh|ch|x|z|o)$", w):
        return w + "es"
    if re.search(r"[^aeiou]y$", w):
        return w[:-1] + "ies"
    return w + "s"


def _ed_form(w: str) -> str:
    if w.endswith("e"):
        return w + "d"
    if re.search(r"[^aeiou]y$", w):
        return w[:-1] + "ied"
    if re.search(r"[^aeiou][aeiou][^aeiouwxy]$", w) and len(w) <= 5:
        return w + w[-1] + "ed"
    return w + "ed"


def _ing_form(w: str) -> str:
    if w.endswith("ie"):
        return w[:-2] + "ying"
    if w.endswith("e") and not w.endswith("ee"):
        return w[:-1] + "ing"
    if re.search(r"[^aeiou][aeiou][^aeiouwxy]$", w) and len(w) <= 5:
        return w + w[-1] + "ing"
    return w + "ing"


IRREGULAR_CORE = {
    ("be", "VBZ"): "is", ("be", "VBP"): "are", ("be", "VBD"): "was",
    ("be", "VBN"): "been", ("be", "VBG"): "being",
    ("have", "VBZ"): "has", ("have", "VBD"): "had", ("have", "VBN"): "had",
    ("do", "VBZ"): "does", ("do", "VBD"): "did", ("do", "VBN"): "done",
}


def inflect(lemma: str, tag: str, wn_pos: str | None = None) -> tuple[str, bool]:
    """(surface form, confident?) for a multiword-safe lemma and a PTB tag.

    `wn_pos` is the WordNet part of speech of the *concept*.  It overrides the
    token's PTB tag for adjectives and adverbs: PMB aligns `addicted.a.01` to a
    token spaCy tags VBN, and inflecting the already-inflected adjective
    produces "unaddicteded".
    """
    if wn_pos in ("a", "r"):
        return lemma.lower(), True
    parts = lemma.split()
    if len(parts) > 1:                      # inflect the head, keep the rest
        head, conf = inflect(parts[0], tag, wn_pos)
        return " ".join([head] + parts[1:]), conf
    w = lemma.lower()
    if (w, tag) in IRREGULAR_CORE:
        return IRREGULAR_CORE[(w, tag)], True

    exc = _inverse_exceptions()
    if tag in ("VBD", "VBN", "VBZ", "VBG", "VBP", "VB"):
        forms = exc.get((w, "v"), [])
        if tag == "VB" or tag == "VBP":
            return w, True
        if tag == "VBZ":
            return _es_form(w), True
        if tag == "VBG":
            ing = [f for f in forms if f.endswith("ing")]
            return (ing[0], True) if ing else (_ing_form(w), True)
        # VBD / VBN
        cands = [f for f in forms if not f.endswith("ing")]
        if not cands:
            return _ed_form(w), True
        if len(cands) == 1:
            return cands[0], True
        # e.g. go -> [gone, went]: -n/-en is the participle
        part = [f for f in cands if f.endswith(("n", "ne"))]
        rest = [f for f in cands if f not in part]
        if tag == "VBN":
            return (part[0], True) if part else (cands[0], False)
        return (rest[0], True) if rest else (cands[0], False)

    if tag in ("NNS", "NNPS"):
        forms = exc.get((w, "n"), [])
        return (forms[0], True) if forms else (_es_form(w), True)
    if tag in ("JJR", "RBR"):
        return (w + "r" if w.endswith("e") else w + "er"), False
    if tag in ("JJS", "RBS"):
        return (w + "st" if w.endswith("e") else w + "est"), False
    return w, True


def match_case(model: str, form: str) -> str:
    if model[:1].isupper():
        return form[:1].upper() + form[1:]
    return form


def indefinite_article(word: str) -> str:
    return "an" if word[:1].lower() in VOWELS else "a"
