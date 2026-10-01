"""
WordNet lookup for the generation agent: a dictionary, nothing more.

    python3 wn_lookup.py senses <lemma> <pos>     # every sense of a word
    python3 wn_lookup.py check  <lemma.pos.NN>    # does this concept exist?

Names are written the PMB way, `<lemma>.<pos>.<NN>` = the NN-th WordNet 3.0
sense of that lemma. This is NOT WordNet's own synset name: PMB `time.n.08`
is WordNet's `fourth_dimension.n.01`, PMB `new.a.02` is `fresh.s.03` (1654 of
the 5932 synsets in train gold differ). Returning `synset.name()` would make
the agent write a different word, so every name printed here is rebuilt from
the queried lemma and its sense number.

Satellite adjectives (WordNet `s`) are part of `a`, as in PMB.

Only definitions and examples are shown. Hypernyms, synonyms and sister terms
are left out on purpose: they would hint at where to look for a reparandum,
which is the judgement the agent is being tested on.
"""
from __future__ import annotations

import sys

from nltk.corpus import wordnet as wn

POS = {"n", "v", "a", "r"}

# Concepts SBN uses as placeholders rather than for a content word. They exist
# in WordNet, but their WordNet definition says little about how SBN uses them.
PLACEHOLDERS = {
    "time.n.08": "SBN placeholder: the time of an event; carries tense (TPR/EQU/TSU now) and dates",
    "entity.n.01": "SBN placeholder: an unspecified referent ('it', 'they'), a plural built with Sub, or a self-repair anchor",
    "person.n.01": "SBN placeholder: a person; 'I'/'you' with EQU speaker/hearer",
    "male.n.02": "SBN placeholder: a male person ('he', a man's name)",
    "female.n.02": "SBN placeholder: a female person ('she', a woman's name)",
    "quantity.n.01": "SBN placeholder: an amount in a comparison ('over a thousand')",
    "location.n.01": "SBN placeholder: a place in a spatial relation ('under the table')",
    "event.n.01": "SBN placeholder: an unspecified event",
    "thing.n.12": "SBN placeholder: an unspecified thing",
}

# In PMB 5.1.0 train gold but not in WordNet 3.0 (all 8 such names).
_KEEP = "in PMB gold but not in WordNet 3.0: keep it where the gold SBN has it, do not introduce it"
PMB_ONLY = {
    "event.v.01": "PMB-only, not in WordNet: marks a fragment without a verb ('A room.'); keep it where the gold SBN has it, do not introduce it",
    **{n: _KEEP for n in (
        "location.v.01", "bmw.n.02", "opel.n.02", "polyacrylonitrile.n.01",
        "research_and_development.n.01", "scrooge_and_marley.n.01",
        "square_kilometer.n.01",
    )},
}


def _exact(lemma: str, pos: str, num: str):
    """The synset PMB means by lemma.pos.num, or None."""
    for p in ([pos, "s"] if pos == "a" else [pos]):
        try:
            return wn.synset(f"{lemma}.{p}.{num}")
        except Exception:  # noqa: BLE001 -- nltk raises WordNetError/ValueError
            pass
    return None


def _senses(lemma: str, pos: str):
    """WordNet senses of `lemma` in PMB order, after reducing inflection."""
    base = wn.morphy(lemma, pos) or lemma
    return base, wn.synsets(base, pos)


def cmd_senses(lemma: str, pos: str) -> int:
    lemma = lemma.lower().replace(" ", "_")
    if pos not in POS:
        print(f"pos must be one of n v a r, not {pos!r}")
        return 2
    base, syns = _senses(lemma, pos)
    if not syns:
        print(f"no {pos} senses for {lemma!r} in WordNet 3.0")
        return 1
    if base != lemma:
        print(f"({lemma} -> {base})")
    for k, s in enumerate(syns, 1):
        name = f"{base}.{pos}.{k:02d}"
        note = PLACEHOLDERS.get(name)
        line = f"{name}  {s.definition()}"
        if s.examples():
            line += f'  e.g. "{s.examples()[0]}"'
        if note:
            line += f"  [{note}]"
        print(line)
    return 0


def cmd_check(name: str) -> int:
    name = name.strip()
    if name in PMB_ONLY:
        print(f"{name}: {PMB_ONLY[name]}")
        return 1
    try:
        lemma, pos, num = name.rsplit(".", 2)
        k = int(num)
    except ValueError:
        print(f"{name}: not of the form lemma.pos.NN")
        return 2
    if pos == "s":
        print(f"{name}: write satellite adjectives with 'a': {lemma}.a.{num}")
        return 1
    if pos not in POS or len(num) != 2:
        print(f"{name}: pos must be n v a r and the sense two digits, e.g. {lemma}.n.01")
        return 2
    # wn.synset() looks the lemma up exactly; wn.synsets() would first reduce
    # inflection and accept `chased.v.01`, which SBN does not allow.
    syn = _exact(lemma, pos, num)
    if syn is None:
        base = wn.morphy(lemma, pos)
        if base and base != lemma:
            print(f"{name}: write the lemma, not an inflected form: {base}.{pos}.{num}")
            return 1
        n = len(wn.synsets(lemma, pos))
        if n:
            print(f"{name}: {lemma!r} has {n} {pos} senses, so .{num} does not exist")
        else:
            print(f"{name}: no {pos} senses for {lemma!r} in WordNet 3.0")
        return 1
    line = f"{name}: exists. {syn.definition()}"
    if name in PLACEHOLDERS:
        line += f"  [{PLACEHOLDERS[name]}]"
    print(line)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "senses":
        return cmd_senses(argv[1], argv[2])
    if len(argv) == 2 and argv[0] == "check":
        return cmd_check(argv[1])
    print(__doc__.strip().splitlines()[2])
    print(__doc__.strip().splitlines()[3])
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
