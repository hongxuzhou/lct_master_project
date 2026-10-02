"""
Pick the gold documents for one LLM-playground run, by repair type.

    python3 select_docs.py [--per-type 10] [--seed 13] [--out pinned_sample.tsv]
                           [--exclude earlier_sample.tsv ...]

For each repair type of prompt_docs/repair_notation.md §4, a document is
eligible when its gold SBN offers at least one site of that type (the
`site_*` functions below). Every eligible site position is written out, so a
`no_site` answer can later be checked against what was actually there.

Choices:
- no length filter: documents are drawn uniformly from the eligible ones, so
  lengths follow gold and any length preference of the model shows up;
- a document is used for one type only;
- excluded: gold documents shown as examples in the prompt documents, and
  gold documents whose own SBN is broken (the reader rejects it, or an index
  points outside the sequence);
- constant types draw at most one document per constant value, so that the
  name cell is not mostly "Tom" (1673 of 2188 male names in gold);
- `--exclude` leaves out documents of earlier samples, so a later round
  draws fresh ones;
- no interregnum: the model writes base repairs only, and editing
  expressions are inserted later by script (README §1, decided 2026-10-02).
  The first round's sample still has the column, from when the interregnum
  was assigned here (30% marked, as generate_repairs.py does it).
"""
from __future__ import annotations

import argparse
import collections
import csv
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "repair_data_gen"))
sys.path.insert(0, str(HERE))

from nltk.corpus import wordnet as wn  # noqa: E402
from sbn_lin import IDX_PATTERN, read_split  # noqa: E402
from validate_with_pmb_parser import check  # noqa: E402
from wn_lookup import PLACEHOLDERS, PMB_ONLY  # noqa: E402

GOLD = ROOT / "data/pmb-5.1.0/split/en/train/gold.sbn"

# Gold documents used as examples in prompt_docs/ (README §3).
EXAMPLE_DOCS = {
    "p62/d1762", "p37/d1861", "p35/d3246", "p81/d1985", "p70/d3477",
    "p18/d2190", "p98/d0867", "p01/d0850", "p49/d1783", "p30/d2997",
    "p48/d3446", "p00/d2270", "p37/d0872", "p96/d0785", "p71/d2819",
    "p30/d2724", "p72/d2409", "p80/d1457", "p68/d3216", "p74/d1927",
    "p04/d3042", "p76/d2001", "p47/d1417", "p14/d2976",
    "p18/d1394",   # example R (question slot), added 2026-10-02
}

TYPES = ("content_word", "constant_name", "constant_number",
         "constant_temporal", "tense", "edge_label", "forwarding")
TEMPORAL_ROLES = {"ClockTime", "DayOfWeek", "DayOfMonth", "MonthOfYear", "YearOfCentury"}
EDGE_LABEL_ROLES = {"Destination", "Source", "Location"}
SUBJECT_ROLES = {"Agent", "Theme", "Experiencer", "Patient", "Pivot"}
TENSE_OPS = {"TPR", "EQU", "TSU"}
NUMBER = re.compile(r"^\d+(\.\d+)?$")
# referents of "nobody", "everybody", "nothing", "everything"
QUANTIFIED = {"person.n.01", "entity.n.01", "thing.n.12"}
BE_FORMS = {"be", "is", "are", "am", "was", "were", "been", "being", "s", "re", "m",
            "isn", "aren", "wasn", "weren"}           # "isn't" -> isn, t


def _target(c, val: str) -> int | None:
    return c.pos + int(val) if IDX_PATTERN.match(val) else None


def _is_tense_time(c) -> bool:
    return c.synset == "time.n.08" and any(
        r in TENSE_OPS and v == "now" for r, v in c.roles)


def site_content_word(doc):
    return [(c.pos, c.synset) for c in doc.concepts
            if c.synset not in PLACEHOLDERS and c.synset not in PMB_ONLY]


def site_constant(doc, roles=None, value_re=None):
    out = []
    for c in doc.concepts:
        for r, v in c.roles:
            if roles is not None and r not in roles:
                continue
            if value_re is not None and not value_re.match(v):
                continue
            out.append((c.pos, f"{r} {v}"))
    return out


def site_name(doc):
    # `Name ?` marks the slot a wh-question asks about ("What country ..."),
    # not a name that could be swapped for another one.
    return site_constant(doc, roles={"Name"}, value_re=re.compile(r'^".*"$'))


def site_number(doc):
    """A plain number (`Quantity 3`, `EQU 100`), not a year or clock time."""
    return [(p, s) for p, s in site_constant(doc, value_re=NUMBER)
            if s.split()[0] not in TEMPORAL_ROLES]


def site_temporal(doc):
    return site_constant(doc, roles=TEMPORAL_ROLES)


def site_tense(doc):
    """A verb with a `Time` role to a tense time.n.08 right next to it, same box."""
    out = []
    cs = doc.concepts
    for c in cs:
        if c.pos_tag != "v":
            continue
        for r, v in c.roles:
            t = _target(c, v) if r == "Time" else None
            if t is not None and abs(t - c.pos) == 1 and 0 <= t < len(cs) \
                    and _is_tense_time(cs[t]) and cs[t].box == c.box:
                out.append((c.pos, c.synset))
    return out


def site_edge_label(doc):
    """A verb with an index-valued Destination/Source/Location role."""
    return [(c.pos, f"{c.synset} {r}") for c in doc.concepts
            if c.pos_tag == "v"
            for r, v in c.roles
            if r in EDGE_LABEL_ROLES and _target(c, v) is not None]


def _quantifier_restrictor(doc, x) -> bool:
    """x is the referent of a quantifier built with NEGATION ("nobody",
    "everybody": `NEGATION <1 person.n.01 …`): the first concept of a box that
    NEGATION opens, with no anchor saying who it is. Replacing it is
    quantifier replacement, which the notation cannot represent (KB, "Noted
    Limitations"). A pronoun before a negation ("I never …",
    `person.n.01 EQU speaker NEGATION <1 …`) is not affected.
    """
    if x.synset not in QUANTIFIED:
        return False                      # "She always …": a pronoun in the
                                          # scope of "always" stays eligible
    opened = any(s.name == "NEGATION" and s.new_box == x.box
                 and s.after_concept == x.pos for s in doc.separators)
    # any constant says who it is: EQU speaker ("I"), Sub speaker ("we"), Name
    anchored = any(_target(x, v) is None for _, v in x.roles)
    return opened and not anchored


def _said_as_often(doc, v) -> bool:
    """The verb is uttered at least as often as the SBN has it.

    Gold writes an elided verb out: "Neither Tom nor Mary goes to the store"
    has `go.v.01` in both clauses but "goes" once, so "Tom goes" was never
    said and cannot be restarted from (forwarding needs subject and verb on
    the surface). Inflection is undone with WordNet's morphy; a multi-word
    lemma (`pick_up`) is matched on its first word.
    """
    head = v.lemma.split("_")[0]
    words = [w.lower() for w in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", doc.sentence)]
    if head == "be":
        said = sum(w in BE_FORMS for w in words)      # "Tom's" -> tom, s
    else:
        # every base form, not only the first: "saw" is see or saw,
        # "found" find or found, "fell" fall or fell. The prefix test covers
        # forms WordNet cannot reduce ("chickened out", "speakest"); it can
        # only keep a site, never drop one.
        said = sum(head in ([w] + wn._morphy(w, wn.VERB))
                   or (len(head) >= 3 and w.startswith(head)) for w in words)
    written = sum(c.synset == v.synset for c in doc.concepts)
    return said >= written


def site_forwarding(doc):
    """A subject immediately followed by its verb, then the verb's tense time.n.08.

    The pattern of example Q: `X V <subject role> -1 … Time +1 time.n.08 …`,
    all three in one box. Pronoun subjects (I, we, he, she) count; quantifier
    subjects built with NEGATION do not, and neither does a verb the sentence
    elides (both decided 2026-10-02, after p99/d2900 and p31/d1485).
    """
    out = []
    cs = doc.concepts
    for i in range(len(cs) - 2):
        x, v, t = cs[i], cs[i + 1], cs[i + 2]
        if v.pos_tag != "v" or x.pos_tag != "n" or not (x.box == v.box == t.box):
            continue
        subj = any(r in SUBJECT_ROLES and _target(v, val) == i for r, val in v.roles)
        tense = _is_tense_time(t) and any(
            r == "Time" and _target(v, val) == i + 2 for r, val in v.roles)
        if subj and tense and not _quantifier_restrictor(doc, x) and _said_as_often(doc, v):
            out.append((i, x.synset))
    return out


SITES = {
    "content_word": site_content_word,
    "constant_name": site_name,
    "constant_number": site_number,
    "constant_temporal": site_temporal,
    "tense": site_tense,
    "edge_label": site_edge_label,
    "forwarding": site_forwarding,
}


def gold_is_sound(doc) -> bool:
    r = check(doc.raw)
    return r["parse"] == "ok" and r["dag"] == "ok" and r["index"] == "ok" \
        and r["magnitude"] == "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, default=HERE / "pinned_sample.tsv")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="earlier sample TSVs whose documents are not drawn again")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    excluded = {r["doc_id"] for p in args.exclude
                for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t")}
    docs = read_split(GOLD)
    unreadable = [d for d, _ in read_split.skipped]  # type: ignore[attr-defined]
    pool, broken = [], []
    for d in docs:
        if d.doc_id in EXAMPLE_DOCS or d.doc_id in excluded:
            continue
        # check() looks for CORRECTION/CONJUNCTION too; only the gold-level
        # checks matter here.
        (pool if gold_is_sound(d) else broken).append(d)

    eligible = {t: [] for t in TYPES}
    for d in pool:
        for t, fn in SITES.items():
            sites = fn(d)
            if sites:
                eligible[t].append((d, sites))

    print(f"gold docs {len(docs) + len(unreadable)}: unreadable by sbn_lin "
          f"{len(unreadable)}, used as prompt examples {len(EXAMPLE_DOCS)}, "
          f"in earlier samples {len(excluded)}, "
          f"broken gold {len(broken)} {[d.doc_id for d in broken]}, "
          f"pool {len(pool)}")
    print("eligible documents per type:")
    for t in TYPES:
        print(f"  {t:18s} {len(eligible[t]):5d}")

    # Rarest type first, so it is not starved by documents taken earlier.
    used: set[str] = set()
    rows = []
    for t in sorted(TYPES, key=lambda t: len(eligible[t])):
        cands = eligible[t][:]
        rng.shuffle(cands)
        seen_values: set[str] = set()
        picked = 0
        for d, sites in cands:
            if picked >= args.per_type:
                break
            if d.doc_id in used:
                continue
            if t.startswith("constant_"):
                vals = {s for _, s in sites}
                if vals <= seen_values:
                    continue
                seen_values |= vals
            used.add(d.doc_id)
            picked += 1
            rows.append({
                "doc_id": d.doc_id,
                "repair_type": t,
                "nl_clean": d.sentence,
                "sbn_clean": d.raw,
                "n_concepts": len(d.concepts),
                "n_tokens_nl": len(d.sentence.split()),
                "eligible_sites": "; ".join(f"{p}:{s}" for p, s in sites),
            })
        if picked < args.per_type:
            print(f"  only {picked} documents for {t}")

    rows.sort(key=lambda r: (TYPES.index(r["repair_type"]), r["doc_id"]))
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {args.out}")
    by = collections.Counter(r["repair_type"] for r in rows)
    print("  " + ", ".join(f"{t} {by[t]}" for t in TYPES))


if __name__ == "__main__":
    main()
