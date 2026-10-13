"""
Pick the gold documents for one LLM-playground run, by repair type.

    python3 select_docs.py [--per-type 10] [--seed 13] [--out samples/pinned_sample.tsv]
                           [--exclude earlier_sample.tsv ...] [--exclude-type type ...]
                           [--quota type=n ...] [--cap type=n ...]
                           [--edge-sites direction|place]

For each repair type of prompts/base/repair_notation.md §4, a document is
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
- constant types draw at most `--cap` documents per constant value (default
  1), so that the name cell is not mostly "Tom" (1673 of 2188 male names in
  gold); 0 means no cap. One per value was right for 10-item samples; the
  full run uses name 3, number and temporal no cap (decided 2026-10-04);
- `--quota` sets the number per type, overriding `--per-type`;
- `--exclude` leaves out documents of earlier samples, so a later round
  draws fresh ones; with `--exclude-type`, only those of the given types,
  and a document used once for another type can be drawn again (at most two
  items per source document; decided 2026-10-10 for places);
- `--edge-sites place` draws edge-label documents whose only sites are
  places (§4.4 third kind, prompts/edge_r4); the default `direction` is
  the rule of the earlier runs and draws the same as before;
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

# Gold documents used as examples in prompts/base/ (README §3).
EXAMPLE_DOCS = {
    "p62/d1762", "p37/d1861", "p35/d3246", "p81/d1985", "p70/d3477",
    "p18/d2190", "p98/d0867", "p01/d0850", "p49/d1783", "p30/d2997",
    "p48/d3446", "p00/d2270", "p37/d0872", "p96/d0785", "p71/d2819",
    "p30/d2724", "p72/d2409", "p80/d1457", "p68/d3216", "p74/d1927",
    "p04/d3042", "p76/d2001", "p47/d1417", "p14/d2976",
    "p18/d1394",   # example R (question slot), added 2026-10-02
    "p61/d0863", "p16/d2891", "p91/d3375",  # examples S-U (place), prompts/edge_r4, 2026-10-10
}

TYPES = ("content_word", "constant_name", "constant_number",
         "constant_temporal", "tense", "edge_label", "forwarding")
TEMPORAL_ROLES = {"ClockTime", "DayOfWeek", "DayOfMonth", "MonthOfYear", "YearOfCentury"}
EDGE_LABEL_ROLES = {"Destination", "Source", "Location"}
# Edge label sites (decided 2026-10-04, after the round-2 audit): a gold
# Source or Destination, where swapping direction is a change of intent
# ("taken from, to the orphanage"), or the Location of be.v.03 with a person
# subject ("She's from, in Boston"; narrowed 2026-10-05 after round 3).
# Location on other verbs gave "living from, in Tokyo".
EDGE_LABEL_SITE_ROLES = {"Source", "Destination"}
# The preposition that must be in the sentence for each gold role.
EDGE_PREPS = {
    "Source": {"from", "off", "out"},
    # directional only: PMB also writes Destination for placement ("spat on
    # the ground", "planting in his backyard"), where "from" reads wrong
    "Destination": {"to", "into", "onto", "toward", "towards", "for"},
    "Location": {"in", "on", "at", "inside"},
}
_UNITS = ("zero one two three four five six seven eight nine ten eleven twelve "
          "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth",
        "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
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


def _has_wh(c) -> bool:
    """The concept is the slot a question asks about (`EQU ?`, `Name ?`, …)."""
    return any(v == "?" for _, v in c.roles)


def _words(doc) -> list[str]:
    # "thirty-four" and "1,500" stay one token
    return [w.lower() for w in re.findall(r"[A-Za-z0-9]+(?:[-,.][A-Za-z0-9]+)*", doc.sentence)]


def _number_words(n: int) -> set[str]:
    """Spellings of n < 1,000,000 as words, cardinal and ordinal, one token
    per word ("twenty-one" -> {"twenty-one"}, "one hundred" -> {"hundred"})."""
    if n < 20:
        w = _UNITS[n]
    elif n < 100:
        w = _TENS[n // 10] + ("-" + _UNITS[n % 10] if n % 10 else "")
    elif n < 1000:
        return {"hundred", "hundredth"}
    elif n < 1_000_000:
        return {"thousand", "thousandth"}
    else:
        return {"million", "millionth", "billion"}
    last = w.split("-")[-1]
    ordinal = _ORD.get(last, last.rstrip("y") + "ieth" if last.endswith("y") else last + "th")
    return {w, w.rsplit("-", 1)[0] + "-" + ordinal if "-" in w else ordinal} \
        | ({"dozen"} if n == 12 else set())


def _numeral_said(doc, value: str) -> bool:
    """The sentence writes the number as a numeral: digits ("3", "1,500",
    "3rd") or a number word ("three", "third"). "Tom and Mary both laughed"
    has `Quantity 2` but no numeral, and "Tom and Mary three, both" is not a
    repair anyone makes (a0038)."""
    words = _words(doc)
    plain = {w.replace(",", "") for w in words}
    if value in plain or any(re.fullmatch(re.escape(value) + r"(st|nd|rd|th)", w) for w in plain):
        return True
    try:
        n = float(value)
    except ValueError:
        return False
    return n.is_integer() and bool(_number_words(int(n)) & set(words))


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
    """A plain number (`Quantity 3`, `EQU 100`), not a year or clock time,
    written as a numeral in the sentence."""
    return [(p, s) for p, s in site_constant(doc, value_re=NUMBER)
            if s.split()[0] not in TEMPORAL_ROLES and _numeral_said(doc, s.split()[1])]


def site_temporal(doc):
    # `YearOfCentury ?` is the slot of "In what year …", as with `Name ?`
    return [(p, v) for p, v in site_constant(doc, roles=TEMPORAL_ROLES)
            if not v.endswith(" ?")]


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


PERSONS = {"person.n.01", "male.n.02", "female.n.02"}


def _be_person(doc, c) -> bool:
    """be.v.03 whose subject is a person, not an existential "there is".

    "In" against "from" is a change of intent only for something that can
    come from a place: "She's from, in Boston", but not "There's a cat from,
    on the desk" or "My room is from, on the fourth floor" (round 3,
    2026-10-05: 6 of 19 be.v.03 sites read naturally, all with a person
    subject). Existential be has `EQU` to its `entity.n.01`; a NEGATION
    quantifier subject ("Nobody is …") is excluded as for forwarding.
    """
    if any(r == "EQU" for r, _ in c.roles):
        return False
    theme = [_target(c, v) for r, v in c.roles if r == "Theme"]
    if not theme or theme[0] is None or not 0 <= theme[0] < len(doc.concepts):
        return False
    x = doc.concepts[theme[0]]
    return x.synset in PERSONS and not _quantifier_restrictor(doc, x)


def site_edge_label(doc):
    """Every edge-label site: a direction (`site_direction`) or a place
    (`site_place`). `analyse_run.site_ok` uses this, so an answer of either
    kind counts as on site; what a run draws is set by `--edge-sites`."""
    direction = site_direction(doc)
    seen = {p for p, _ in direction}
    return direction + [s for s in site_place(doc) if s[0] not in seen]


def site_direction(doc):
    """A verb with a Source or Destination role, or be.v.03 with Location and
    a person subject, whose preposition is in the sentence. be.v.03 with
    Source ("She is from China") is not offered: §4.4 of the edge-label
    prompt allows only its Location, and the model rightly said `no_site`.

    The target must come after the verb and must not be the question slot:
    PMB uses Source for a subject ("We ran out of food") and for objects
    with no preposition ("He shined his shoes"), and a fronted wh-target
    only takes a stranded preposition ("Where on earth were you from, where
    on earth were you?", a0002). A `location.n.01` target is a preposition
    built with a spatial operator ("under" = SZN, "near" = APX), not a role.
    """
    words = set(_words(doc))
    out = []
    for c in doc.concepts:
        if c.pos_tag != "v":
            continue
        for r, v in c.roles:
            t = _target(c, v)
            if t is None or not 0 <= t < len(doc.concepts) or t < c.pos:
                continue
            if c.synset == "be.v.03":
                if r != "Location" or not _be_person(doc, c):
                    continue
            elif r not in EDGE_LABEL_SITE_ROLES:
                continue
            x = doc.concepts[t]
            if _has_wh(x) or x.synset == "location.n.01" or not (EDGE_PREPS[r] & words):
                continue
            out.append((c.pos, f"{c.synset} {r}"))
    return out


# Places (§4.4 third kind, prompts/edge_r4, 2026-10-10). The operators of
# a gold `location.n.01`, with the words that must be in the sentence for it.
PLACE_OPS = {
    "SZN": {"under", "below", "beneath", "underneath"},
    "SZP": {"above", "over"},
    "SXP": {"behind"},
    "APX": {"near", "close", "by", "besides"},
    "SY1": {"beside", "by"},
    "STI": {"in", "inside", "into", "within"},
    "STO": {"outside"},
    "SXY": {"around", "round"},
}
# A plain role is a place only with one of these right before the thing.
PLAIN_PLACE_PREPS = {"on", "in", "at", "into", "onto"}
# Things no other place fits: a part of the body ("patted me on the shoulder",
# "under his arm"), a country or larger region ("born in Switzerland": not
# "near"), and nouns that are themselves a place relative to something ("in
# the middle", "on the right").
NO_PLACE_GROUND = {"body_part.n.01", "country.n.02", "state.n.01", "continent.n.01"}
RELATIONAL_NOUNS = {"middle", "center", "centre", "side", "right", "left", "top",
                    "bottom", "front", "back", "rear", "edge", "end", "corner",
                    "inside", "outside", "interior", "exterior", "surface"}


def _place_ground_ok(x) -> bool:
    """The thing can have another place by it: a physical thing, not a body
    part, country or relational noun, and not the question slot. PMB names
    are lemma.pos.NN, which nltk resolves by sense number."""
    if _has_wh(x) or x.lemma in RELATIONAL_NOUNS or x.synset in PERSONS:
        return False
    try:
        s = wn.synset(x.synset)
    except Exception:  # noqa: BLE001 -- PMB-only names
        return False
    hyp = {h.name() for h in s.closure(lambda y: y.hypernyms() + y.instance_hypernyms())}
    hyp.add(s.name())
    return "physical_entity.n.01" in hyp and not hyp & NO_PLACE_GROUND


def _prep_before(doc, x) -> str | None:
    """The preposition nearest before the thing's head word (5 words back);
    the head is the last word of the lemma or the first word of a name."""
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", doc.sentence)]
    heads = {x.lemma.split("_")[-1].lower()}
    heads |= {v.strip('"').split()[0].lower() for r, v in x.roles
              if r == "Name" and v.strip('"')}
    for i, w in enumerate(words):
        if w in heads or (wn.morphy(w, wn.NOUN) or w) in heads:
            for j in range(i - 1, max(-1, i - 6), -1):
                if words[j] in PLAIN_PLACE_PREPS | set().union(*PLACE_OPS.values()) \
                        | {"of", "from", "to", "with", "for", "through", "across"}:
                    return words[j]
            return None
    return None


def site_place(doc):
    """A verb with a Location or Destination that is a place by a thing.

    Two forms, as the gold writes them: `location.n.01` with a spatial
    operator whose word is in the sentence (the reparandum is "on" or
    another operator), or a plain role to the thing with "on", "in", "at",
    "into" or "onto" right before it (the reparandum is an operator). The
    thing must pass `_place_ground_ok`, and, as for directions, it comes
    after the verb. The site string gives the form and the thing.
    """
    words = set(_words(doc))
    cs = doc.concepts
    out = []
    for c in cs:
        if c.pos_tag != "v":
            continue
        for r, v in c.roles:
            t = _target(c, v)
            if r not in ("Location", "Destination") or t is None \
                    or not 0 <= t < len(cs) or t < c.pos:
                continue
            x = cs[t]
            if x.synset == "location.n.01":
                if _has_wh(x):
                    continue                       # "where": location.n.01 EQU ?
                ops = [(o, _target(x, w)) for o, w in x.roles if o in PLACE_OPS]
                if not ops or ops[0][1] is None or not 0 <= ops[0][1] < len(cs):
                    continue
                op, g = ops[0]
                if PLACE_OPS[op] & words and _place_ground_ok(cs[g]):
                    out.append((c.pos, f"{c.synset} {r} {op} {cs[g].synset}"))
            elif x.pos_tag == "n" and _place_ground_ok(x):
                p = _prep_before(doc, x)
                if p in PLAIN_PLACE_PREPS:
                    out.append((c.pos, f"{c.synset} {r} plain:{p} {x.synset}"))
    return out


def site_place_only(doc):
    """Place sites of a document with no direction site, so that the model,
    which picks the site itself, cannot fall back on "from, to"."""
    return [] if site_direction(doc) else site_place(doc)


# What `--edge-sites` draws for edge_label.
EDGE_SITES = {"direction": site_direction, "place": site_place_only}


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
        # a question subject ("Who killed …") restarts into a question:
        # "Carlos killed, Who killed …" (a0043)
        if subj and tense and not _quantifier_restrictor(doc, x) and not _has_wh(x) \
                and _said_as_often(doc, v):
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
    ap.add_argument("--out", type=Path, default=HERE / "samples" / "pinned_sample.tsv")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[],
                    help="earlier sample TSVs whose documents are not drawn again")
    ap.add_argument("--exclude-type", nargs="*", default=[], metavar="TYPE",
                    help="leave out only earlier documents of these types; others "
                         "may be drawn again if used once")
    ap.add_argument("--quota", nargs="*", default=[], metavar="TYPE=N",
                    help="documents per type, overriding --per-type")
    ap.add_argument("--cap", nargs="*", default=[], metavar="TYPE=N",
                    help="documents per constant value (default 1; 0 = no cap)")
    ap.add_argument("--edge-sites", choices=sorted(EDGE_SITES), default="direction",
                    help="edge_label sites to draw: direction (from/to, as before "
                         "2026-10-10) or place (documents with place sites only)")
    args = ap.parse_args()
    quota = {t: args.per_type for t in TYPES}
    quota.update({k: int(v) for k, v in (x.split("=") for x in args.quota)})
    cap = {t: 1 for t in TYPES}
    cap.update({k: int(v) for k, v in (x.split("=") for x in args.cap)})
    for k in list(quota) + list(cap) + args.exclude_type:
        if k not in TYPES:
            ap.error(f"unknown type {k}")
    rng = random.Random(args.seed)

    earlier = [r for p in args.exclude
               for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t")]
    # With --exclude-type, a document drawn earlier for another type may be
    # drawn again (decided 2026-10-10), up to two items per source document
    # (model_experiment_pipeline.md §5.3): one used twice is always left out.
    uses = collections.Counter(r["doc_id"] for r in earlier)
    excluded = {r["doc_id"] for r in earlier
                if not args.exclude_type or r["repair_type"] in args.exclude_type}
    excluded |= {d for d, n in uses.items() if n >= 2}
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
    draw = dict(SITES, edge_label=EDGE_SITES[args.edge_sites])
    for d in pool:
        for t, fn in draw.items():
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
        per_value: collections.Counter = collections.Counter()
        picked = 0
        for d, sites in cands:
            if picked >= quota[t]:
                break
            if d.doc_id in used:
                continue
            if t.startswith("constant_"):
                vals = {s for _, s in sites}
                if cap[t] and all(per_value[v] >= cap[t] for v in vals):
                    continue
                per_value.update(vals)
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
        if picked < quota[t]:
            print(f"  only {picked} of {quota[t]} documents for {t}")

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
