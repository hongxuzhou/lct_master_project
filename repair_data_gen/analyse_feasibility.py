"""
How far does the LARD-on-gold-SBN recipe actually reach?

Measured over data/pmb-5.1.0/split/en/train/gold.sbn:
  1. surface alignment  -- can the concept be tied to an uttered word at all?
  2. reparandum supply  -- does WordNet offer an SBN-writable neighbour?
  3. splice legality    -- does CORRECTION/CONJUNCTION keep every index legal,
                           and at what cost (which device)?
  4. surface monotonicity -- device (1) ("the argument was never uttered")
                           is only honest if SBN order tracks word order.

Run:  python3 analyse_feasibility.py
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

from nltk.corpus import wordnet as wn

from sbn_lin import INVERSE_OF, SYNSET_PATTERN, read_split
from repair_transform import STRATEGY_ORDER, build_repair
from wn_candidates import candidates, raw_pool_size

GOLD = Path(__file__).resolve().parents[1] / "data/pmb-5.1.0/split/en/train/gold.sbn"

# Concepts that are structural / pragmatic scaffolding rather than an
# utterable content word.  Repairing these needs a different template.
NON_LEXICAL = {
    "time.n.08", "entity.n.01", "person.n.01", "male.n.02", "female.n.02",
    "quantity.n.01", "location.n.01", "event.n.01", "thing.n.12",
}
POS_OK = {"n", "v", "a"}
WORD_RE = re.compile(r"[A-Za-z']+")


def surface_forms(synset_id: str) -> set[str]:
    m = SYNSET_PATTERN.match(synset_id)
    if not m:
        return set()
    lemma, pos, num = m.groups()
    forms = {lemma.replace("_", " ").lower()}
    try:
        ss = wn.synset(f"{lemma}.{pos}.{num}")
        forms |= {l.name().replace("_", " ").lower() for l in ss.lemmas()}
    except Exception:
        pass
    return forms


def token_match(forms: set[str], lowered: list[str]) -> int | None:
    for f in forms:
        head = f.split()[0]
        for k, t in enumerate(lowered):
            if t == head:
                return k
            if len(head) >= 4 and len(t) >= 4 and t[:4] == head[:4] and (
                t.startswith(head) or head.startswith(t)
            ):
                return k
    return None


def pct(a: int, b: int) -> str:
    return f"{a/b:6.1%}" if b else "     -"


def main() -> None:
    docs = read_split(GOLD)
    print(f"gold docs parsed: {len(docs)}  (skipped {len(read_split.skipped)})\n")

    S = collections.Counter()
    by_pos = collections.defaultdict(collections.Counter)
    strat_counts = collections.Counter()
    device_counts = collections.Counter()
    blocker_counts = collections.Counter()
    hard_role = collections.Counter()
    non_monotone = 0
    sat_lost = collections.Counter()
    tail_risk = 0
    sites_per_doc = collections.Counter()
    examples = collections.defaultdict(list)

    for doc in docs:
        lowered = [t.lower() for t in WORD_RE.findall(doc.sentence)]
        aligned: dict[int, int] = {}
        for c in doc.concepts:
            if c.pos_tag in POS_OK and c.synset not in NON_LEXICAL:
                k = token_match(surface_forms(c.synset), lowered)
                if k is not None:
                    aligned[c.pos] = k
        seq = [aligned[p] for p in sorted(aligned)]
        if any(b < a for a, b in zip(seq, seq[1:])):
            non_monotone += 1

        n_ok = 0
        for c in doc.concepts:
            tag = c.pos_tag
            if tag not in POS_OK or c.synset in NON_LEXICAL:
                S["nonlexical"] += 1
                continue
            S["lexical"] += 1
            by_pos[tag]["lexical"] += 1

            if c.pos not in aligned:
                S["unaligned"] += 1
                by_pos[tag]["unaligned"] += 1
                continue
            S["aligned"] += 1
            by_pos[tag]["aligned"] += 1

            raw, writable = raw_pool_size(c.synset)
            if raw and not writable:
                sat_lost[tag] += 1
            cands = candidates(c.synset)
            if not cands:
                S["no_candidate"] += 1
                by_pos[tag]["no_candidate"] += 1
                if len(examples["no_candidate"]) < 6:
                    examples["no_candidate"].append((doc.doc_id, c.synset, doc.sentence))
                continue
            S["usable"] += 1
            by_pos[tag]["usable"] += 1

            res = build_repair(doc, c.pos, cands[0])
            if res.ok:
                S["feasible"] += 1
                by_pos[tag]["feasible"] += 1
                strat_counts[res.strategy.value] += 1
                by_pos[tag]["strat:" + res.strategy.value] += 1
                for d in res.devices:
                    device_counts[d.value] += 1
                if res.separator_in_tail:
                    tail_risk += 1
                n_ok += 1
            else:
                S["blocked"] += 1
                by_pos[tag]["blocked"] += 1
                # attribute to the cheapest strategy's blocker set
                worst = build_repair(doc, c.pos, cands[0],
                                     strategy=STRATEGY_ORDER[-1])
                for b in worst.blockers:
                    blocker_counts[b.value] += 1
                    if len(examples[b.value]) < 6:
                        examples[b.value].append((doc.doc_id, c.synset, doc.sentence))
                for (s, r, t) in doc.edges():
                    if t == c.pos and s < c.pos and r not in INVERSE_OF:
                        hard_role[r] += 1
        sites_per_doc[min(n_ok, 6)] += 1

    L, A, U = S["lexical"], S["aligned"], S["usable"]
    print("=== funnel: gold concepts -> usable repair sites ===")
    print(f"  structural / non-lexical concepts   {S['nonlexical']:7d}")
    print(f"  lexical concepts (n/v/a)            {L:7d}   100.0%")
    print(f"    not alignable to an uttered word  {S['unaligned']:7d}  {pct(S['unaligned'], L)}")
    print(f"    aligned                           {A:7d}  {pct(A, L)}")
    print(f"      no WordNet reparandum available {S['no_candidate']:7d}  {pct(S['no_candidate'], L)}")
    print(f"      usable candidate site           {U:7d}  {pct(U, L)}")
    print()
    print("=== splice outcome on usable sites ===")
    print(f"  legal      {S['feasible']:7d}  {pct(S['feasible'], U)}")
    print(f"  blocked    {S['blocked']:7d}  {pct(S['blocked'], U)}")
    print(f"    (of the legal ones, {tail_risk} have a separator downstream, so")
    print(f"     the CONJUNCTION tail swallows a box opener -- C5 risk)")
    print()
    print("  strategy actually needed:", dict(strat_counts))
    print("  devices used            :", dict(device_counts))
    print("  blockers (all strategies exhausted):", dict(blocker_counts))
    print("  non-invertible inbound roles:", hard_role.most_common(10))
    print()
    print("=== by WordNet POS ===")
    hdr = f"  {'pos':>3} {'lexical':>8} {'aligned':>8} {'usable':>8} {'legal':>8}   {'inplace/anchor/hoist'}"
    print(hdr)
    for p in ("n", "v", "a"):
        b = by_pos[p]
        print(f"  {p:>3} {b['lexical']:8d} {b['aligned']:8d} {b['usable']:8d} "
              f"{b['feasible']:8d}   "
              f"{b['strat:inplace']}/{b['strat:anchor']}/{b['strat:hoist']}")
    print()
    print("  concepts whose entire WordNet neighbour pool is unwritable in SBN")
    print(f"  (satellite `.s.` adjectives): {dict(sat_lost)}")
    print()
    print(f"=== surface order ===")
    print(f"  docs whose aligned SBN concept order is NOT monotone in the "
          f"sentence: {non_monotone} / {len(docs)} ({non_monotone/len(docs):.1%})")
    print()
    print("=== legal sites per document ===")
    tot = sum(sites_per_doc.values())
    for k in sorted(sites_per_doc):
        lbl = f"{k}" if k < 6 else "6+"
        print(f"  {lbl}: {sites_per_doc[k]:5d} docs  {pct(sites_per_doc[k], tot)}")
    usable_docs = tot - sites_per_doc[0]
    print(f"  -> {usable_docs} / {tot} docs ({usable_docs/tot:.1%}) yield at "
          f"least one repair sample")
    print()
    print("=== examples ===")
    for k, v in examples.items():
        print(f"  [{k}]")
        for doc_id, syn, sent in v:
            print(f"    {doc_id}  {syn:24s} {sent}")


if __name__ == "__main__":
    main()
