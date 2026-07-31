"""
Non-finite verb constructions (participial / infinitival adjuncts) in PMB gold,
and whether the repair transform can handle a self-repair that targets them.

"Non-finite verb" here means a VBG/VBN/VB token that is NOT the main predicate
of its clause -- spaCy dep_ in {advcl, acl, xcomp} and not ROOT. This covers:
  - past-participle adjuncts:   "..., deeply moved by X, ..."
  - present-participle adjuncts: "Having finished X, he went out."
  - bare-infinitive adjuncts:    "To win the race, she trained."
These can precede or follow the main clause; SBN has no special box for them
(no separator is attested for participial/infinitival adjuncts in
SBNSpec.NEW_BOX_INDICATORS), so whatever concept the participle aligns to is
just an ordinary verb concept wired into the same box as the main clause --
the question is only whether *its own* argument structure (which typically
includes an inbound link from/e to the noun it modifies) survives a splice.

Run:  python3 nonfinite_probe.py [n_docs]
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import spacy

from sbn_lin import SYNSET_PATTERN, read_split
from repair_transform import build_repair
from wn_candidates import candidates
from generate_repairs import align, NON_LEXICAL

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/pmb-5.1.0/split/en/train/gold.sbn"

NONFINITE_DEPS = {"advcl", "acl", "xcomp"}
NONFINITE_TAGS = {"VBG", "VBN", "VB"}


def find_nonfinite_tokens(sdoc):
    out = []
    for t in sdoc:
        if t.tag_ in NONFINITE_TAGS and t.dep_ in NONFINITE_DEPS and t.dep_ != "ROOT":
            # exclude a finite copula/aux immediately governing it as part of
            # a full passive/progressive main clause, e.g. "was moved" as ROOT
            if t.head.dep_ == "ROOT" and t.head.tag_ in ("VBD", "VBZ", "VBP"):
                continue
            out.append(t)
    return out


def main() -> None:
    n_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    docs = read_split(GOLD)
    if n_docs:
        docs = docs[:n_docs]
    nlp = spacy.load("en_core_web_sm")

    stats = collections.Counter()
    dep_tag_counts = collections.Counter()
    examples = []
    repair_results = []

    texts = [d.sentence for d in docs]
    for doc_sbn, sdoc in zip(docs, nlp.pipe(texts, batch_size=256)):
        nf_toks = find_nonfinite_tokens(sdoc)
        if not nf_toks:
            continue
        stats["docs_with_nonfinite"] += 1
        amap = align(doc_sbn, sdoc)                 # concept pos -> token idx
        tok_to_concept = {v: k for k, v in amap.items()}

        for t in nf_toks:
            dep_tag_counts[f"{t.dep_}/{t.tag_}"] += 1
            stats["nonfinite_tokens"] += 1
            cpos = tok_to_concept.get(t.i)
            if cpos is None:
                stats["unaligned_to_concept"] += 1
                continue
            stats["aligned_to_concept"] += 1
            if len(examples) < 12:
                examples.append((doc_sbn.doc_id, doc_sbn.sentence, t.text,
                                 t.tag_, t.dep_, doc_sbn.concepts[cpos].synset))

            cands = candidates(doc_sbn.concepts[cpos].synset)
            if not cands:
                stats["no_wn_candidate"] += 1
                continue
            res = build_repair(doc_sbn, cpos, cands[0])
            repair_results.append((doc_sbn.doc_id, doc_sbn.sentence, t.text,
                                   doc_sbn.concepts[cpos].synset, cands[0],
                                   res))
            if res.ok:
                stats["repair_legal"] += 1
            else:
                stats["repair_blocked"] += 1

    print(f"docs scanned: {len(docs)}")
    print(f"docs containing >=1 non-finite verb token: {stats['docs_with_nonfinite']}")
    print(f"non-finite tokens found: {stats['nonfinite_tokens']}")
    print(f"  aligned to an SBN concept:     {stats['aligned_to_concept']}")
    print(f"  NOT aligned (spaCy/SBN mismatch): {stats['unaligned_to_concept']}")
    print()
    print("breakdown by dep/tag:", dict(dep_tag_counts.most_common()))
    print()
    print("=== example constructions found ===")
    for doc_id, sent, tok, tag, dep, synset in examples:
        print(f"  {doc_id}  [{tok}/{tag}/{dep} -> {synset}]  {sent}")
    print()
    print("=== repair attempts on the non-finite verb's own concept ===")
    print(f"  no WordNet candidate at all: {stats['no_wn_candidate']}")
    print(f"  legal splice:                {stats['repair_legal']}")
    print(f"  blocked:                     {stats['repair_blocked']}")
    print()
    for doc_id, sent, tok, synset, cand, res in repair_results:
        flag = "OK  " if res.ok else "FAIL"
        print(f"  [{flag}] {doc_id}  {tok}({synset}->{cand})  "
              f"strategy={res.strategy.value if res.ok else '-'} "
              f"blockers={[b.value for b in res.blockers]}")
        print(f"         {sent}")
        print(f"         {res.sbn}")


if __name__ == "__main__":
    main()
