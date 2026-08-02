"""Build the candidate pool: one row per (repair site x candidate reparandum).

This is the artefact DESIGN.md §4.7 calls the surplus pool, and the input to
selection.  Keeping it as a file on disk rather than a step inside the
generator is deliberate: the pool is what a later re-tuning of the band
re-filters, and it is what a human reads when judging whether the filter is
behaving.

Two layers run here, in the order DESIGN.md §5 requires:

  rule layer      `build_repair` on every candidate.  A candidate whose splice
                  is illegal is out regardless of how natural it reads, and
                  the check is free because the splice has to be built anyway.
  language layer  optional (`--nli`): the lower bound of the band, rejecting
                  candidates that are synonyms of the repair.  See
                  `nli_filter` for why this is an NLI model and not a cosine.

Note what `nl_substituted` is: the *clean* sentence with the repair word
swapped for the candidate.  It is not the repaired sentence.  The question the
filter asks is whether the two words are interchangeable in this context, which
is about the pair, not about the repair construction.

Run:
    python3 build_pool.py --split train --out pool_train.tsv --nli
    python3 build_pool.py --split train --out pool_head.tsv --limit 200 --nli
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import spacy

from generate_repairs import NON_LEXICAL, WN2UD, align
from inflect_en import inflect, match_case
from repair_transform import build_repair
from sbn_lin import SYNSET_PATTERN, read_split
from wn_candidates import candidates

ROOT = Path(__file__).resolve().parents[1]

SPLITS = {
    "train": "data/pmb-5.1.0/split/en/train/gold.sbn",
    "dev": "data/pmb-5.1.0/split/en/dev/standard.sbn",
    "test": "data/pmb-5.1.0/split/en/test/standard.sbn",
    "test_long": "data/pmb-5.1.0/split/en/test/long.sbn",
}

FIELDS = [
    "doc_id", "split", "concept_pos", "repair_synset", "reparandum_synset",
    "reparandum_surface", "inflection_confident", "repair_pos",
    "legal", "strategy", "devices", "blockers", "max_abs_index",
    "nl_clean", "nl_substituted",
]
NLI_FIELDS = ["entail_fwd", "entail_bwd", "synonymy", "contradiction", "neutral"]


def substituted_sentence(spacy_doc, tok_i: int, surface: str) -> str:
    """The clean sentence with token `tok_i` replaced by `surface`."""
    tok = spacy_doc[tok_i]
    new = match_case(tok.text, surface)
    if tok_i == 0:
        new = match_case(spacy_doc[0].text, new)
    return "".join(
        (new + t.whitespace_) if t.i == tok_i else t.text_with_ws
        for t in spacy_doc
    ).strip()


def build(split: str, limit: int) -> list[dict]:
    docs = read_split(ROOT / SPLITS[split])
    if limit:
        docs = docs[:limit]
    nlp = spacy.load("en_core_web_sm")

    rows: list[dict] = []
    for doc_sbn, sdoc in zip(docs, nlp.pipe([d.sentence for d in docs],
                                            batch_size=256)):
        amap, _named = align(doc_sbn, sdoc)
        for cpos, tok_i in sorted(amap.items()):
            c = doc_sbn.concepts[cpos]
            if c.pos_tag not in WN2UD or c.synset in NON_LEXICAL:
                continue
            tok = sdoc[tok_i]
            for cand in candidates(c.synset):
                res = build_repair(doc_sbn, cpos, cand)
                m = SYNSET_PATTERN.match(cand)
                surface, conf = inflect(m.group(1).replace("_", " "),
                                        tok.tag_, c.pos_tag)
                rows.append({
                    "doc_id": doc_sbn.doc_id,
                    "split": split,
                    "concept_pos": cpos,
                    "repair_synset": c.synset,
                    "reparandum_synset": cand,
                    "reparandum_surface": surface,
                    "inflection_confident": int(conf),
                    "repair_pos": c.pos_tag,
                    "legal": int(res.ok),
                    "strategy": res.strategy.value,
                    "devices": "|".join(d.value for d in res.devices),
                    "blockers": "|".join(b.value for b in res.blockers),
                    "max_abs_index": res.max_abs_index,
                    "nl_clean": doc_sbn.sentence,
                    "nl_substituted": substituted_sentence(sdoc, tok_i, surface),
                })
    return rows


def add_nli(rows: list[dict], batch_size: int) -> None:
    """Score the legal rows in place.  Illegal rows are left blank: they are
    already excluded, and scoring them would waste the run's whole budget on
    candidates that cannot be used."""
    from nli_filter import judge_pairs

    todo = [r for r in rows if r["legal"]]
    print(f"scoring {len(todo)} legal candidates with NLI "
          f"({len(rows) - len(todo)} illegal rows skipped)")
    t0 = time.time()
    verdicts = judge_pairs([(r["nl_clean"], r["nl_substituted"]) for r in todo],
                           batch_size=batch_size)
    dt = time.time() - t0
    print(f"  {dt:.0f}s  ({len(todo) / dt:.0f} pairs/s)")

    for r in rows:
        for f in NLI_FIELDS:
            r[f] = ""
    for r, v in zip(todo, verdicts):
        r["entail_fwd"] = round(v.entail_fwd, 4)
        r["entail_bwd"] = round(v.entail_bwd, 4)
        r["synonymy"] = round(v.synonymy, 4)
        r["contradiction"] = round(v.contradiction, 4)
        r["neutral"] = round(v.neutral, 4)


def summarise(rows: list[dict], with_nli: bool) -> None:
    import collections

    sites = {(r["doc_id"], r["concept_pos"]) for r in rows}
    legal = [r for r in rows if r["legal"]]
    print()
    print(f"repair sites            {len(sites):8d}")
    print(f"candidate rows          {len(rows):8d}  "
          f"({len(rows) / max(len(sites), 1):.1f} per site)")
    print(f"  splice legal          {len(legal):8d}  {len(legal)/len(rows):6.1%}")
    by_pos = collections.Counter(r["repair_pos"] for r in legal)
    print(f"  by POS                {dict(by_pos)}")

    if not with_nli:
        return
    from nli_filter import SYNONYMY_REJECT_ABOVE

    scored = [r for r in legal if r["synonymy"] != ""]
    rej = [r for r in scored if float(r["synonymy"]) > SYNONYMY_REJECT_ABOVE]
    print(f"  NLI: too similar      {len(rej):8d}  {len(rej)/len(scored):6.1%} "
          f"(synonymy > {SYNONYMY_REJECT_ABOVE})")
    for p in "nva":
        s = [r for r in scored if r["repair_pos"] == p]
        if not s:
            continue
        r_ = [x for x in s if float(x["synonymy"]) > SYNONYMY_REJECT_ABOVE]
        print(f"    {p}: {len(r_)}/{len(s)} = {len(r_)/len(s):.1%}")

    # How many sites keep at least one candidate?  This is the number that
    # decides whether the filter costs coverage.
    surviving = {(r["doc_id"], r["concept_pos"]) for r in scored
                 if float(r["synonymy"]) <= SYNONYMY_REJECT_ABOVE}
    legal_sites = {(r["doc_id"], r["concept_pos"]) for r in scored}
    print(f"  sites keeping >=1     {len(surviving):8d} / {len(legal_sites)} "
          f"= {len(surviving)/max(len(legal_sites),1):.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=sorted(SPLITS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N source documents")
    ap.add_argument("--nli", action="store_true",
                    help="score legal candidates with the NLI lower bound")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    rows = build(args.split, args.limit)
    if args.nli:
        add_nli(rows, args.batch_size)

    fields = FIELDS + (NLI_FIELDS if args.nli else [])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    summarise(rows, args.nli)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
