"""Place repairs, round 5: the site is given (2026-10-11).

Round 4 (`sample_train_place.tsv`, `prompt_docs_edge_r4/`) let the model pick
the site, and 121 of 627 documents gave a passing place item. The rest went
three ways: a role swap to "from" on `be.v.03` with a thing as subject (the
person-subject rule is in selection only), a "from" swap on a placement
`Destination` ("put the key in the lock"; the preposition condition is in
selection only), or `no_site` on a plain "on"/"in"/"at" with a verb other
than `be.v.03` (the model read the third kind as "be.v.03 or a gold
operator"). Round 5 runs the documents without a passing item again, with
`prompt_docs_place_r5/`: a place-only §4.4 and the site as input, as
`prompt_docs_temporal_r2/` does for temporal constants.

One row per document (run_llm.py keys on doc_id); the pinned site is the
document's first place site. `form` and `verb` say how the gold writes the
place (`plain`, `operator`) and whether the verb is `be.v.03`, for the
pilot strata and for reporting. Writes the full sample and a pilot drawn
from it; the full sample leaves the pilot documents out, so pilot items
pool with it as long as the prompt does not change after the pilot.

    python3 sample_place_r5.py     # sample_place_r5.tsv, sample_place_r5_pilot.tsv
"""
import collections
import csv
import random
from pathlib import Path

import select_docs as S
from audit import auto_pass

HERE = Path(__file__).parent
R4_SAMPLE = HERE / "sample_train_place.tsv"
R4_RESULT = HERE / "results" / "meta-models_Muse-Glimmer-30B.train_place.checked.tsv"
OUT = HERE / "sample_place_r5.tsv"
PILOT = HERE / "sample_place_r5_pilot.tsv"
# pilot strata (form, verb): the cell r4 failed on gets the most items
PILOT_QUOTA = {("plain", "other"): 25, ("plain", "be"): 10,
               ("operator", "other"): 10, ("operator", "be"): 5}
SEED = 13


def main() -> None:
    csv.field_size_limit(10**9)
    r4 = list(csv.DictReader(open(R4_SAMPLE, encoding="utf-8"), delimiter="\t"))
    passed = {r["doc_id"] for r in csv.DictReader(open(R4_RESULT, encoding="utf-8"),
                                                  delimiter="\t") if auto_pass(r)}
    want = {r["doc_id"] for r in r4} - passed
    docs = {d.doc_id: d for d in S.read_split(S.GOLD) if d.doc_id in want}

    rows, no_site = [], []
    for doc_id in sorted(want):
        d = docs[doc_id]
        sites = S.site_place(d)
        if not sites:
            no_site.append(doc_id)
            continue
        pos, site = sites[0]
        form = "plain" if " plain:" in site else "operator"
        verb = "be" if site.startswith("be.v.03 ") else "other"
        rows.append({
            "doc_id": d.doc_id,
            "repair_type": "edge_label",
            "nl_clean": d.sentence,
            "sbn_clean": d.raw,
            "n_concepts": len(d.concepts),
            "n_tokens_nl": len(d.sentence.split()),
            "eligible_sites": "; ".join(f"{p}:{s}" for p, s in sites),
            "pinned_site": f"{pos} {site}",
            "form": form,
            "verb": verb,
        })

    rng = random.Random(SEED)
    rng.shuffle(rows)
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["form"], r["verb"])].append(r)
    pilot = [r for cell, n in PILOT_QUOTA.items() for r in cells[cell][:n]]
    pilot_ids = {r["doc_id"] for r in pilot}
    rest = [r for r in rows if r["doc_id"] not in pilot_ids]

    for path, rs in ((OUT, rest), (PILOT, pilot)):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rs)
    print(f"r4: {len(r4)} documents, {len(passed)} with a passing item, "
          f"{len(want)} to run again")
    print("cells (form, verb):",
          {f"{k[0]}/{k[1]}": len(v) for k, v in sorted(cells.items())})
    print(f"{len(pilot)} -> {PILOT.name}, {len(rest)} -> {OUT.name}")
    if no_site:
        print(f"no place site under the current rule: {no_site}")


if __name__ == "__main__":
    main()
