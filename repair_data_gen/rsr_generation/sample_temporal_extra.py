"""Second temporal item per document, with the site given (2026-10-10).

Temporal constants occur in few train documents (297 under the current site
rules), so the first pass (full run + round 2, 269 passing) is short of the
~500 target. Every eligible document gets one more item, run with
`prompts/temporal_r2/`, which takes the site as input:

- `new_site`: a site no passing temporal item of this document uses. A
  document with a passing item and a second site gets that second site; a
  document with no passing item gets its first site.
- `same_site`: the document's only site already has a passing item; the new
  item repairs it again with a different reparandum (`avoid_value`).

Use by other repair types is not counted: the per-document cap is a
reporting matter (items and source documents per type), not a selection
rule (decided 2026-10-10, user). One row per document, since run_llm.py
keys results and raw files on doc_id.

    python3 sample_temporal_extra.py      # writes sample_train_temporal_extra.tsv
"""
import collections
import csv
import random
from pathlib import Path

import select_docs as S
from audit import auto_pass

HERE = Path(__file__).parent
FIRST_PASS = [HERE / "results" / f"meta-models_Muse-Glimmer-30B.{t}.checked.tsv"
              for t in ("train_full", "r2")]
OUT = HERE / "samples" / "sample_train_temporal_extra.tsv"


def site_str(doc, pos: int, role_value: str) -> str:
    return f"{pos} {doc.concepts[pos].synset} {role_value}"


def used_site(sites, row):
    """The site a passing item repaired: its position, and the role whose
    reparandum value sits in the CORRECTION box."""
    for pos, rv in sites:
        role = rv.split()[0]
        if pos == int(row["site_index"]) and \
                f"{role} {row['reparandum_value']}" in row["sbn_repair"]:
            return (pos, rv)
    return None


def main() -> None:
    csv.field_size_limit(10**9)
    first = [r for p in FIRST_PASS
             for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t")
             if r["repair_type"] == "constant_temporal" and auto_pass(r)]
    passed = collections.defaultdict(list)
    for r in first:
        passed[r["doc_id"]].append(r)

    rows, kinds, unmatched = [], collections.Counter(), []
    for d in S.read_split(S.GOLD):
        if d.doc_id in S.EXAMPLE_DOCS or not S.gold_is_sound(d):
            continue
        sites = sorted(set(S.site_temporal(d)))
        if not sites:
            continue
        used = {}
        for r in passed.get(d.doc_id, []):
            s = used_site(sites, r)
            if s is None:
                unmatched.append(d.doc_id)
            else:
                used[s] = r["reparandum_value"]
        free = [s for s in sites if s not in used]
        if free:
            site, kind, avoid = free[0], "new_site", "none"
        else:
            site = sites[0]
            kind, avoid = "same_site", used[site]
        kinds[kind] += 1
        rows.append({
            "doc_id": d.doc_id,
            "repair_type": "constant_temporal",
            "nl_clean": d.sentence,
            "sbn_clean": d.raw,
            "n_concepts": len(d.concepts),
            "n_tokens_nl": len(d.sentence.split()),
            "eligible_sites": "; ".join(f"{p}:{s}" for p, s in sites),
            "kind": kind,
            "pinned_site": site_str(d, *site),
            "avoid_value": avoid,
            "first_pass_items": len(passed.get(d.doc_id, [])),
        })

    random.Random(13).shuffle(rows)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} documents -> {OUT.name}: {dict(kinds)}")
    print(f"first-pass passing items: {len(first)} in {len(passed)} documents")
    if unmatched:
        print(f"passing items whose site is not a current site: {unmatched}")


if __name__ == "__main__":
    main()
