"""Did a round-5 place item keep to its given site and repair a place?

Runs after analyse_run.py, on its checked TSV, and joins the sample from
sample_place_r5.py on doc_id. Writes the checked rows plus one column,
`pin`: `ok`, or what is wrong:

- `site`: `site_index` is not the given verb;
- `not_place`: `reparandum_value` is neither a spatial operator nor `plain`
  (a role swap such as `Source`);
- `role`: the CORRECTION box carries a role other than the gold one;
- `same_as_gold`: the first version is written as the gold is (`plain`
  against a plain gold, or the gold's own operator).

`no_site` rows get `-`. Kept out of analyse_run.py on purpose: it is a check
for this run only.

    python3 check_place_r5.py results/meta-models_Muse-Glimmer-30B.place_r5.checked.tsv
"""
import argparse
import collections
import csv
from pathlib import Path

from select_docs import PLACE_OPS

HERE = Path(__file__).parent


def pin(row: dict, s: dict) -> str:
    if row.get("status") != "ok" or not row.get("sbn_repair"):
        return "-"
    pos, _verb, role, form, _thing = s["pinned_site"].split(" ")
    value = row["reparandum_value"].strip()
    correction = row["sbn_repair"].split("CORRECTION", 1)[-1].split("CONJUNCTION", 1)[0]
    if row["site_index"] != pos:
        return "site"
    if value not in PLACE_OPS and value != "plain":
        return "not_place"
    if f" {role} " not in f" {correction} " or any(
            f" {r} " in f" {correction} " for r in ("Source", "Destination", "Location")
            if r != role):
        return "role"
    gold = "plain" if form.startswith("plain:") else form
    if value == gold:
        return "same_as_gold"
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checked", type=Path)
    ap.add_argument("--sample", type=Path, nargs="+",
                    default=[HERE / "samples" / "sample_place_r5.tsv", HERE / "samples" / "sample_place_r5_pilot.tsv"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    csv.field_size_limit(10**9)
    sample = {r["doc_id"]: r for p in args.sample
              for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t")}
    rows = list(csv.DictReader(open(args.checked, encoding="utf-8"), delimiter="\t"))
    tally = collections.Counter()
    for r in rows:
        s = sample[r["doc_id"]]
        r["pin"] = pin(r, s)
        tally[(f"{s['form']}/{s['verb']}", r["pin"])] += 1
    out = args.out or args.checked.with_suffix(".pin.tsv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    for (cell, p), n in sorted(tally.items()):
        print(f"  {cell:15s} {p:13s} {n:4d}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
