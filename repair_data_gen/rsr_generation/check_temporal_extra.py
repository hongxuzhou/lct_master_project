"""Did a temporal-extra item keep to its given site and avoid the used value?

Runs after analyse_run.py, on its checked TSV, and joins the sample from
sample_temporal_extra.py on doc_id. Writes the checked rows plus one column,
`pin`: `ok`, or what is wrong:

- `site`: `site_index` is not the given position, or the CORRECTION box
  does not hold the given role with the reparandum value;
- `same_as_used`: the reparandum value is the one the first item used;
- `same_as_gold`: the reparandum value is the gold value.

`no_site` rows get `-`. Kept out of analyse_run.py on purpose: it is a check
for this run only.

    python3 check_temporal_extra.py results/meta-models_Muse-Glimmer-30B.train_temporal_extra.checked.tsv
"""
import argparse
import collections
import csv
from pathlib import Path

HERE = Path(__file__).parent


def norm(v: str) -> str:
    return v.strip().strip("'\"").lower()


def pin(row: dict, s: dict) -> str:
    if row.get("status") != "ok" or not row.get("sbn_repair"):
        return "-"
    pos, _synset, role, gold = s["pinned_site"].split(" ", 3)
    value = row["reparandum_value"]
    correction = row["sbn_repair"].split("CORRECTION", 1)[-1].split("CONJUNCTION", 1)[0]
    if row["site_index"] != pos or f"{role} {value}" not in correction:
        return "site"
    if s["avoid_value"] != "none" and norm(value) == norm(s["avoid_value"]):
        return "same_as_used"
    if norm(value) == norm(gold):
        return "same_as_gold"
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checked", type=Path)
    ap.add_argument("--sample", type=Path, default=HERE / "samples" / "sample_train_temporal_extra.tsv")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    csv.field_size_limit(10**9)
    sample = {r["doc_id"]: r for r in csv.DictReader(open(args.sample, encoding="utf-8"),
                                                     delimiter="\t")}
    rows = list(csv.DictReader(open(args.checked, encoding="utf-8"), delimiter="\t"))
    tally = collections.Counter()
    for r in rows:
        s = sample[r["doc_id"]]
        r["pin"] = pin(r, s)
        tally[(s["kind"], r["pin"])] += 1
    out = args.out or args.checked.with_suffix(".pin.tsv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    for (kind, p), n in sorted(tally.items()):
        print(f"  {kind:10s} {p:13s} {n:4d}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
