"""
Summarise one run of run_llm.py, per repair type.

    python3 analyse_run.py results/<model>.tsv [--out results/<model>.checked.tsv]

Per row it adds:
- the structural checks of validate_with_pmb_parser.check();
- `wn_ok`: every concept in the repaired SBN that is not in the gold SBN
  exists in WordNet (wn_lookup.cmd_check);
- `oracle`: for content-word repairs, whether the SBN equals what
  repair_transform.build_repair writes for the same site and reparandum
  (`same`, `differs`, or `n/a` where build_repair cannot be run);
- `lookups`: tool calls the model made (num_turns - 1).
Then it prints counts per repair type.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from repair_transform import build_repair  # noqa: E402
from sbn_lin import parse_sbn_line  # noqa: E402
from validate_with_pmb_parser import check, strip_comments  # noqa: E402
import wn_lookup  # noqa: E402

CONCEPT = re.compile(r"^[^\s\"]+\.[nvar]\.\d\d$")
CHECKS = ("parse", "dag", "index", "magnitude", "edges", "boxes")
TYPES = ("content_word", "constant_name", "constant_number",
         "constant_temporal", "tense", "edge_label", "forwarding")


def new_concepts_exist(gold: str, rep: str) -> tuple[str, str]:
    """('ok' or 'missing', the missing names) for concepts not in gold."""
    gold_c = {t for t in gold.split() if CONCEPT.match(t)}
    missing = []
    for t in strip_comments(rep).split():
        if CONCEPT.match(t) and t not in gold_c:
            with contextlib.redirect_stdout(io.StringIO()):
                if wn_lookup.cmd_check(t) != 0:
                    missing.append(t)
    return ("ok" if not missing else "missing"), " ".join(sorted(set(missing)))


def oracle(row: dict) -> str:
    if row["repair_type"] != "content_word":
        return ""
    try:
        doc = parse_sbn_line(row["sbn_clean"])
        i = int(row["site_index"])
        if not 0 <= i < len(doc.concepts) or doc.concepts[i].synset != row["repair_synset"]:
            return "n/a: site_index does not match site_synset"
        res = build_repair(doc, i, row["reparandum_synset"])
    except Exception as e:  # noqa: BLE001
        return f"n/a: {type(e).__name__}"
    if not res.ok:
        return "n/a: build_repair blocked"
    return "same" if res.sbn.strip() == row["sbn_repair"].strip() else "differs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.tsv, encoding="utf-8"), delimiter="\t"))

    for r in rows:
        r["lookups"] = str(int(r["num_turns"]) - 1) if r.get("num_turns") else ""
        answered = r.get("parse") == "ok" and r.get("status") == "ok" and r.get("sbn_repair")
        if not answered:
            for k in CHECKS + ("wn_ok", "wn_missing", "oracle", "check_detail"):
                r[k] = ""
            continue
        res = check(r["sbn_repair"])
        for k in CHECKS:
            r[k] = res[k] or ""
        r["check_detail"] = res["detail"]
        r["wn_ok"], r["wn_missing"] = new_concepts_exist(r["sbn_clean"], r["sbn_repair"])
        r["oracle"] = oracle(r)

    by = collections.defaultdict(list)
    for r in rows:
        by[r["repair_type"]].append(r)
    head = ("type", "n", "json", "no_site", "answered", "struct_ok", "wn_ok",
            "oracle_same", "median_lookups")
    print("  ".join(f"{h:>15s}" if k else f"{h:18s}" for k, h in enumerate(head)))
    for t in TYPES + ("ALL",):
        rs = rows if t == "ALL" else by.get(t, [])
        if not rs:
            continue
        ans = [r for r in rs if r.get("parse") == "ok" and r.get("status") == "ok"]
        struct = [r for r in ans if all(r[k] == "ok" for k in CHECKS)]
        cw = [r for r in ans if r["repair_type"] == "content_word"]
        same = sum(r["oracle"] == "same" for r in cw)
        look = sorted(int(r["lookups"]) for r in rs if r["lookups"])
        cells = [t, len(rs),
                 sum(r.get("parse") == "ok" for r in rs),
                 sum(r.get("status") == "no_site" for r in rs),
                 len(ans), len(struct),
                 sum(r["wn_ok"] == "ok" for r in ans),
                 f"{same}/{len(cw)}" if cw else "-",
                 look[len(look) // 2] if look else "-"]
        print("  ".join(f"{str(c):>15s}" if k else f"{str(c):18s}" for k, c in enumerate(cells)))

    print("\nfailures (answered rows failing a check):")
    for r in rows:
        bad = [f"{k}={r[k]}" for k in CHECKS if r.get(k) not in ("", "ok")]
        if r.get("wn_ok") == "missing":
            bad.append(f"not in WordNet: {r['wn_missing']}")
        if r.get("oracle", "").startswith("differs"):
            bad.append("oracle differs")
        if r.get("parse") not in ("ok", None, ""):
            bad.append(f"parse={r['parse']}")
        if bad:
            print(f"  {r['doc_id']} [{r['repair_type']}] {'; '.join(bad)}")
            if r.get("check_detail"):
                print(f"      {r['check_detail'][:160]}")

    print("\nno_site answers:")
    for r in rows:
        if r.get("status") == "no_site":
            print(f"  {r['doc_id']} [{r['repair_type']}] {r['reason'][:140]}")

    dev = collections.Counter()
    for r in rows:
        if r.get("status") == "ok":
            for d in (r.get("devices") or "").split("|"):
                dev[(r["repair_type"], d or "(none)")] += 1
    print("\nself-reported devices by type:")
    for t in TYPES:
        items = [(d, n) for (tt, d), n in dev.items() if tt == t]
        if items:
            print(f"  {t:18s} " + ", ".join(f"{d} {n}" for d, n in sorted(items)))

    if args.out:
        fields = list(rows[0].keys())
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
