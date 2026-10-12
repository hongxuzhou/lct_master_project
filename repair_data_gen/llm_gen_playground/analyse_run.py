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
- `untouched`: `ok`, or the gold edges away from the repair site that the
  repaired SBN lost (check_untouched.missing_edges); `error: …` if the
  check cannot run (e.g. a wrong self-reported site_index);
- `repair_part`: `ok`, or the issues check_repair_part.problems finds in
  the repaired part and the sentence (R5, empty reparandum, rest of the
  sentence changed, …);
- `withdraw`: `ok`, or how the repaired SBN differs from gold once the
  CORRECTION box is removed and the dummies are merged
  (check_withdraw.diff); stricter than `untouched` at the repair site;
- `place_inside`: for name repairs, `yes` when WordNet places the
  reparandum inside the repair ("Osaka, Japan": part holonyms), which reads
  as an address, not a repair; counts as a failure. `no` otherwise, `n/a`
  when WordNet has no place of that name (small places are not covered);
- `site_ok`: `ok` if the model's `site_index` is one of the sites
  select_docs.py offers for this type, else `outside`. Selection only makes
  sure a document has a good site; the model picks among all of them;
- `ambiguous`: why the repair, read without an editing expression, has a
  fluent reading (`number_range`: "two, three brothers", "10, 15 years";
  `initial_name`: "Jim, Tom broke it", read as a vocative), or empty. Not a
  failure: such items go to the marked share of the training data and are
  reported separately at test time (decided 2026-10-08). Apposition and
  lists are not detected yet;
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
import check_repair_part  # noqa: E402
import check_untouched  # noqa: E402
import check_withdraw  # noqa: E402
import select_docs  # noqa: E402

CONCEPT = re.compile(r"^[^\s\"]+\.[nvar]\.\d\d$")
CHECKS = ("parse", "dag", "index", "magnitude", "edges", "boxes")
EXTRA = ("untouched", "repair_part", "withdraw", "site_ok")   # ok, or what is wrong
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


def wn_relation(row: dict) -> str:
    """How the reparandum relates to the repair in WordNet (content word only).

    `same_synset` is a synonym, which task.md rules out: it counts as a
    failure. `hypernym` / `hyponym` (the repair is more general / more
    specific than the reparandum) and `similar_to` (adjective clusters) are
    recorded only: correcting to a more exact or more general word is a
    natural repair (decided 2026-10-02). `other` is anything further apart.
    """
    if row["repair_type"] != "content_word":
        return ""
    syn = []
    for name in (row["reparandum_synset"], row["repair_synset"]):
        try:
            lemma, pos, num = name.rsplit(".", 2)
        except ValueError:
            return "n/a"
        syn.append(wn_lookup._exact(lemma, pos, num))
    rep, fix = syn
    if rep is None or fix is None:
        return "n/a"
    if rep == fix:
        return "same_synset"
    up = lambda s: set(s.hypernyms() + s.instance_hypernyms())  # noqa: E731
    if fix in up(rep):
        return "hypernym"
    if rep in up(fix):
        return "hyponym"
    if fix in rep.similar_tos() or rep in fix.similar_tos():
        return "similar_to"
    return "other"


def _places(name: str):
    from nltk.corpus import wordnet as wn
    return [s for s in wn.synsets(name.strip('"').replace(" ", "_"), "n")
            if s.instance_hypernyms()]


def place_inside(row: dict) -> str:
    """Is the reparandum place part of the repair place? (decided 2026-10-04:
    "small place, large place" is a fluent address, even before any
    interregnum is inserted)."""
    if row["repair_type"] != "constant_name":
        return ""
    try:
        site = parse_sbn_line(row["sbn_clean"]).concepts[int(row["site_index"])]
        gold = next(v for r, v in site.roles if r == "Name")
    except (StopIteration, IndexError, ValueError):
        return "n/a"
    inner, outer = _places(row["reparandum_value"]), set(_places(gold))
    if not inner or not outer:
        return "n/a"
    hit = any(outer & set(s.closure(lambda x: x.part_holonyms())) for s in inner)
    return "yes" if hit else "no"


def site_ok(row: dict) -> str:
    try:
        doc = parse_sbn_line(row["sbn_clean"])
        doc.sentence = row["nl_clean"]
        fn = select_docs.SITES[row["repair_type"]]
        if row["repair_type"] == "edge_label":
            # the kind of repair picks the rule: a place answer gives an
            # operator or "plain" (task.md), a direction answer a role; "There's
            # a cat from, on the desk" is a place site but not a direction one
            place = row.get("reparandum_value", "") in select_docs.PLACE_OPS \
                or row.get("reparandum_value", "") == "plain"
            fn = select_docs.site_place if place else select_docs.site_direction
        sites = {p for p, _ in fn(doc)}
        return "ok" if int(row["site_index"]) in sites else "outside"
    except Exception as e:  # noqa: BLE001
        return f"error: {type(e).__name__}"


_NUM_ROLES = ("EQU", "Quantity", "APX")
_WEEKDAY = re.compile(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?")


def _box_numbers(concepts) -> list[float]:
    out = []
    for c in concepts:
        for r, v in c.roles:
            if r in _NUM_ROLES:
                try:
                    out.append(float(v))
                except ValueError:
                    pass
    return out


def ambiguous(row: dict) -> str:
    """A fluent reading of the base form (no editing expression), or "".

    number_range: reparandum and repair ascend like an approximate range,
    "n, n+1" up to 20 ("two, three brothers") or two multiples of 5 no
    further apart than the smaller ("10, 15 years", "five, ten dollars").
    initial_name: a name repair at the very start of the sentence, which
    reads as someone being addressed ("Jim, Tom broke it").
    weekday_list: a weekday repair whose repair part starts with the weekday
    itself, so nothing before it is said again; it reads as a list ("every
    Saturday, Sunday", "I hate Wednesdays, Mondays"). With a word repeated
    ("on Tuesday, on Monday", "next Monday, next Friday") it does not.
    Added 2026-10-11 after the temporal-extra audit (a0008).
    """
    t = row["repair_type"]
    try:
        if t == "constant_temporal" and row.get("repair_onset"):
            bx = check_repair_part.boxes(parse_sbn_line(strip_comments(row["sbn_repair"])))
            if bx and any(r == "DayOfWeek" for c in bx[0] for r, _ in c.roles):
                right = row["nl_repair"][int(row["repair_onset"]):].split()
                if right and _WEEKDAY.fullmatch(right[0].strip(",.?!").lower()):
                    return "weekday_list"
        if t == "constant_number":
            bx = check_repair_part.boxes(parse_sbn_line(strip_comments(row["sbn_repair"])))
            if not bx:
                return ""
            for a in _box_numbers(bx[0]):
                for g in _box_numbers(bx[1]):
                    if a < g and ((g - a == 1 and a <= 20)
                                  or (a % 5 == 0 and g % 5 == 0 and g - a <= a)):
                        return "number_range"
        if t == "constant_name" and row.get("repair_onset"):
            right = row["nl_repair"][int(row["repair_onset"]):]
            norm = check_repair_part.norm
            if norm(right).lower() == norm(row["nl_clean"]).lower():
                return "initial_name"
    except Exception:  # noqa: BLE001
        return ""
    return ""


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
        # The run TSV's `parse` (did the reply parse) shares its name with the
        # SBN reader check `parse` written below; keep the former apart.
        r["reply"] = r.get("parse", "")
        answered = r["reply"] == "ok" and r.get("status") == "ok" and r.get("sbn_repair")
        if not answered:
            for k in CHECKS + EXTRA + ("wn_ok", "wn_missing", "wn_relation", "oracle",
                                       "check_detail", "place_inside", "ambiguous"):
                r[k] = ""
            continue
        r["wn_relation"] = wn_relation(r)
        res = check(r["sbn_repair"])
        for k in CHECKS:
            r[k] = res[k] or ""
        r["check_detail"] = res["detail"]
        r["wn_ok"], r["wn_missing"] = new_concepts_exist(r["sbn_clean"], r["sbn_repair"])
        r["oracle"] = oracle(r)
        r["place_inside"] = place_inside(r)
        r["site_ok"] = site_ok(r)
        r["ambiguous"] = ambiguous(r)
        try:
            lost = check_untouched.missing_edges(r)
            r["untouched"] = "ok" if not lost else "lost " + "; ".join(" ".join(t) for t in lost)
        except Exception as e:  # noqa: BLE001
            r["untouched"] = f"error: {type(e).__name__}: {e}"
        try:
            probs = check_repair_part.problems(r)
            r["repair_part"] = "ok" if not probs else "; ".join(probs)
        except Exception as e:  # noqa: BLE001
            r["repair_part"] = f"error: {type(e).__name__}: {e}"
        try:
            missing, added = check_withdraw.diff(r)
            fmt = lambda d: "; ".join(" ".join(t) for t in d)  # noqa: E731
            r["withdraw"] = ("ok" if not (missing or added) else
                             f"missing {fmt(missing)} | added {fmt(added)}")
        except Exception as e:  # noqa: BLE001
            r["withdraw"] = f"error: {type(e).__name__}: {e}"

    by = collections.defaultdict(list)
    for r in rows:
        by[r["repair_type"]].append(r)
    head = ("type", "n", "json", "no_site", "answered", "struct_ok", "wn_ok",
            "untouched_ok", "repair_part_ok", "withdraw_ok", "oracle_same",
            "median_lookups")
    print("  ".join(f"{h:>15s}" if k else f"{h:18s}" for k, h in enumerate(head)))
    for t in TYPES + ("ALL",):
        rs = rows if t == "ALL" else by.get(t, [])
        if not rs:
            continue
        ans = [r for r in rs if r["reply"] == "ok" and r.get("status") == "ok"]
        struct = [r for r in ans if all(r[k] == "ok" for k in CHECKS)]
        cw = [r for r in ans if r["repair_type"] == "content_word"]
        same = sum(r["oracle"] == "same" for r in cw)
        look = sorted(int(r["lookups"]) for r in rs if r["lookups"])
        cells = [t, len(rs),
                 sum(r["reply"] == "ok" for r in rs),
                 sum(r.get("status") == "no_site" for r in rs),
                 len(ans), len(struct),
                 sum(r["wn_ok"] == "ok" for r in ans),
                 sum(r["untouched"] == "ok" for r in ans),
                 sum(r["repair_part"] == "ok" for r in ans),
                 sum(r["withdraw"] == "ok" for r in ans),
                 f"{same}/{len(cw)}" if cw else "-",
                 look[len(look) // 2] if look else "-"]
        print("  ".join(f"{str(c):>15s}" if k else f"{str(c):18s}" for k, c in enumerate(cells)))

    print("\nfailures (rows without an answer, or answered rows failing a check):")
    for r in rows:
        bad = [f"{k}={r[k]}" for k in CHECKS if r.get(k) not in ("", "ok")]
        if r.get("wn_ok") == "missing":
            bad.append(f"not in WordNet: {r['wn_missing']}")
        bad += [f"{k}: {r[k]}" for k in EXTRA if r.get(k) not in ("", "ok")]
        if r.get("oracle", "").startswith("differs"):
            bad.append("oracle differs")
        if r.get("place_inside") == "yes":
            bad.append(f"place inside: {r['reparandum_value']} is part of the gold place")
        if r.get("wn_relation") == "same_synset":
            bad.append(f"synonym: {r['reparandum_synset']} is the same synset as {r['repair_synset']}")
        if r["reply"] != "ok":
            bad.append(f"no answer: {r['reply']}")
        if bad:
            print(f"  {r['doc_id']} [{r['repair_type']}] {'; '.join(bad)}")
            if r.get("check_detail"):
                print(f"      {r['check_detail'][:160]}")

    rel = collections.Counter(r["wn_relation"] for r in rows if r.get("wn_relation"))
    if rel:
        print("\ncontent word, reparandum -> repair in WordNet (same_synset fails; the rest recorded):")
        print("  " + ", ".join(f"{k} {n}" for k, n in rel.most_common()))

    amb = collections.Counter((r["repair_type"], r["ambiguous"]) for r in rows if r.get("ambiguous"))
    if amb:
        print("\nfluent reading without an editing expression (not a failure; marked share):")
        print("  " + ", ".join(f"{t} {k} {n}" for (t, k), n in sorted(amb.items())))

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
