"""Numbers for train_rsr_base.qmd, printed as Markdown tables.

The report is published as static HTML, so nothing is computed at render
time; run this locally and copy the tables in.

    python3 documentation/data_report/report_numbers.py
"""
import csv
import statistics
import sys
from collections import Counter
from pathlib import Path

GEN = Path(__file__).resolve().parents[2] / "repair_data_gen" / "rsr_generation"
sys.path.insert(0, str(GEN))
from audit import CHECKS, auto_pass  # noqa: E402

RESULTS = GEN / "results"
PREFIX = "meta-models_Muse-Glimmer-30B."
CATEGORIES = ["content_word", "constant_name", "constant_number",
              "constant_temporal", "tense", "forwarding", "edge_label"]

# Plain-language check groups (report §Automatic checks) -> code names.
GROUPS = {
    "well-formed graph": lambda r: any(r.get(k) != "ok" for k in CHECKS)
    or r.get("wn_ok") != "ok",
    "withdrawal gives back gold": lambda r: r.get("withdraw") != "ok"
    or r.get("untouched") != "ok",
    "sentence matches graph": lambda r: r.get("repair_part") != "ok",
    "site fits category": lambda r: r.get("site_ok") != "ok"
    or r.get("place_inside") == "yes" or r.get("pin", "ok") not in ("", "ok"),
    "synonym, not a repair": lambda r: r.get("wn_relation") == "same_synset",
}


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def run_rows(run):
    pin = RESULTS / f"{PREFIX}{run}.checked.pin.tsv"
    return read(pin if pin.exists() else RESULTS / f"{PREFIX}{run}.checked.tsv")


def answered(r):
    return r.get("reply") == "ok" and r.get("status") == "ok" and bool(r.get("sbn_repair"))


def passes(r):
    return auto_pass(r) and r.get("pin", "ok") in ("", "ok")


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "–"


def table(head, rows):
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join("---" if i == 0 else "---:" for i in range(len(head))) + "|")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")
    print()


def pass_rates():
    print("## Pass rates by run and category (answered items)\n")
    runs = ["r2", "train_full", "train_forwarding_extra", "train_temporal_extra",
            "r3_edge", "train_edge", "train_place", "place_r5", "place_r5_rerun"]
    rows = []
    for run in runs:
        rs = run_rows(run)
        for cat in CATEGORIES:
            a = [r for r in rs if r["repair_type"] == cat and answered(r)]
            if a:
                p = sum(passes(r) for r in a)
                declined = sum(r["repair_type"] == cat and not answered(r) for r in rs)
                rows.append((run, cat, len(a), p, pct(p, len(a)), declined))
    table(["run", "category", "answered", "pass", "rate", "not answered"], rows)


def failures():
    rs = [r for r in run_rows("train_full") if answered(r)]
    bad = [r for r in rs if not passes(r)]
    print(f"## Failure reasons, main run (train_full): {len(bad)} of {len(rs)} answered fail\n")
    print("An item can fail more than one group.\n")
    rows = []
    for g, f in GROUPS.items():
        n = sum(f(r) for r in bad)
        only = sum(f(r) and not any(h(r) for k, h in GROUPS.items() if k != g) for r in bad)
        rows.append((g, n, pct(n, len(rs)), only))
    rows.sort(key=lambda x: -x[1])
    table(["check group", "items failing", "of answered", "failing this group only"], rows)

    print("### By category (appendix)\n")
    rows = []
    for cat in CATEGORIES[:-1]:
        c = [r for r in rs if r["repair_type"] == cat]
        rows.append([cat, len(c)] + [sum(f(r) for r in c if not passes(r)) for f in GROUPS.values()])
    table(["category", "answered"] + list(GROUPS), rows)

    print("### Detail of 'sentence matches graph' failures\n")
    det = Counter(r["repair_part"] for r in bad if r["repair_part"] != "ok")
    table(["reason", "items"], det.most_common())


def audits():
    print("## Human audits by category (yes / (yes + no); unsure apart)\n")
    rows = []
    for d in ["r2", "train_full", "train_temporal_extra", "train_forwarding_reaudit",
              "r3_edge"]:
        rs = read(GEN / "audit" / d / "merged.tsv")
        for cat in CATEGORIES:
            c = [r for r in rs if r["repair_type"] == cat]
            if not c:
                continue
            cell = []
            for dim in ("meaning", "natural"):
                k = Counter(r[dim] for r in c)
                cell += [f"{k['yes']}/{k['yes'] + k['no']}", k["unsure"]]
            blank = sum(not r["meaning"] and not r["natural"] for r in c)
            rows.append([d, cat, len(c)] + cell + [blank])
    table(["audit", "category", "items", "meaning yes", "meaning unsure",
           "natural yes", "natural unsure", "unlabelled"], rows)


def dataset():
    rs = read(GEN / "dataset" / "train_rsr.base.tsv")
    print(f"## Final set: {len(rs)} items\n")
    rows = []
    for cat in CATEGORIES:
        c = [r for r in rs if r["repair_type"] == cat]
        words = [len(r["nl_repair"].split()) for r in c]
        docs = Counter(r["doc_id"] for r in c)
        rows.append((cat, len(c), len(docs), sum(v > 1 for v in docs.values()),
                     f"{statistics.median(words):g}", min(words), max(words),
                     sum(bool(r["ambiguous"]) for r in c)))
    words = [len(r["nl_repair"].split()) for r in rs]
    docs = Counter(r["doc_id"] for r in rs)
    rows.append(("all", len(rs), len(docs), sum(v > 1 for v in docs.values()),
                 f"{statistics.median(words):g}", min(words), max(words),
                 sum(bool(r["ambiguous"]) for r in rs)))
    table(["category", "items", "documents", "documents with 2+ items",
           "median words", "min", "max", "ambiguous"], rows)

    print("Ambiguity labels: " + ", ".join(
        f"{k} {v}" for k, v in Counter(r["ambiguous"] for r in rs if r["ambiguous"]).most_common()))
    multi = Counter(len(set(r["repair_type"] for r in rs if r["doc_id"] == d))
                    for d in docs)
    print("\nDocuments by number of categories they serve: "
          + ", ".join(f"{k}: {v}" for k, v in sorted(multi.items())))
    print("\nLabel source: " + ", ".join(
        f"{k} {v}" for k, v in Counter(r["label_source"] for r in rs).most_common()))


if __name__ == "__main__":
    pass_rates()
    failures()
    audits()
    dataset()
