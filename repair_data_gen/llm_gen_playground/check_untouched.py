"""Untouched-subgraph check: every gold edge whose endpoints are both away
from the repair site must survive in the repaired graph.

Edges are compared as (source token, label, target token) multisets after
normalising inversions: ROf(y,x) -> R(x,y); symmetric operators get a fixed
direction; converse operators map to one member. Tokens rather than node ids
are compared, so two identical edges elsewhere are counted as a multiset.
"""
import collections
import csv
import sys
from pathlib import Path

ROOT = Path("/Users/hongxuzhou/Documents/GitHub/lct_master_project")
sys.path.insert(0, str(ROOT / "data/pmb-5.1.0/src/sbn"))
sys.path.insert(0, str(ROOT / "repair_data_gen"))
from sbn_smatch import SBNGraph, SBNSource  # noqa: E402
from sbn_spec import SBN_EDGE_TYPE, SBNSpec  # noqa: E402
from sbn_lin import parse_sbn_line  # noqa: E402

SYM = {"EQU", "NEQ", "APX", "SY1"}
CONV = {"TSU": "TPR", "TCT": "TIN", "MOR": "LES", "BOT": "TOP", "SZP": "SZN", "SXN": "SXP"}


def triples(sbn):
    G = SBNGraph(source=SBNSource.INFERENCE).from_string(sbn, is_single_line=True)
    out = collections.Counter()
    for u, v, d in G.edges(data=True):
        if d.get("type") not in (SBN_EDGE_TYPE.ROLE, SBN_EDGE_TYPE.DRS_OPERATOR):
            continue
        a, lab, b = str(G.nodes[u]["token"]), d["token"], str(G.nodes[v]["token"])
        if lab in SBNSpec.INVERTIBLE_ROLES and lab.endswith("Of"):
            a, lab, b = b, lab[:-2], a
        if lab in CONV:
            a, lab, b = b, CONV[lab], a
        if lab in SYM and a > b:
            a, b = b, a
        out[(a, lab, b)] += 1
    return out


def site_tokens(row):
    doc = parse_sbn_line(row["sbn_clean"])
    i = int(row["site_index"])
    toks = {doc.concepts[i].synset}
    if row["repair_type"] == "tense":          # the verb and its tense time.n.08
        toks.add("time.n.08")
    if row["repair_type"] == "forwarding":     # subject, verb, time.n.08
        toks |= {doc.concepts[i + 1].synset, "time.n.08"}
    return toks


def missing_edges(r):
    """Gold edges away from the repair site that the repaired SBN lost."""
    site = site_tokens(r)
    g, rep = triples(r["sbn_clean"]), triples(r["sbn_repair"])
    need = collections.Counter({t: n for t, n in g.items()
                                if t[0] not in site and t[2] not in site})
    return dict(need - rep)


def main():
    rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
    bad = 0
    for r in rows:
        if r.get("status") != "ok" or not r.get("sbn_repair"):
            continue
        try:
            missing = missing_edges(r)
        except Exception as e:  # noqa: BLE001
            print(f"{r['doc_id']}: could not check ({e})")
            continue
        if missing:
            bad += 1
            print(f"{r['doc_id']} [{r['repair_type']}] missing gold edges: {missing}")
            print(f"    gold: {r['sbn_clean']}\n    rep : {r['sbn_repair']}")
    print(f"\n{bad} answered rows lose an untouched gold edge")


if __name__ == "__main__":
    main()
