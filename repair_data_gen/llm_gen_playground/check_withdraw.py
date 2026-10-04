"""Withdraw check: remove the reparandum and the gold SBN must come back.

CORRECTION withdraws its box. So the repaired SBN, with the CORRECTION box
and every edge into it removed and the dummies merged back, must have the
same edges as the gold SBN: none missing, none added. Unlike
check_untouched.py, no edge at the repair site is exempt. This catches an
index copied from gold that now lands on another concept after the
insertion (`spread, buttered`: `butter.v.01 Agent -2` hits `sister.n.01`
instead of `person.n.01`), a role renamed in the repair box, or an edge that
was dropped from the repair box.

Dummies are merged like this:
- ③ an `entity.n.01` in the repair box whose `EQU` points to a verb is the
  event (edge label, repair_notation §4.4);
- ⑤ the anchor is the `entity.n.01` right before the CORRECTION box, when the
  SBN has more `entity.n.01` outside the reparandum box than gold has. It is
  merged into the repair-box concept that points to it with `EQU`.

Edges are compared as (source, label, target) multisets after normalising
inversions, as in check_untouched.py. A node is its synset plus its
constant-valued roles (`person.n.01[EQU=speaker]`, `time.n.08[TPR=now]`),
so an index that moves to another concept of the same synset still shows
("Jean Reno is my favorite actor": `Co-Theme` to the speaker instead of the
actor).

Box structure is compared too: each box operator other than the pair
(`NEGATION <1`, `CONTRAST <1`, …) and each box-valued role (`Proposition
>1`) is an edge between two boxes, a box named by the concepts in it, with
the CONJUNCTION box counted as part of its parent. Inserting the two boxes
renumbers every later box pointer, and a wrong one that makes no cycle
passes every other check.

    python3 check_withdraw.py results/<model>.tsv

`diff(row)` is also called by analyse_run.py, which writes the `withdraw`
column.
"""
import collections
import csv
import re
import sys
from pathlib import Path

ROOT = Path("/Users/hongxuzhou/Documents/GitHub/lct_master_project")
sys.path.insert(0, str(ROOT / "data/pmb-5.1.0/src/sbn"))
sys.path.insert(0, str(ROOT / "repair_data_gen"))
from sbn_spec import SBNSpec  # noqa: E402
from sbn_lin import parse_sbn_line  # noqa: E402

IDX = re.compile(r"^[+-]\d+$")
SYM = {"EQU", "NEQ", "APX", "SY1"}
CONV = {"TSU": "TPR", "TCT": "TIN", "MOR": "LES", "BOT": "TOP", "SZP": "SZN", "SXN": "SXP"}
DUMMY = "entity.n.01"


def norm(a, lab, b):
    if lab in SBNSpec.INVERTIBLE_ROLES and lab.endswith("Of") and lab != "Of":
        a, lab, b = b, lab[:-2], a
    if lab in CONV:
        a, lab, b = b, CONV[lab], a
    if lab in SYM and a > b:
        a, b = b, a
    return (a, lab, b)


BOXVAL = re.compile(r"^[<>]\d+$")   # a box-valued role (`Proposition >1`)


def edges(doc, drop_box=None, merge=None, fix=None, parent=None):
    """Edge multiset, without concepts in drop_box, with merged positions.

    Box structure is part of it: every box operator other than the
    CORRECTION/CONJUNCTION pair, and every box-valued role, becomes an edge
    between boxes, each box named by its concepts. The CONJUNCTION box
    (`fix`) is read as part of its parent box, since it is asserted there.
    """
    merge = merge or {}

    def tok(pos):
        c = doc.concepts[merge.get(pos, pos)]
        consts = sorted(f"{r}={v}" for r, v in c.roles
                        if not IDX.match(v) and not BOXVAL.match(v))
        return c.synset + ("[" + ",".join(consts) + "]" if consts else "")

    def box(b):
        return parent if b == fix and fix is not None else b

    content = collections.defaultdict(list)
    for c in doc.concepts:
        if c.box != drop_box and c.pos not in merge:
            content[box(c.box)].append(tok(c.pos))
    name = lambda b: "{" + " ".join(sorted(content.get(box(b), []))) + "}"  # noqa: E731

    out = collections.Counter()
    for sep in doc.separators:
        if sep.new_box in (drop_box, fix) or sep.target_box is None:
            continue
        out[(name(sep.target_box), sep.name, name(sep.new_box))] += 1
    for c in doc.concepts:
        if c.box == drop_box:
            continue
        for r, v in c.roles:
            if BOXVAL.match(v):
                b = c.box + int(v[1:]) if v[0] == ">" else c.box - int(v[1:])
                out[(tok(c.pos), r, name(b))] += 1
                continue
            if not IDX.match(v):
                out[norm(tok(c.pos), r, v)] += 1
                continue
            t = c.pos + int(v)
            if doc.concepts[t].box == drop_box:
                continue
            if merge.get(c.pos, c.pos) == merge.get(t, t):   # a dummy's own EQU
                continue
            out[norm(tok(c.pos), r, tok(t))] += 1
    return out


def withdrawn(sbn, gold):
    """Edges of the repaired SBN with the reparandum removed, dummies merged.
    Raises ValueError without a CORRECTION box followed by CONJUNCTION."""
    doc = parse_sbn_line(sbn)
    corr = next((s for s in doc.separators if s.name == "CORRECTION"), None)
    conj = corr and next((s for s in doc.separators
                          if s.name == "CONJUNCTION" and s.new_box == corr.new_box + 1), None)
    if conj is None:
        raise ValueError("no CORRECTION box followed by a CONJUNCTION box")
    rep, fix = corr.new_box, conj.new_box
    first_rep = min((c.pos for c in doc.concepts if c.box == rep), default=None)
    if first_rep is None:
        raise ValueError("empty reparandum box")
    g_dummies = sum(c.synset == DUMMY for c in parse_sbn_line(gold).concepts)
    surplus = sum(c.synset == DUMMY for c in doc.concepts if c.box != rep) - g_dummies
    anchor = max((c.pos for c in doc.concepts if c.box not in (rep, fix) and c.pos < first_rep),
                  default=None)
    merge = {}
    for c in doc.concepts:
        if c.synset != DUMMY:
            continue
        equ = [c.pos + int(v) for r, v in c.roles if r == "EQU" and IDX.match(v)]
        if c.box == fix and equ and doc.concepts[equ[0]].pos_tag == "v":           # ③
            merge[c.pos] = equ[0]
        elif c.pos == anchor and surplus > 0:                                       # ⑤
            q = [d.pos for d in doc.concepts if d.box == fix and
                 any(r == "EQU" and IDX.match(v) and d.pos + int(v) == c.pos for r, v in d.roles)]
            if q:
                merge[c.pos] = q[0]
    return edges(doc, drop_box=rep, merge=merge, fix=fix, parent=conj.target_box)


def diff(row):
    """(gold edges missing, edges added) after withdrawing the reparandum."""
    g = edges(parse_sbn_line(row["sbn_clean"]))
    w = withdrawn(row["sbn_repair"], row["sbn_clean"])
    return dict(g - w), dict(w - g)


def main():
    rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
    bad = 0
    for r in rows:
        if r.get("status") != "ok" or not r.get("sbn_repair"):
            continue
        try:
            missing, added = diff(r)
        except ValueError as e:
            missing, added = {"error": str(e)}, {}
        if missing or added:
            bad += 1
            print(f"{r['doc_id']} [{r['repair_type']}] missing {missing}; added {added}")
            print(f"    gold: {r['sbn_clean']}\n    rep : {r['sbn_repair']}")
    print(f"\n{bad} answered rows do not give back the gold SBN")


if __name__ == "__main__":
    main()
