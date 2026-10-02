"""Checks on the repaired part itself (R5, 'something changed', sentence)."""
import csv
import re
import sys
from pathlib import Path

ROOT = Path("/Users/hongxuzhou/Documents/GitHub/lct_master_project")
sys.path.insert(0, str(ROOT / "repair_data_gen"))
from sbn_lin import parse_sbn_line  # noqa: E402
from validate_with_pmb_parser import strip_comments  # noqa: E402

INTER = ["I mean", "no wait", "no", "actually"]
WORD = re.compile(r"[A-Za-z0-9']+")


def boxes(doc):
    corr = next(s for s in doc.separators if s.name == "CORRECTION")
    conj = next(s for s in doc.separators if s.name == "CONJUNCTION" and s.new_box == corr.new_box + 1)
    rep = [c for c in doc.concepts if c.box == corr.new_box]
    fix = [c for c in doc.concepts if c.box == conj.new_box]
    return rep, fix


def hooks(doc, c, others_pos):
    """Edges of c to concepts outside its own list (both directions)."""
    n = 0
    for r, v in c.roles:
        if re.match(r"^[+-]\d+$", v) and (c.pos + int(v)) not in others_pos:
            n += 1
    for d in doc.concepts:
        if d.pos in others_pos:
            continue
        for r, v in d.roles:
            if re.match(r"^[+-]\d+$", v) and d.pos + int(v) == c.pos:
                n += 1
    return n


def in_order(sub, words):
    it = iter(words)
    return all(w in it for w in sub)


rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
issues = 0
for r in rows:
    if r.get("status") != "ok":
        continue
    probs = []
    doc = parse_sbn_line(strip_comments(r["sbn_repair"]))
    rep, fix = boxes(doc)
    rep_pos = {c.pos for c in rep}
    if not any(hooks(doc, c, rep_pos) for c in rep):
        probs.append("R5: reparandum box has no edge to the rest")
    sig = lambda cs: [(c.synset, sorted((rr, vv) for rr, vv in c.roles if not re.match(r"^[+-]\d+$", vv))) for c in cs]
    # compare the reparandum box with the same number of leading repair-box concepts
    if sig(rep) == sig(fix[:len(rep)]) and [ [x for x,_ in c.roles] for c in rep] == [[x for x,_ in c.roles] for c in fix[:len(rep)]]:
        probs.append("reparandum identical to repair")
    nl = r["nl_repair"]
    inter = r.get("interregnum", "none")
    has = [m for m in INTER if re.search(rf"(^|[ ,]){re.escape(m)}([ ,]|$)", nl, re.I)]
    if inter != "none" and inter.lower() not in [h.lower() for h in has]:
        probs.append(f"interregnum {inter!r} missing")
    if inter == "none" and has and not any(w in r["nl_clean"].lower().split() for w in [h.lower() for h in has]):
        probs.append(f"unrequested interregnum {has}")
    # Repair onset (runs with nl_left/nl_right only): the join is a single
    # ", ", and the text before it ends with the reported reparandum.
    if r.get("repair_onset"):
        k = int(r["repair_onset"])
        left, right = nl[:k - 2], nl[k:]
        if left.endswith(",") or right.startswith(","):
            probs.append("onset: extra comma at the join")
        surf = r["reparandum_surface"].strip()
        if not left.rstrip(",").lower().endswith(surf.lower()):
            probs.append(f"onset: text before the join does not end with "
                         f"reparandum_surface {surf!r}")
        # Rest of the sentence unchanged (task.md step 4): the repair part is
        # the end of the gold sentence, and the gold text before it opens the
        # reparandum part. Only the first letter may change case ("The fly"
        # -> "the fly"). Contractions written out ("Tom's" -> "Tom is") are
        # flagged too; read those by hand.
        gold = r["nl_clean"].strip()
        cut = len(gold) - len(right)
        if cut < 0 or gold[cut:].lower() != right.lower() or not left.startswith(gold[:cut].rstrip()):
            probs.append("rest of the sentence changed: repair part is not the "
                         "end of the gold sentence")
    if not in_order([w.lower() for w in WORD.findall(r["nl_clean"])], [w.lower() for w in WORD.findall(nl)]):
        probs.append("gold words not kept in order")
    if probs:
        issues += 1
        print(f"{r['doc_id']} [{r['repair_type']}] {'; '.join(probs)}")
        print(f"    {r['nl_clean']}  ->  {nl}")
        print(f"    {r['sbn_repair']}")
print(f"\n{issues} answered rows with an issue in the repaired part")
