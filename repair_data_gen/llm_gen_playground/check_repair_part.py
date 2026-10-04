"""Checks on the repaired part itself (R5, 'something changed', sentence).

    python3 check_repair_part.py results/<model>.tsv

`problems(row)` is also called by analyse_run.py, which writes the result
to the `repair_part` column.
"""
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
CONTRACTIONS = [(r"(?i)\bcan't\b", "can not"), (r"(?i)\bwon't\b", "will not"),
                (r"n't\b", " not"), (r"'re\b", " are"), (r"'m\b", " am"),
                (r"'ve\b", " have"), (r"'ll\b", " will"), (r"'d\b", " would"),
                (r"'s\b", " is")]


def boxes(doc):
    """(reparandum box, repair box), or None without a CORRECTION box
    followed directly by a CONJUNCTION box."""
    corr = next((s for s in doc.separators if s.name == "CORRECTION"), None)
    conj = corr and next((s for s in doc.separators
                          if s.name == "CONJUNCTION" and s.new_box == corr.new_box + 1), None)
    if conj is None:
        return None
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


def norm(text):
    """Surface normalisation for the sentence checks: no quote marks, and
    contractions written out, so "He's" and "He is" compare equal. A
    possessive 's is expanded too; both sides get the same treatment."""
    text = text.replace('"', "").strip()
    for a, b in CONTRACTIONS:
        text = re.sub(a, b, text)
    return text


def problems(r):
    """Issues in the repaired part of one answered row ([] if none)."""
    probs = []
    doc = parse_sbn_line(strip_comments(r["sbn_repair"]))
    bx = boxes(doc)
    if bx is None:
        return ["no CORRECTION box followed by a CONJUNCTION box"]
    rep, fix = bx
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
        # The reparandum part (the text before the join, less the gold
        # prefix) contains the reported reparandum. It need not end with it:
        # "He dug, They dug …" (forwarding), "was from the table, on the
        # table" (edge label). The surface may also be wider than the part
        # when it includes gold text before it ("October the fourth").
        surf = [w.lower() for w in WORD.findall(norm(r["reparandum_surface"]))]
        gold, right_n = norm(r["nl_clean"]), norm(right)
        prefix = gold[:max(len(gold) - len(right_n), 0)].rstrip()
        part = norm(left)[len(prefix):] if norm(left).startswith(prefix) else norm(left)
        words = [w.lower() for w in WORD.findall(part)]
        sub = lambda a, b: any(b[i:i + len(a)] == a for i in range(len(b) - len(a) + 1))  # noqa: E731
        # either way round: "October the fourth" reported, "the fourth" repeated
        if not (sub(surf, words) or sub(words, surf)):
            probs.append(f"onset: reparandum part {part.strip()!r} does not "
                         f"contain reparandum_surface {r['reparandum_surface']!r}")
        # Rest of the sentence unchanged (task.md step 4): the repair part is
        # the end of the gold sentence, and the gold text before it opens the
        # reparandum part. Only the first letter may change case ("The fly"
        # -> "the fly"). Both sides go through norm(), so quote marks and
        # written-out contractions ("He's" -> "He is") are not differences.
        gold = norm(r["nl_clean"])
        left, right = norm(left), norm(right)
        cut = len(gold) - len(right)
        if cut < 0 or gold[cut:].lower() != right.lower() or not left.startswith(gold[:cut].rstrip()):
            probs.append("rest of the sentence changed: repair part is not the "
                         "end of the gold sentence")
    if not in_order([w.lower() for w in WORD.findall(norm(r["nl_clean"]))],
                    [w.lower() for w in WORD.findall(norm(nl))]):
        probs.append("gold words not kept in order")
    return probs


def main():
    rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
    issues = 0
    for r in rows:
        if r.get("status") != "ok" or not r.get("sbn_repair"):
            continue
        probs = problems(r)
        if probs:
            issues += 1
            print(f"{r['doc_id']} [{r['repair_type']}] {'; '.join(probs)}")
            print(f"    {r['nl_clean']}  ->  {r['nl_repair']}")
            print(f"    {r['sbn_repair']}")
    print(f"\n{issues} answered rows with an issue in the repaired part")


if __name__ == "__main__":
    main()
