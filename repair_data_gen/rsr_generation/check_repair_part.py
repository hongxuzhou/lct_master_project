"""Checks on the repaired part itself (R5, 'something changed', place
overlap for edge label, sentence).

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


def shape(cs):
    """Each concept's synset and roles, with index targets resolved: a
    target inside the box by its place in the box, one outside by its
    position in the document. Two boxes with the same shape say the same."""
    pos = [c.pos for c in cs]
    out = []
    for c in cs:
        roles = []
        for r, v in c.roles:
            if re.match(r"^[+-]\d+$", v):
                t = c.pos + int(v)
                v = ("in", pos.index(t)) if t in pos else ("out", t)
            roles.append((r, v))
        out.append((c.synset, sorted(roles, key=str)))
    return out


SPATIAL_OPS = {"SZN", "SZP", "SXP", "SXN", "APX", "SY1", "STI", "STO", "SXY"}
IN_WORDS = {"in", "inside", "into"}


def place(doc, cs):
    """The place a box gives (edge label, third kind): the operator of a
    `location.n.01` its Location/Destination points to, "plain" for a role
    straight to the thing, None if the box has no such role."""
    for c in cs:
        for r, v in c.roles:
            if r in ("Location", "Destination") and re.match(r"^[+-]\d+$", v):
                t = doc.concepts[c.pos + int(v)]
                if t.synset != "location.n.01":
                    return "plain"
                return next((rr for rr, _ in t.roles if rr in SPATIAL_OPS), None)
    return None


def in_order(sub, words):
    it = iter(words)
    return all(w in it for w in sub)


def norm(text):
    """Surface normalisation for the sentence checks: no quote marks, no
    final punctuation, plain apostrophes, and contractions written out, so "He's" and "He is" compare equal. A
    possessive 's is expanded too; both sides get the same treatment."""
    # typographic apostrophes and quotes as plain ones ("That’s" = "That's"),
    # and sentence-final punctuation ignored ("slicing tofu" = "slicing tofu.")
    text = text.translate(str.maketrans("’‘“”", "''\"\""))
    text = text.replace('"', "").strip()
    text = re.sub(r"\s*[.?!]+$", "", text)
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
    # compare the reparandum box with the same number of leading repair-box
    # concepts. Targets count: "on, under the table" has the same roles on
    # both dummies, but one points to the table and one to location.n.01
    if shape(rep) == shape(fix)[:len(rep)]:
        probs.append("reparandum identical to repair")
    nl = r["nl_repair"]
    # Edge label, place: one place must not include the other (§4.4).
    # "on/in/at" against each other is caught above (both plain).
    if r.get("repair_type") == "edge_label":
        a, b = place(doc, rep), place(doc, fix)
        if {a, b} == {"APX", "SY1"}:
            probs.append("edge label: 'near' and 'beside' overlap (APX, SY1)")
        if {a, b} == {"plain", "STI"}:
            words = {w.lower() for w in WORD.findall(r.get("reparandum_surface", ""))}
            onset = r.get("repair_onset")
            first = WORD.findall(nl[int(onset):])[:1] if onset else []
            if words & IN_WORDS and {w.lower() for w in first} & IN_WORDS:
                probs.append("edge label: 'in' and 'inside' overlap (plain, STI)")
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
        # the gold prefix is known only when the repair part ends the gold
        # sentence; otherwise read the whole left side (the rest-of-sentence
        # check below reports the mismatch)
        fits = gold.lower().endswith(right_n.lower()) and len(right_n) <= len(gold)
        prefix = gold[:len(gold) - len(right_n)].rstrip() if fits else ""
        part = norm(left)[len(prefix):] if prefix and norm(left).startswith(prefix) else norm(left)
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
