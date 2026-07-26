#!/usr/bin/env python3
"""
Decide the counting domain for SBN negative indices:
  (A) LINEAR  : -n counts back n concepts in the raw token sequence, boxes ignored
  (B) REGISTER: -n counts back n entries in the *current box's register*
                (Bos 2021 shift/archive/load model), so concepts sitting in a
                box that is not on the current box's load-path do not count.

Strategy: replay every gold SBN, resolve each negative index under both models,
and dump every case where the two disagree.  Any disagreement is a decisive
test case, because the gold annotation can only be right under one of them.
"""
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]        # repo root
GOLD = ROOT / "data/pmb-5.1.0/split/en/train/gold.sbn"
SPEC = ROOT / "data/pmb-5.1.0/src/sbn/sbn_spec.py"
OUT = Path(__file__).resolve().parents[1] / "output"

SYNSET = re.compile(r"^(.+)\.(n|v|a|r|x)\.(\d+)$")
IDX = re.compile(r"^([-+<>])(\d+)$")

NEW_BOX = {
    "ALTERNATION", "ATTRIBUTION", "CONDITION", "CONSEQUENCE", "CONTINUATION",
    "CONTRAST", "EXPLANATION", "NECESSITY", "NEGATION", "POSSIBILITY",
    "PRECONDITION", "RESULT", "SOURCE", "CONJUNCTION", "ELABORATION",
    "CORRECTION",
}
OPERATORS = {
    "TSU", "MOR", "BOT", "TOP", "ESU", "EPR", "EQU", "NEQ", "APX", "LES",
    "LEQ", "TPR", "TAB", "TIN", "SZP", "SZN", "SXP", "SXN", "STI", "STO",
    "SY1", "SY2", "SXY", "ANA", "TCT",
}


def load_roles(spec_path):
    """Pull the ROLES set out of the vendored sbn_spec.py."""
    src = open(spec_path).read()
    # anchor at line start so INVERTIBLE_ROLES doesn't shadow ROLES
    body = re.split(r"^    ROLES = \{", src, flags=re.M)[1].split("}", 1)[0]
    inv = re.split(r"^    INVERTIBLE_ROLES = \{", src, flags=re.M)[1].split("}", 1)[0]
    # "CauserOf" occurs in gold but is absent from the vendored spec's ROLES
    return (set(re.findall(r'"([^"]+)"', body))
            | set(re.findall(r'"([^"]+)"', inv))
            | {"CauserOf"})


def read_records(path):
    """gold.sbn is 3-line records: doc-id / raw sentence / flattened SBN."""
    block = []
    for line in open(path):
        line = line.rstrip("\n")
        if not line.strip():
            if block:
                yield block
                block = []
        else:
            block.append(line)
    if block:
        yield block


def parse(sbn_line, roles):
    """Replay one flattened SBN.

    Returns (concepts, edges, boxes) where
      concepts : list of (position, token, box_id)
      edges    : list of dicts, one per role/operator carrying a numeric index
      boxes    : list of (box_id, parent_box_id, separator_token)
    """
    toks = sbn_line.split()
    concepts = []          # position -> (token, box)
    edges = []
    boxes = [(0, None, "TOP")]
    reg = {0: []}          # box -> ordered list of concept positions in register
    cur_box = 0
    i = 0
    while i < len(toks):
        t = toks[i]
        i += 1

        if t in NEW_BOX:
            if i >= len(toks):
                raise ValueError("separator missing index")
            m = IDX.match(toks[i])
            i += 1
            if not m:
                raise ValueError("bad box index")
            sign, mag = m.group(1), int(m.group(2))
            idx = -mag if sign in "<-" else mag
            new_box = len(boxes)
            # mirrors sbn_smatch.py: parent = active_box + idx + 1
            parent = cur_box + idx + 1
            boxes.append((new_box, parent, t))
            # REGISTER model: the new box loads the parent's archived register
            reg[new_box] = list(reg.get(parent, []))
            cur_box = new_box

        elif t in roles or t in OPERATORS:
            if i >= len(toks):
                raise ValueError("role missing target")
            tgt = toks[i]
            i += 1
            if tgt.startswith('"'):                 # names may contain spaces
                while not tgt.endswith('"') and i < len(toks):
                    tgt = toks[i]
                    i += 1
                continue
            m = IDX.match(tgt)
            if not m:
                continue                            # plain constant (now, speaker, 3, ...)
            sign, mag = m.group(1), int(m.group(2))
            if sign in "<>":                        # SYN_BOX_CONNECT: counts BOXES
                edges.append({"kind": "box_ptr", "role": t, "raw": tgt,
                              "src": len(concepts) - 1, "box": cur_box})
                continue
            n = mag if sign == "+" else -mag
            src = len(concepts) - 1                 # index is relative to the active concept
            edges.append({"kind": "concept_ptr", "role": t, "raw": tgt,
                          "n": n, "src": src, "box": cur_box})

        elif SYNSET.match(t):
            pos = len(concepts)
            concepts.append((pos, t, cur_box))
            reg[cur_box].append(pos)
            # snapshot the register as it stood when this concept became active,
            # so we can resolve its outgoing indices later
            reg.setdefault("snap", {})[pos] = list(reg[cur_box])

        else:
            raise ValueError(f"unexpected token {t!r}")

    return concepts, edges, boxes, reg.get("snap", {})


def main():
    gold, roles = GOLD, load_roles(SPEC)

    stats = Counter()
    diverge = []
    sep_counts = Counter()

    for block in read_records(gold):
        if len(block) < 3:
            stats["malformed_record"] += 1
            continue
        doc, sent, sbn = block[0], block[1].strip(), " ".join(block[2:])
        stats["records"] += 1
        try:
            concepts, edges, boxes, snap = parse(sbn, roles)
        except Exception as e:
            stats["parse_error"] += 1
            if stats["parse_error"] <= 5:
                print(f"  !! {doc}: {e}\n     {sbn[:160]}")
            continue

        if len(boxes) > 1:
            stats["multi_box_records"] += 1
        for _, _, sep in boxes[1:]:
            sep_counts[sep] += 1

        for e in edges:
            if e["kind"] == "box_ptr":
                stats["box_pointer_edges"] += 1
                continue
            stats["concept_ptr_edges"] += 1
            n, src = e["n"], e["src"]
            if n < 0:
                stats["neg_idx"] += 1
            else:
                stats["pos_idx"] += 1

            lin = src + n
            lin_ok = 0 <= lin < len(concepts)

            # REGISTER model: walk n steps inside the register snapshot
            L = snap.get(src, [])
            try:
                here = L.index(src)
            except ValueError:
                here = len(L) - 1
            rpos = here + n
            regt = L[rpos] if 0 <= rpos < len(L) else None

            if n < 0:
                # does this index reach back across a box boundary?
                if lin_ok and concepts[lin][2] != e["box"]:
                    stats["neg_idx_crosses_box"] += 1
                if regt != (lin if lin_ok else None):
                    stats["neg_idx_MODELS_DISAGREE"] += 1
                    diverge.append({
                        "doc": doc, "sent": sent, "sbn": sbn,
                        "role": e["role"], "raw": e["raw"],
                        "src": src, "src_tok": concepts[src][1] if src >= 0 else "?",
                        "src_box": e["box"],
                        "lin": lin, "lin_tok": concepts[lin][1] if lin_ok else "OUT-OF-RANGE",
                        "reg": regt,
                        "reg_tok": concepts[regt][1] if regt is not None else "OUT-OF-RANGE",
                        "concepts": concepts, "boxes": boxes,
                    })

    print("=" * 78)
    print("CORPUS: PMB 5.1.0  en/train/gold.sbn")
    print("=" * 78)
    for k in ("records", "parse_error", "multi_box_records", "concept_ptr_edges",
              "box_pointer_edges", "neg_idx", "pos_idx", "neg_idx_crosses_box",
              "neg_idx_MODELS_DISAGREE"):
        print(f"  {k:34s} {stats[k]:7d}")
    print("\n  separators:", dict(sep_counts.most_common()))
    print(f"\n  DIVERGENT CASES (decisive test set): {len(diverge)}")

    import json
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "divergent_cases.json"
    with open(out, "w") as f:
        json.dump([{k: v for k, v in d.items() if k not in ("concepts", "boxes")}
                   | {"concepts": [list(c) for c in d["concepts"]],
                      "boxes": [list(b) for b in d["boxes"]]}
                   for d in diverge], f, ensure_ascii=False, indent=1)
    print(f"  written -> {out}")


if __name__ == "__main__":
    main()
