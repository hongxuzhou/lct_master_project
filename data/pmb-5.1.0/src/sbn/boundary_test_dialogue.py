"""
Boundary test: cross-sentential / discourse-level CORRECTION.

Input is the user's Joe/Ferrari/Jaguar/wedding dialogue, which contains
BOTH an intra-sentence repair (Ferrari -> Jaguar) AND a discourse-level
repair (A's utterance -> B's utterance).

The discourse-level CORRECTION has no sibling CONJUNCTION:
  - intra-sentence pattern : (parent) -[CORRECTION]-> reparandum-box
                             (parent) -[CONJUNCTION]-> repair-box
  - discourse-level pattern: (parent) -[CORRECTION]-> repair-box
                             (no sibling box; preceding content is reparandum)

Polarity is INVERTED: in discourse-level, the CORRECTION-box content is
what SURVIVES; the matrix/preceding content is the discarded reparandum.

What we want to learn:
  1. Does the parser accept the dialogue at all?
  2. What graph topology does it produce?
  3. What does project_repair currently do to it?
  4. Where does it land relative to the truth-value-equivalent "final" SBN
     (= B's utterance with pronouns resolved to A's referents)?
"""
from __future__ import annotations
import sys
from copy import deepcopy

try:
    from sbn_smatch import SBNGraph
    from sbn_spec import SBN_NODE_TYPE, SBN_EDGE_TYPE, SBNError
    from smatch import score_amr_pairs
    from project_repair import project_repair, find_repair_pairs
except ImportError as e:
    sys.exit(f"Import error: {e}\nRun from .../pmb-5.1.0/src/sbn/")


DIALOGUE_SBN = (
    # A's utterance
    "person.n.01 Name \"Joe\" "
    "time.n.01 TPR now "
    "buy.v.01 Agent -2 Time -1 Theme +1 Manner +3 "
    "CORRECTION <1 "
    "car.n.01 Name \"Ferrari\" "
    "CONJUNCTION <2 "
    "car.n.01 Name \"Jaguar\" ThemeOf -2 "
    "gift.n.01 PartOf \"birthday\" MannerOf -3 "
    # Discourse-level CORRECTION (no paired CONJUNCTION)
    "CORRECTION <1 "
    # B's utterance, inside the discourse-level CORRECTION box
    "male.n.01 EQU -6 "
    "buy.v.01 Agent -1 Theme -3 Manner +1 "
    "gift.n.01 PartOf \"wedding\""
)

# Truth-value target: B's claim with pronouns resolved.
#   "Joe bought a Jaguar as a wedding gift."
TARGET_FINAL_SBN = (
    "person.n.01 Name \"Joe\" "
    "time.n.01 TPR now "
    "buy.v.01 Agent -2 Time -1 Theme +1 Manner +2 "
    "car.n.01 Name \"Jaguar\" "
    "gift.n.01 PartOf \"wedding\""
)


def short_id(nid):
    abbrev = {SBN_NODE_TYPE.BOX: "B", SBN_NODE_TYPE.SYNSET: "S", SBN_NODE_TYPE.CONSTANT: "C"}
    return f"{abbrev[nid[0]]}{nid[1]}"


def dump_graph(G, label):
    print(f"\n  --- {label} ---")
    boxes = sorted(
        [(nid, ndat["token"]) for nid, ndat in G.nodes(data=True)
         if ndat["type"] == SBN_NODE_TYPE.BOX],
        key=lambda x: x[0][1],
    )
    syns = sorted(
        [(nid, ndat["token"]) for nid, ndat in G.nodes(data=True)
         if ndat["type"] == SBN_NODE_TYPE.SYNSET],
        key=lambda x: x[0][1],
    )
    consts = sorted(
        [(nid, ndat["token"]) for nid, ndat in G.nodes(data=True)
         if ndat["type"] == SBN_NODE_TYPE.CONSTANT],
        key=lambda x: x[0][1],
    )
    print(f"  Boxes    : {[short_id(b) for b, _ in boxes]}")
    print(f"  Synsets  : {[(short_id(s), tok) for s, tok in syns]}")
    print(f"  Constants: {[(short_id(c), tok) for c, tok in consts]}")

    bb_edges = [(u, v, d["token"]) for u, v, d in G.edges(data=True)
                if d["type"] == SBN_EDGE_TYPE.BOX_BOX_CONNECT]
    print(f"  BOX_BOX_CONNECT:")
    for u, v, tok in bb_edges:
        print(f"    {short_id(u)} -> {short_id(v)}  [{tok}]")

    mem = {}
    for u, v, d in G.edges(data=True):
        if d["type"] == SBN_EDGE_TYPE.BOX_CONNECT:
            mem.setdefault(u, []).append(v)
    print(f"  Membership:")
    for bid, _ in boxes:
        contents = mem.get(bid, [])
        labels = [f"{short_id(c)}:{G.nodes[c]['token']}" for c in contents]
        print(f"    {short_id(bid)} -> {labels or ['<empty>']}")


def smatch_f1(pa, pb):
    for p, r, f in score_amr_pairs([pa], [pb]):
        return float(p), float(r), float(f)
    return None, None, None


def main():
    print("=" * 88)
    print("BOUNDARY TEST: cross-sentential / discourse-level CORRECTION")
    print("=" * 88)

    print("\nDialogue SBN input:")
    print(f"  {DIALOGUE_SBN}")

    try:
        G_dialogue = SBNGraph().from_string(DIALOGUE_SBN, is_single_line=True)
    except Exception as e:
        print(f"\n[STEP 1] PARSE FAILED: {type(e).__name__}: {e}")
        return
    print(f"\n[STEP 1] PARSE OK  (is_dag={G_dialogue.is_dag}, "
          f"is_possibly_ill_formed={G_dialogue.is_possibly_ill_formed})")
    dump_graph(G_dialogue, "raw parse")

    pairs = find_repair_pairs(G_dialogue)
    print(f"\n[STEP 2] find_repair_pairs identified {len(pairs)} pair(s):")
    for parent, c, cj in pairs:
        print(f"    parent={short_id(parent)}  correction={short_id(c)}  conjunction={short_id(cj)}")

    try:
        G_projected = project_repair(G_dialogue)
    except Exception as e:
        print(f"\n[STEP 3] PROJECTION FAILED: {type(e).__name__}: {e}")
        return
    print(f"\n[STEP 3] project_repair completed.")
    dump_graph(G_projected, "after project_repair")

    try:
        pen_proj = G_projected.to_penman_string()
        print(f"\n[STEP 4] Projected Penman:")
        for ln in pen_proj.splitlines():
            print(f"    {ln}")
    except Exception as e:
        print(f"\n[STEP 4] PENMAN FAILED on projection: {type(e).__name__}: {e}")
        return

    try:
        G_target = SBNGraph().from_string(TARGET_FINAL_SBN, is_single_line=True)
        pen_target = G_target.to_penman_string()
        print(f"\n[STEP 5] Target final-truth-value SBN's Penman:")
        for ln in pen_target.splitlines():
            print(f"    {ln}")
    except Exception as e:
        print(f"\n[STEP 5] TARGET PARSE FAILED: {type(e).__name__}: {e}")
        return

    p, r, f = smatch_f1(pen_proj, pen_target)
    print(f"\n[STEP 6] Smatch(projected, target_final):")
    print(f"    P={p:.4f}  R={r:.4f}  F1={f:.4f}")
    if f is not None and abs(f - 1.0) < 1e-6:
        print("    → discourse-level handled correctly")
    else:
        print("    → boundary: current method does NOT collapse discourse-level CORRECTION")


if __name__ == "__main__":
    main()
