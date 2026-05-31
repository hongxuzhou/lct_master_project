"""
Probe what the current pmb-5.1.0 SBN parser (sbn_smatch.py) produces for:
  - 4 self-repair examples (CORRECTION + CONJUNCTION) of the proposed design
  - 3 PMB-gold universal-quantifier examples (NEGATION + NEGATION + CONJUNCTION)

Does not modify any existing file. Run from this directory:
    python3 verify_conjunction.py
"""
import sys
import textwrap

try:
    from sbn_smatch import SBNGraph
    from sbn_spec import SBN_NODE_TYPE, SBN_EDGE_TYPE, SBNError
except ImportError as e:
    sys.exit(f"Import error: {e}\nRun from .../pmb-5.1.0/src/sbn/")


EXAMPLES = [
    # --- proposed self-repair design ---
    ("repair-1: banana bread / cherry pie",
     "person.n.01 EQU speaker time.n.01 TPR now NEGATION <1 "
     "order.v.01 Agent -2 Time -1 Theme +1 CORRECTION <1 "
     "banana_bread.n.01 CONJUNCTION <2 cherry_pie.n.01 ThemeOf -2"),

    ("repair-2: donkey beats / feeds  [CORRECTED: NEGATION <1, not <2]",
     "NEGATION <1 farmer.n.01 own.v.01 Pivot -1 Theme +1 donkey.n.01 "
     "NEGATION <1 entity.n.01 EQU -1 CORRECTION <1 "
     "beat.v.01 Agent -4 Patient -1 CONJUNCTION <2 "
     "feed.v.01 Agent -5 Patient -2"),

    ("repair-3: Monday / Tuesday",
     "class.n.01 time.n.08 EQU now be.v.01 Theme -2 Time -1 "
     "CORRECTION <1 time.n.08 DayOfWeek \"Monday\" TimeOf -1 "
     "CONJUNCTION <2 time.n.08 DayOfWeek \"Tuesday\" TimeOf -2"),

    ("repair-4: John / Mary",
     "CORRECTION <1 person.n.01 Name \"John\" AgentOf +3 "
     "CONJUNCTION <2 person.n.01 Name \"Mary\" time.n.08 EQU now "
     "play.v.01 Time -1 Agent -2 Theme +1 tennis.n.01"),

    # --- PMB-5.1.0 gold universal-quantifier sentences ---
    ("pmb-universal-1 (p20/d2820): Tom is liked by everyone",
     "male.n.02 Name \"Tom\" NEGATION <1 NEGATION <1 time.n.08 EQU now "
     "like.v.03 Stimulus -2 Time -1 Experiencer +1 CONJUNCTION <2 person.n.01"),

    ("pmb-universal-2 (p64/d1787): Tom managed to carry everything himself",
     "male.n.02 Name \"Tom\" NEGATION <1 NEGATION <1 "
     "manage.v.01 Agent -1 Time +1 Topic +2 time.n.08 TPR now "
     "carry.v.01 Agent -3 Theme +1 CONJUNCTION <2 entity.n.01 "
     "CONJUNCTION <2 male.n.02 EQU -5"),

    ("pmb-universal-3 (p00/d0850): She used to play tennis every Sunday",
     "NEGATION <1 NEGATION <1 female.n.02 time.n.08 TPR now "
     "play.v.01 Agent -2 Time -1 Theme +1 Time +2 tennis.n.01 "
     "CONJUNCTION <2 time.n.08 DayOfWeek sunday"),
]


def list_boxes(G):
    """Return list of (box_id, token) sorted by box index."""
    return sorted(
        [(nid, ndat["token"]) for nid, ndat in G.nodes(data=True)
         if ndat["type"] == SBN_NODE_TYPE.BOX],
        key=lambda x: x[0][1],
    )


def list_synsets_and_constants(G):
    return sorted(
        [(nid, ndat["token"]) for nid, ndat in G.nodes(data=True)
         if ndat["type"] != SBN_NODE_TYPE.BOX],
        key=lambda x: (x[0][0].value, x[0][1]),
    )


def list_edges_by_type(G, edge_type):
    out = []
    for u, v, d in G.edges(data=True):
        if d["type"] == edge_type:
            out.append((u, v, d.get("token")))
    return out


def short_id(nid):
    type_, idx = nid
    abbrev = {
        SBN_NODE_TYPE.BOX: "B",
        SBN_NODE_TYPE.SYNSET: "S",
        SBN_NODE_TYPE.CONSTANT: "C",
    }
    return f"{abbrev.get(type_, '?')}{idx}"


def membership_map(G):
    """box -> list of (short_id, token) for members."""
    m = {}
    for u, v, d in G.edges(data=True):
        if d["type"] == SBN_EDGE_TYPE.BOX_CONNECT:
            m.setdefault(u, []).append((short_id(v), G.nodes[v]["token"]))
    return m


def verify_one(name, sbn_str):
    print("=" * 88)
    print(name)
    print("-" * 88)
    print("SBN input:")
    print(textwrap.fill(sbn_str, width=82, initial_indent="  ", subsequent_indent="  "))
    print()

    try:
        G = SBNGraph().from_string(sbn_str, is_single_line=True)
    except SBNError as e:
        print(f"  PARSE STATUS: SBNError -> {e}")
        return
    except Exception as e:
        print(f"  PARSE STATUS: {type(e).__name__} -> {e}")
        return

    print(f"  PARSE STATUS: OK  (is_dag={G.is_dag}, "
          f"is_possibly_ill_formed={G.is_possibly_ill_formed})")
    print()

    print("  Boxes (B{idx} = box index):")
    for bid, btok in list_boxes(G):
        print(f"    {short_id(bid):4s}  token={btok}")

    print()
    print("  Synsets/Constants:")
    for nid, tok in list_synsets_and_constants(G):
        print(f"    {short_id(nid):4s}  {tok}")

    print()
    print("  BOX_BOX_CONNECT edges (source-box -> target-box [relation-token]):")
    for u, v, tok in list_edges_by_type(G, SBN_EDGE_TYPE.BOX_BOX_CONNECT):
        print(f"    {short_id(u)} -> {short_id(v)}  [{tok}]")

    print()
    print("  BOX_CONNECT (membership) edges (box :member of synset/constant):")
    mm = membership_map(G)
    for bid, btok in list_boxes(G):
        contents = mm.get(bid, [])
        labels = [f"{sid}:{tok}" for sid, tok in contents] or ["<empty>"]
        print(f"    {short_id(bid):4s}: {labels}")

    syn_box = list_edges_by_type(G, SBN_EDGE_TYPE.SYN_BOX_CONNECT)
    if syn_box:
        print()
        print("  SYN_BOX_CONNECT edges (synset -> box [role-token]):")
        for u, v, tok in syn_box:
            print(f"    {short_id(u)} -> {short_id(v)}  [{tok}]")

    print()
    print("  Penman output:")
    try:
        pen = G.to_penman_string()
        for ln in pen.splitlines():
            print(f"    {ln}")
    except SBNError as e:
        print(f"    Penman conversion FAILED (SBNError): {e}")
    except Exception as e:
        print(f"    Penman conversion FAILED [{type(e).__name__}]: {e}")
    print()


if __name__ == "__main__":
    for name, sbn in EXAMPLES:
        verify_one(name, sbn)
