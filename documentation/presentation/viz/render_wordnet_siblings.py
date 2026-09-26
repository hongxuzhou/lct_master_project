"""
Render where the reparandum `storm` comes from: WordNet siblings of `weather`.

The sibling list is read from WordNet at render time, so the figure shows what
repair_data_gen/wn_candidates._cohyponyms returns (one step up, all children).
Colours follow render_status_graphs.py: the reparandum (withdrawn) is red and
dashed, the repair (accepted) green and solid.

Run from anywhere:  python documentation/presentation/viz/render_wordnet_siblings.py
"""

from pathlib import Path

import pydot
from nltk.corpus import wordnet as wn

OUT_DIR = Path(__file__).resolve().parent

RED = "#c0392b"
GREEN = "#1e8449"
GREY = "#6b6b6b"
FONT = "Times,serif"

REPAIR = "weather.n.01"
REPARANDUM = "storm.n.01"
SHOWN_SIBLINGS = ["cloud.n.02", "sunset.n.02", "aurora.n.02"]


def main():
    repair = wn.synset(REPAIR)
    (parent,) = repair.hypernyms()
    children = {c.name() for c in parent.hyponyms()}
    assert REPARANDUM in children and set(SHOWN_SIBLINGS) <= children
    n_hidden = len(children) - 2 - len(SHOWN_SIBLINGS)

    path = repair.hypernym_paths()[0]
    above = f"{path[0].name()}  ›  …  ›  {path[-3].name()}"

    g = pydot.Dot("G", graph_type="digraph", rankdir="TB", nodesep="0.25",
                  ranksep="0.45", fontname=FONT)
    g.set_node_defaults(fontname=FONT, fontsize="16", shape="ellipse")
    g.set_edge_defaults(arrowsize="0.7", color=GREY)

    g.add_node(pydot.Node("above", label=above, shape="plaintext",
                          fontcolor=GREY, fontsize="14"))
    g.add_node(pydot.Node("parent", label=parent.name(), shape="box",
                          style="rounded", penwidth="1.6"))
    g.add_edge(pydot.Edge("above", "parent", style="dotted"))

    kids = pydot.Subgraph(rank="same")
    kids.add_node(pydot.Node("repair", label=REPAIR, color=GREEN,
                             fontcolor=GREEN, penwidth="1.8"))
    kids.add_node(pydot.Node("reparandum", label=REPARANDUM, color=RED,
                             fontcolor=RED, style="dashed", penwidth="1.4"))
    for i, s in enumerate(SHOWN_SIBLINGS):
        kids.add_node(pydot.Node(f"sib{i}", label=s, color=GREY, fontcolor=GREY))
    kids.add_node(pydot.Node("more", label=f"… {n_hidden} more", shape="plaintext",
                             fontcolor=GREY))
    g.add_subgraph(kids)

    g.add_edge(pydot.Edge("parent", "repair", color=GREEN, penwidth="1.6"))
    g.add_edge(pydot.Edge("parent", "reparandum", color=RED, style="dashed",
                          penwidth="1.4"))
    for i in range(len(SHOWN_SIBLINGS)):
        g.add_edge(pydot.Edge("parent", f"sib{i}"))
    g.add_edge(pydot.Edge("parent", "more", style="invis"))

    out = OUT_DIR / "wordnet_siblings.svg"
    g.write_svg(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
