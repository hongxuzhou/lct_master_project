"""
Render the paired-construction commutative square (Figure "fig:paired-transformation"
in the RP) as SVG, matching the plain black/white, Times-serif style of the other
diagrams in documentation/presentation/viz/.

Run from anywhere:  python documentation/presentation/viz/render_commutative_square.py
"""

from pathlib import Path

import pydot

OUT_DIR = Path(__file__).resolve().parent

FONT = "Times,serif"


def make_node(name, label, pos):
    return pydot.Node(
        name,
        label=label,
        shape="plaintext",
        fontname=FONT,
        fontsize="14",
        pos=f"{pos[0]},{pos[1]}!",
    )


def make_edge(src, dst, label, dashed=False, label_pos=None):
    e = pydot.Edge(
        src,
        dst,
        label=label,
        fontname=FONT,
        fontsize="12",
        color="black",
        arrowsize="0.8",
    )
    if dashed:
        e.set("style", "dashed")
    if label_pos:
        e.set("labelangle", "0")
        e.set("labeldistance", "1")
    return e


def main():
    g = pydot.Dot("G", graph_type="digraph", layout="neato")
    g.set_node_defaults(fontname=FONT, fontsize="14")

    g.add_node(make_node("fluent_sentence", "fluent sentence", (0, 2)))
    g.add_node(make_node("rsr_utterance", "RSR utterance", (5, 2)))
    g.add_node(make_node("fluent_sbn", "fluent gold SBN", (0, 0)))
    g.add_node(make_node("repair_sbn", "repair-aware SBN", (5, 0)))

    g.add_edge(make_edge("fluent_sentence", "rsr_utterance", "insert reparandum"))
    g.add_edge(make_edge("fluent_sentence", "fluent_sbn", "PMB gold"))
    g.add_edge(make_edge("fluent_sbn", "repair_sbn", "RSR transformation"))
    g.add_edge(make_edge("rsr_utterance", "repair_sbn", "?", dashed=True))

    svg = g.create_svg(prog="neato")
    out_path = OUT_DIR / "paired_transformation.svg"
    out_path.write_bytes(svg)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
