"""
Render the data-construction flowchart (Data section) as SVG, in the plain
black/white, Times-serif style of the other diagrams in this folder.

Layout: left to right, two lanes. The top lane is the sentence, the bottom lane
is the SBN, matching the two rows of paired_transformation.svg. Steps that act on
both sides span both lanes; only the transformation splits into two cells.
Checks follow the lanes too: linguistic filtering works on the sentence, structural
validation on the graph. Dashed = not yet done (linguistic filtering), per the RP timeline.

Running example: the Siri request. The reparandum candidate (storm.n.01), the
surface form and the SBN splice are the generator's actual output for this
sentence (wn_candidates.candidates, generate_repairs.make_nl with repeated
function words, repair_transform.build_repair; strategy: anchor).

Run from anywhere:  python documentation/presentation/viz/render_data_pipeline.py
"""

from pathlib import Path

import pydot

OUT_DIR = Path(__file__).resolve().parent

FONT = "Times,serif"
MONO = "Courier,monospace"

LANE_H = 125   # height of one lane, in points
STEP_W = 105  # width of a step box


def title(text):
    return f'<FONT POINT-SIZE="21"><B>{text}</B></FONT>'


def desc(text):
    return f'<FONT POINT-SIZE="18">{text}</FONT>'


def ex(text, mono=False):
    face = MONO if mono else FONT
    style = "" if mono else "<I>"
    close = "" if mono else "</I>"
    return f'<FONT FACE="{face}" POINT-SIZE="17">{style}{text}{close}</FONT>'


def cell(body, height, width=STEP_W, style=None, port=None):
    attrs = [f'HEIGHT="{height}"', f'WIDTH="{width}"', 'FIXEDSIZE="FALSE"',
             'CELLPADDING="6"']
    if style:
        attrs.append(f'STYLE="{style}"')
    if port:
        attrs.append(f'PORT="{port}"')
    return f'<TD {" ".join(attrs)}>{body}</TD>'


def table(rows):
    """rows: list of <TD>...</TD> strings, one per table row."""
    inner = "".join(f"<TR>{r}</TR>" for r in rows)
    return f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">{inner}</TABLE>>'


def spanning(t, d, e):
    body = f"{title(t)}<BR/>{desc(d)}<BR/><BR/>{e}"
    return table([cell(body, 2 * LANE_H)])


def node(name, label):
    return pydot.Node(name, label=label, shape="plaintext", fontname=FONT,
                      fontsize="14", margin="0")


def edge(src, dst, **kw):
    return pydot.Edge(src, dst, color="black", arrowsize="0.7", **kw)


def main():
    g = pydot.Dot("G", graph_type="digraph", rankdir="LR", nodesep="0.3",
                  ranksep="0.35", splines="line")
    g.set_node_defaults(fontname=FONT, fontsize="14")

    # Lane labels
    lanes = table([
        f'<TD BORDER="0" HEIGHT="{LANE_H}"><FONT POINT-SIZE="19"><I>sentence</I></FONT></TD>',
        f'<TD BORDER="0" HEIGHT="{LANE_H}"><FONT POINT-SIZE="19"><I>SBN</I></FONT></TD>',
    ])
    g.add_node(node("lanes", lanes))

    g.add_node(node("gold", spanning(
        "① Gold<BR/>PMB pair",
        "sentence, SBN,<BR/>alignment",
        ex("What is<BR/>the weather<BR/>like in Dublin?"))))

    g.add_node(node("site", spanning(
        "② Choose<BR/>repair site",
        "aligned noun,<BR/>verb or<BR/>adjective",
        ex("weather<BR/>weather.n.01"))))

    g.add_node(node("rep", spanning(
        "③ Draw<BR/>reparandum",
        "WordNet,<BR/>same part<BR/>of speech",
        ex("storm<BR/>storm.n.01"))))

    transform = table([
        cell(f"{title('④ Insert before<BR/>the repair')}<BR/>"
             f"{desc('± editing expression')}<BR/>"
             f"{ex('What is the storm,<BR/>I mean, is the weather<BR/>like in Dublin?')}",
             LANE_H, width=190, port="nl"),
        cell(f"{title('④ Add paired<BR/>contexts')}<BR/>"
             f"{desc('both linked to<BR/>shared content')}<BR/>"
             f"{ex('CORRECTION &lt;1<BR/>storm.n.01 …<BR/>CONJUNCTION &lt;2<BR/>weather.n.01 …', mono=True)}",
             LANE_H, width=190, port="sbn"),
    ])
    g.add_node(node("transform", transform))

    check = table([
        cell(f"{title('⑤ Linguistic<BR/>filtering')}<BR/>"
             f"{desc('contrast (NLI),<BR/>contextual fit<BR/>(LM)')}",
             LANE_H, style="dashed", port="ling"),
        cell(f"{title('⑤ Structural<BR/>validation')}<BR/>"
             f"{desc('well-formed,<BR/>status, links,<BR/>rest unchanged')}",
             LANE_H, port="struct"),
    ])
    g.add_node(node("check", check))

    out = table([
        cell(f"{title('RSR<BR/>utterance')}", LANE_H, width=110, port="nl"),
        cell(f"{title('repair-aware<BR/>SBN')}<BR/>{desc('(fluent gold<BR/>SBN kept)')}",
             LANE_H, width=110, port="sbn"),
    ])
    g.add_node(node("out", out))

    g.add_node(pydot.Node("excluded", label="excluded", shape="plaintext",
                          fontname=FONT, fontsize="18"))
    same = pydot.Subgraph(rank="same")
    same.add_node(pydot.Node("check"))
    same.add_node(pydot.Node("excluded"))
    g.add_subgraph(same)

    g.add_edge(pydot.Edge("lanes", "gold", style="invis"))
    g.add_edge(edge("gold", "site"))
    g.add_edge(edge("site", "rep"))
    g.add_edge(edge("rep", "transform:nl:w"))
    g.add_edge(edge("rep", "transform:sbn:w"))
    g.add_edge(edge("transform:nl:e", "check:ling:w"))
    g.add_edge(edge("transform:sbn:e", "check:struct:w"))
    g.add_edge(edge("check:ling:e", "out:nl:w"))
    g.add_edge(edge("check:struct:e", "out:sbn:w"))
    g.add_edge(edge("check", "excluded", style="dashed"))

    svg = g.create_svg(prog="dot")
    out_path = OUT_DIR / "data_pipeline.svg"
    out_path.write_bytes(svg)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
