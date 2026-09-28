"""
Render repair-aware graphs with the status of each candidate made visible.

The stock PMB rendering (render_dags.py) shows CORRECTION and CONJUNCTION only
as small edge labels. For an audience that does not read SBN, the
withdrawn/accepted distinction has to be carried by the picture itself:
withdrawn = red, dashed; accepted = green, solid; links to the content both
candidates share = bold blue (dashed when they leave the withdrawn side).
Line style repeats the colour contrast so the figures also work without colour.

A link counts as shared only if both candidates point at the same node (e.g.
*storm* and *weather* both EQU the anchor entity.n.01 that *like* takes as its
Theme); a role only one candidate carries is drawn as an ordinary edge of its side.

Run from anywhere:  python documentation/presentation/viz/render_status_graphs.py
"""

import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR.parents[2] / "data" / "pmb-5.1.0" / "src" / "sbn"))

from sbn_smatch import SBNGraph  # noqa: E402
from render_dags import SIRI_LATER_REFERENCE, SIRI_RSR  # noqa: E402

RED = "#c0392b"
GREEN = "#1e8449"
BLUE = "#1f4e79"
PURPLE = "#6c3483"

# Data walkthrough item: generator output (anchor construction, device ⑤) for
# the fluent Siri request; see storm_weather_scores.md. The fluent graph is
# fluent_siri_example.svg (render_dags.py).
STORM_WEATHER_RSR = """
entity.n.01 EQU ?
time.n.08 EQU now
entity.n.01
    CORRECTION <1
storm.n.01 EQU -1
    CONJUNCTION <2
weather.n.01 EQU -2
like.a.01 Co-Theme -5 Time -4 Theme -3 Location +1
city.n.01 Name "Dublin"
"""

# pydot node ids are the node tokens; a repeated token gets "-1", "-2", ...
SIRI = dict(withdrawn={"B-1", "city.n.01", '"Berlin"'},
            accepted={"B-2", "city.n.01-1", '"Dublin"'})
STORM_WEATHER = dict(withdrawn={"B-1", "storm.n.01"},
                     accepted={"B-2", "weather.n.01"})
BOX_LABELS = {"B-1": "withdrawn", "B-2": "accepted"}


def style_status(p_graph, withdrawn, accepted, highlight_label=None):
    """Colour a repair-aware pydot graph by status.

    `highlight_label` (e.g. "ANA") draws edges with that label in bold purple,
    for the later-reference figure.
    """
    edges = p_graph.get_edges()
    targets = {}
    for e in edges:
        targets.setdefault(e.get_source(), set()).add(e.get_destination())
    outside = lambda n: n not in withdrawn and n not in accepted  # noqa: E731
    from_w = set().union(*(targets.get(n, set()) for n in withdrawn))
    from_a = set().union(*(targets.get(n, set()) for n in accepted))
    shared = {n for n in from_w & from_a if outside(n)}

    for node in p_graph.get_nodes():
        name = node.get_name()
        if name in BOX_LABELS and (name in withdrawn or name in accepted):
            node.set("label", BOX_LABELS[name])
            node.set("fontsize", "16")
        if name in withdrawn:
            node.set("color", RED)
            node.set("fontcolor", RED)
            node.set("style", "dashed")
        elif name in accepted:
            node.set("color", GREEN)
            node.set("fontcolor", GREEN)
            node.set("penwidth", "1.8")

    for edge in edges:
        src, dst = edge.get_source(), edge.get_destination()
        label = (edge.get("label") or "").strip('"')
        if highlight_label and label == highlight_label:
            edge.set("color", PURPLE)
            edge.set("fontcolor", PURPLE)
            edge.set("penwidth", "2.2")
        elif dst in shared and not outside(src):
            edge.set("color", BLUE)
            edge.set("fontcolor", BLUE)
            edge.set("penwidth", "2.2")
            edge.set("style", "dashed" if src in withdrawn else "solid")
        elif dst in withdrawn:
            edge.set("color", RED)
            edge.set("fontcolor", RED)
            edge.set("style", "dashed")
        elif dst in accepted or (src in accepted and not dst.startswith("B-")):
            edge.set("color", GREEN)
            edge.set("fontcolor", GREEN)
        elif src in withdrawn:
            edge.set("color", RED)
            edge.set("fontcolor", RED)
            edge.set("style", "dashed")
    return p_graph


def render(sbn, out_name, **status):
    p_graph = SBNGraph().from_string(sbn.strip()).to_pydot()
    if status:
        p_graph = style_status(p_graph, **status)
    p_graph.write_svg(str(OUT_DIR / f"{out_name}.svg"))
    print(f"wrote {out_name}.svg")


def main():
    render(SIRI_RSR, "siri_rsr_status", **SIRI)
    render(SIRI_LATER_REFERENCE, "siri_later_reference_status",
           highlight_label="ANA", **SIRI)
    render(STORM_WEATHER_RSR, "storm_weather_rsr_status", **STORM_WEATHER)


if __name__ == "__main__":
    main()
