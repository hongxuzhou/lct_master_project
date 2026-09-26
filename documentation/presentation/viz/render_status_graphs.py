"""
Render the Siri RSR graph with the status of each candidate made visible.

The stock PMB rendering (render_dags.py -> siri_rsr.svg) shows CORRECTION and
CONJUNCTION only as small edge labels. For an audience that does not read SBN,
the withdrawn/accepted distinction has to be carried by the picture itself:
withdrawn = red, dashed; accepted = green, solid; the shared role = bold.
Line style repeats the colour contrast so the figure also works without colour.

Run from anywhere:  python documentation/presentation/viz/render_status_graphs.py
"""

import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR.parents[2] / "data" / "pmb-5.1.0" / "src" / "sbn"))

from sbn_smatch import SBNGraph  # noqa: E402
from render_dags import SIRI_RSR  # noqa: E402

RED = "#c0392b"
GREEN = "#1e8449"
BLUE = "#1f4e79"

# pydot node ids are the node tokens; a repeated token gets "-1", "-2", ...
WITHDRAWN = {"B-1", "city.n.01", '"Berlin"'}
ACCEPTED = {"B-2", "city.n.01-1", '"Dublin"'}
BOX_LABELS = {"B-1": "withdrawn", "B-2": "accepted"}
SHARED_ROLE = "LocationOf"


def style_status(p_graph):
    for node in p_graph.get_nodes():
        name = node.get_name()
        if name in BOX_LABELS:
            node.set("label", BOX_LABELS[name])
            node.set("fontsize", "16")
        if name in WITHDRAWN:
            node.set("color", RED)
            node.set("fontcolor", RED)
            node.set("style", "dashed")
        elif name in ACCEPTED:
            node.set("color", GREEN)
            node.set("fontcolor", GREEN)
            node.set("penwidth", "1.8")

    for edge in p_graph.get_edges():
        src, dst = edge.get_source(), edge.get_destination()
        label = edge.get("label") or ""
        if label.strip('"') == SHARED_ROLE:
            edge.set("color", BLUE)
            edge.set("fontcolor", BLUE)
            edge.set("penwidth", "2.2")
            edge.set("style", "dashed" if src in WITHDRAWN else "solid")
        elif dst in WITHDRAWN:
            edge.set("color", RED)
            edge.set("fontcolor", RED)
            edge.set("style", "dashed")
        elif dst in ACCEPTED:
            edge.set("color", GREEN)
            edge.set("fontcolor", GREEN)
    return p_graph


def main():
    p_graph = SBNGraph().from_string(SIRI_RSR.strip()).to_pydot()
    style_status(p_graph).write_svg(str(OUT_DIR / "siri_rsr_status.svg"))
    print("wrote siri_rsr_status.svg")


if __name__ == "__main__":
    main()
