"""
Render the SBN DAGs used in research_seminar_pre.qmd as SVG.

Uses the PMB SBN kit unchanged (data/pmb-5.1.0/src/sbn). Highlighting is done
on the pydot object returned by SBNGraph.to_pydot(), so no PMB code is touched.

Run from anywhere:  python documentation/presentation/viz/render_dags.py
"""

import sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR.parents[2] / "data" / "pmb-5.1.0" / "src" / "sbn"))

from sbn_smatch import SBNGraph  # noqa: E402

RED = "#c0392b"
GREY = "#9a9a9a"

GOAT_COW_FLUENT = """
cow.n.01
give_birth.v.01 Agent -1 Time +1 Patient +2
time.n.08 TPR now
calf.n.01 Part +2
quantity.n.01 EQU 2
head.n.01 Quantity -1
"""

GOAT_COW_RSR_PARSE = """
goat.n.01
person.n.01 EQU speaker
mean.v.01 Theme -2 Agent -1 Time +1
time.n.08 EQU now
cow.n.01
give_birth.v.01 Agent -1 Time +1 Patient +2
time.n.08 TPR now
calf.n.01 Part +2
quantity.n.01 EQU 2
head.n.01 Quantity -1
"""

NEGATION_JOHN = """
male.n.02 Name "John"
    NEGATION <1
time.n.08 TPR now
buy.v.01 Agent -2 Time -1 Theme +1
car.n.01
"""

SIRI_RSR = """
entity.n.01 EQU ?
time.n.08 EQU now
weather.n.01
like.a.01 Co-Theme -3 Time -2 Theme -1
    CORRECTION <1
city.n.01 Name "Berlin" LocationOf -1
    CONJUNCTION <2
city.n.01 Name "Dublin" LocationOf -2
"""

SIRI_LATER_REFERENCE = SIRI_RSR + """
    CONTINUATION <1
person.n.01 EQU hearer
show.v.04 Agent -1 Recipient +1 Theme +2
person.n.01 EQU speaker
airport.n.01 Location +2
quantity.n.01 EQU 2
city.n.01 ANA -7 ANA -6 Quantity -1
"""

# pydot node ids are the node tokens; a repeated token gets "-1", "-2", ...
# in order of occurrence. The first time.n.08/now belong to "I mean".
GOAT_COW_SPURIOUS = {"goat.n.01", "person.n.01", "speaker",
                     "mean.v.01", "time.n.08", "now"}


def highlight(p_graph, marked, on=RED, off=GREY):
    """Colour marked nodes `on` and all others `off`; edges follow their target."""
    def colour(name):
        return on if name.strip('"') in marked else off

    for node in p_graph.get_nodes():
        c = colour(node.get_name())
        node.set("color", c)
        node.set("fontcolor", c)
    for edge in p_graph.get_edges():
        c = colour(edge.get_destination())
        edge.set("color", c)
        edge.set("fontcolor", c)
    return p_graph


def main():
    plain = {
        "goat_cow_fluent": GOAT_COW_FLUENT,
        "negation_john": NEGATION_JOHN,
        "siri_rsr": SIRI_RSR,
        "siri_later_reference": SIRI_LATER_REFERENCE,
    }
    for name, sbn in plain.items():
        SBNGraph().from_string(sbn.strip()).to("svg", OUT_DIR / name)
        print(f"wrote {name}.svg")

    p_graph = SBNGraph().from_string(GOAT_COW_RSR_PARSE.strip()).to_pydot()
    highlight(p_graph, GOAT_COW_SPURIOUS).write_svg(
        str(OUT_DIR / "goat_cow_rsr_parse.svg"))
    print("wrote goat_cow_rsr_parse.svg")


if __name__ == "__main__":
    main()
