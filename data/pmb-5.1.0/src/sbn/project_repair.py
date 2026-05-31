"""
Graph-level projection from repair-aware SBN to its truth-value-equivalent
standard SBN, for use in the Smatch evaluation pipeline.

Disambiguation rule
-------------------
A CONJUNCTION sub-box is a repair-CONJUNCTION iff its parent box has another
sub-box reached via a CORRECTION BOX_BOX_CONNECT edge.

Universal-quantifier CONJUNCTIONs (e.g. ¬∃¬ encoding in PMB gold) have no
sibling CORRECTION and survive unchanged.

Projection (per repair pair)
----------------------------
  1. Reparandum removal:
       drop the CORRECTION sub-box and all nodes reachable from it
       (NetworkX descendants closure).
  2. CONJUNCTION collapse:
       re-parent every :member of the CONJUNCTION box to the parent box,
       then drop the CONJUNCTION box node itself. Role-edges between
       surviving synsets are preserved automatically (they live on the
       synset nodes, not on the box nodes).

Indices
-------
Relative indices (`-n`, `+n`, `<n`, `>n`) are resolved to absolute
NetworkX node IDs at parse time, so projection never needs to renumber.
Downstream `to_penman_string` re-assigns var_ids from current node ordering,
which is also unaffected by the removed nodes.
"""
from __future__ import annotations

from copy import deepcopy
from typing import List, Set, Tuple

from sbn_smatch import SBNGraph
from sbn_spec import SBN_EDGE_TYPE, SBN_NODE_TYPE, SBNSpec

RepairTriple = Tuple[tuple, tuple, tuple]  # (parent_box_id, corr_box_id, conj_box_id)

# Edges that participate in the box-containment hierarchy. Role edges
# (ROLE, DRS_OPERATOR, SYN_BOX_CONNECT) are deliberately excluded — a
# reparandum synset may reference matrix-scope synsets via role edges,
# and those targets must survive.
_STRUCTURAL_EDGE_TYPES = (
    SBN_EDGE_TYPE.BOX_BOX_CONNECT,
    SBN_EDGE_TYPE.BOX_CONNECT,
)


def find_repair_pairs(G: SBNGraph) -> List[RepairTriple]:
    """All sibling (CORRECTION, CONJUNCTION) sub-box pairs under a shared parent."""
    children_by_parent: dict = {}
    for u, v, d in G.edges(data=True):
        if d["type"] == SBN_EDGE_TYPE.BOX_BOX_CONNECT:
            children_by_parent.setdefault(u, []).append((v, d.get("token")))

    pairs: List[RepairTriple] = []
    for parent, kids in children_by_parent.items():
        corrs = [b for b, tok in kids if tok == "CORRECTION"]
        conjs = [b for b, tok in kids if tok == "CONJUNCTION"]
        if not (corrs and conjs):
            continue
        for c in corrs:
            for cj in conjs:
                pairs.append((parent, c, cj))
    return pairs


def _reparandum_closure(H: SBNGraph, corr_box) -> Set:
    """Box-structural closure of the reparandum: the CORRECTION box plus every
    node reachable from it via BOX_BOX_CONNECT / BOX_CONNECT edges only.

    Role edges are NOT traversed, so references from reparandum synsets back
    into matrix-scope referents (e.g. an Agent edge into a surviving entity)
    do not drag those referents into the dead set.
    """
    dead = {corr_box}
    stack = [corr_box]
    while stack:
        node = stack.pop()
        for _, v, d in H.out_edges(node, data=True):
            if d["type"] in _STRUCTURAL_EDGE_TYPES and v not in dead:
                dead.add(v)
                stack.append(v)
    return dead


def _sweep_orphan_constants(H: SBNGraph) -> None:
    """Drop constants that lost their last in-edge after reparandum removal.
    Constants are leaf attributes — when their owning synset disappears, they
    no longer carry semantic content and would otherwise inflate Smatch
    triples."""
    orphans = [
        n for n, ndat in H.nodes(data=True)
        if ndat["type"] == SBN_NODE_TYPE.CONSTANT and H.in_degree(n) == 0
    ]
    H.remove_nodes_from(orphans)


def normalize_inverse_roles(G: SBNGraph) -> SBNGraph:
    """Rewrite every ``:XxxOf`` role edge as its forward equivalent ``:Xxx``
    with endpoints swapped.

    PMB's ``smatch.py`` matches relations by exact name (after lowercasing),
    so ``Theme`` and ``Theme-of`` count as distinct relations and a graph
    that uses an inverse role only matches another graph that happens to use
    the same inverse role. Applying this normalization to *both* sides of a
    Smatch comparison yields apples-to-apples triples regardless of the
    surface direction in which the role was written.

    Standard AMR Smatch implementations invert ``-of`` automatically; we
    perform the inversion at the graph layer instead to keep ``smatch.py``
    unmodified.
    """
    H = deepcopy(G)
    rewrites = []
    for u, v, d in H.edges(data=True):
        if d["type"] != SBN_EDGE_TYPE.ROLE:
            continue
        tok = d.get("token", "")
        if tok not in SBNSpec.INVERTIBLE_ROLES or not tok.endswith("Of"):
            continue
        # Skip role-to-constant edges: PMB/Penman convention reserves the
        # source position for synsets/boxes, never for string constants. An
        # edge like `gift :PartOf "birthday"` must stay forward, otherwise
        # the constant lands on the source side and the Penman DFS can no
        # longer reach it.
        if H.nodes[v]["type"] == SBN_NODE_TYPE.CONSTANT:
            continue
        if H.nodes[u]["type"] == SBN_NODE_TYPE.CONSTANT:
            continue
        new_tok = tok[:-2]
        new_data = {k: val for k, val in d.items() if k != "_id"}
        new_data["token"] = new_tok
        rewrites.append((u, v, new_data))

    for u, v, _ in rewrites:
        H.remove_edge(u, v)
    for u, v, data in rewrites:
        # Swap endpoints: original was (u -XxxOf-> v); becomes (v -Xxx-> u).
        H.add_edge(v, u, **data)
    return H


def project_repair(G: SBNGraph, normalize_roles: bool = True) -> SBNGraph:
    """Return a deep-copied SBNGraph with all repair-pair structures projected
    away. Universal-quantifier CONJUNCTIONs are left intact.

    When ``normalize_roles`` is True (default), inverse roles (``ThemeOf`` etc.)
    are rewritten as forward roles so downstream Smatch can compare against a
    standard SBN that uses the natural forward-direction encoding.
    """
    H = deepcopy(G)
    pairs = find_repair_pairs(H)

    for parent, corr_box, conj_box in pairs:
        if corr_box not in H or conj_box not in H:
            continue

        dead = _reparandum_closure(H, corr_box)

        # CONJUNCTION collapse: re-parent its members to the shared parent box.
        new_member_edges = []
        for _, v, d in list(H.out_edges(conj_box, data=True)):
            if d["type"] == SBN_EDGE_TYPE.BOX_CONNECT:
                edge_data = {k: val for k, val in d.items() if k != "_id"}
                new_member_edges.append((parent, v, edge_data))
        for from_, to_, data in new_member_edges:
            H.add_edge(from_, to_, **data)

        dead.add(conj_box)
        H.remove_nodes_from(dead)

    _sweep_orphan_constants(H)

    if normalize_roles:
        H = normalize_inverse_roles(H)

    return H
