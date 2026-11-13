#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Repair stripping: repair-aware SBN graph -> "clean" graph (speaker's final commitment).
-- Hongxu Zhou, 2026

Metric B of the evaluation pipeline needs the *semantics the speaker actually
committed to*, with the self-repair scaffolding removed, so it can be scored
against natural (fluent-sentence) SBN. This module performs that removal at the
GRAPH level, never at the string level: `SBNGraph.from_string` has already
resolved every `+n`/`-n` index into a concrete node reference
(`sbn_smatch.py:253-256`), so deleting nodes needs no index renumbering, and
`to_penman_string` regenerates all variable names from scratch
(`sbn_smatch.py:614-617`). Do not round-trip back to an SBN string: that WOULD
require recomputing indices and is deliberately out of scope.

The stripper is a PARTIAL FUNCTION with an explicit, machine-checkable domain.
Anything outside the domain returns status "na" and belongs in the challenge
set, evaluated on Metric A + qualitative analysis only. This is a feature: the
claim is not "all repairs are cleanable", it is "here is the cleanable
fragment, and here is its boundary".

What gets removed
-----------------
1. The CORRECTION subtree (reparandum), transitively -- nested boxes included.
2. The CONJUNCTION box, flattened into its merge target (clarification C5:
   CONJUNCTION does not open a DRS, it merges into its target box).
3. Repair-scaffolding dummies, of which there are two mirror-image kinds:
     - INNER (device 3, edge-label repair, e.g. the preposition case):
       the dummy sits inside a repair box, is the SOURCE of an EQU to a synset
       plus exactly one other role. Dissolved by hoisting that role onto the
       EQU host:  `d EQU X, d R Y`  =>  `X R Y`.
     - ANCHOR (concept repair, e.g. the subject case): the dummy sits in the
       parent box, is the TARGET of EQU edges from the repair candidates, and
       carries no roles of its own. Dissolved by merging it into the surviving
       candidate: inbound edges are redirected, then the dummy is deleted.
   Both are identified by STRUCTURAL SIGNATURE, never by token, because
   `entity.n.01` is the general-purpose placeholder of the notation (1315
   occurrences in gold) and is overwhelmingly used for genuine content
   (pronouns, anaphora). See `_find_dummies` for the discriminators.

Cross-turn CORRECTION (Lascarides & Asher denial proper) is NOT stripped: it is
real discourse content. It is told apart structurally, exactly as the notation
specifies -- an intra-turn repair has a sibling CONJUNCTION, i.e. a CONJUNCTION
box-box edge leaving the SAME source box. No annotation on the token is needed.

Usage
-----
    from sbn_smatch import SBNGraph
    from repair_strip import strip_repair, to_clean_penman_string

    res = strip_repair(SBNGraph().from_string(sbn, is_single_line=True))
    if res.status in ("stripped", "no_repair"):
        penman = res.graph.to_penman_string()

IMPORTANT for scoring: pass `graph_standardizer=GenericStandardizer()` to
Smatchpp. Without it smatch++ does NOT de-invert `:X-of` edges, so every
inverted role (device 2) is scored as a wholly different edge. Verified:
forward-vs-inverted on an otherwise identical graph scores 93.33 by default and
100.0 with the standardizer.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from sbn_env import ensure_on_path

ensure_on_path()
from sbn_spec import SBN_EDGE_TYPE, SBN_NODE_TYPE  # noqa: E402

__all__ = [
    "StripResult",
    "strip_repair",
    # shared with repair_metrics.py: Metric A must run on challenge-set items
    # too, so repair discovery cannot live behind the stripper's gates.
    "Repair",
    "find_repairs",
    "box_subtree",
    "box_members",
    "to_clean_penman_string",
    "StripError",
]

SBN_ID = Tuple[Any, int]

# Roles/operators that carry meaning, as opposed to box membership bookkeeping.
_CONTENT_EDGE_TYPES = {
    SBN_EDGE_TYPE.ROLE,
    SBN_EDGE_TYPE.DRS_OPERATOR,
    SBN_EDGE_TYPE.SYN_BOX_CONNECT,
}


class StripError(Exception):
    """Raised when stripping produced a structurally invalid graph."""


@dataclass
class StripResult:
    """Outcome of stripping one graph.

    status:
        "no_repair" -- no intra-turn repair present; graph returned unchanged.
                       Still valid input for Metric B.
        "stripped"  -- at least one repair removed; `graph` is the clean graph.
        "na"        -- outside the stripper's domain; `graph` is None and the
                       item belongs in the challenge set. `reason` says which
                       gate rejected it.
    """

    status: str
    graph: Optional[Any] = None
    reason: str = ""
    n_repairs: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------

def _box_box_edges(G, token: Optional[str] = None) -> List[Tuple[SBN_ID, SBN_ID]]:
    """All box->box separator edges, optionally filtered by separator name."""
    return [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d["type"] == SBN_EDGE_TYPE.BOX_BOX_CONNECT
        and (token is None or d["token"] == token)
    ]


def box_subtree(G, box: SBN_ID) -> Set[SBN_ID]:
    """Every box reachable from `box` through box-box edges, `box` included.

    Deletion must be transitive: a reparandum may itself contain a NEGATION,
    and dropping only the direct members would leave an orphaned subtree that
    `to_penman_string` silently drops (see `_assert_healthy`).
    """
    seen, stack = {box}, [box]
    while stack:
        b = stack.pop()
        for _, v, d in G.out_edges(b, data=True):
            if d["type"] == SBN_EDGE_TYPE.BOX_BOX_CONNECT and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def box_members(G, boxes: Set[SBN_ID]) -> Set[SBN_ID]:
    """Synset/constant nodes that belong to any of `boxes`."""
    return {
        v
        for b in boxes
        for _, v, d in G.out_edges(b, data=True)
        if d["type"] == SBN_EDGE_TYPE.BOX_CONNECT
    }


def _home_box(G, node: SBN_ID) -> Optional[SBN_ID]:
    """The box `node` is a member of, if any."""
    for u, _, d in G.in_edges(node, data=True):
        if d["type"] == SBN_EDGE_TYPE.BOX_CONNECT:
            return u
    return None


def _effective_box(G, node: SBN_ID) -> Optional[SBN_ID]:
    """Where `node` effectively lives.

    Constants get no box-connect edge of their own during parsing -- they hang
    off the synset that introduced them -- so `_home_box` returns None for
    them. A constant lives wherever its owner lives.
    """
    if G.nodes[node]["type"] == SBN_NODE_TYPE.CONSTANT:
        owners = [u for u, _, _ in G.in_edges(node, data=True)]
        return _home_box(G, owners[0]) if owners else None
    return _home_box(G, node)


def _content_out(G, node: SBN_ID) -> List[Tuple[SBN_ID, Dict]]:
    return [
        (v, d) for _, v, d in G.out_edges(node, data=True)
        if d["type"] in _CONTENT_EDGE_TYPES
    ]


def _content_in(G, node: SBN_ID) -> List[Tuple[SBN_ID, Dict]]:
    return [
        (u, d) for u, _, d in G.in_edges(node, data=True)
        if d["type"] in _CONTENT_EDGE_TYPES
    ]


def _is_synset(G, node: SBN_ID) -> bool:
    return G.nodes[node]["type"] == SBN_NODE_TYPE.SYNSET


_KIND_PREFIX = {
    SBN_NODE_TYPE.BOX: "box",
    SBN_NODE_TYPE.SYNSET: "s",
    SBN_NODE_TYPE.CONSTANT: "c",
}


def _fmt(node: SBN_ID) -> str:
    """Readable node id for messages -- `box2`, `s7`.

    Raw ids are `(SBN_NODE_TYPE.BOX, 2)` tuples whose repr leaks the enum into
    every `strip_reason` string, which then lands in the gold table and the
    challenge-set breakdown. Keep those readable.
    """
    return f"{_KIND_PREFIX.get(node[0], '?')}{node[1]}"


# ---------------------------------------------------------------------------
# repair configuration discovery
# ---------------------------------------------------------------------------

@dataclass
class Repair:
    parent: SBN_ID      # box both separators point out of
    corr_box: SBN_ID    # box opened by CORRECTION (holds the reparandum)
    conj_box: SBN_ID    # box opened by the sibling CONJUNCTION


def find_repairs(G) -> Tuple[List[Repair], List[str]]:
    """Locate intra-turn repairs; return (repairs, rejection_reasons).

    The structural test is the one the notation specifies: a CORRECTION is
    intra-turn iff a CONJUNCTION box-box edge leaves the SAME source box. A
    cross-turn CORRECTION followed by a genuine coordinating CONJUNCTION is not
    a false positive -- that CONJUNCTION hangs off the CORRECTION's own box, so
    its source differs.
    """
    repairs: List[Repair] = []
    problems: List[str] = []
    conjunctions = _box_box_edges(G, "CONJUNCTION")

    for parent, corr_box in _box_box_edges(G, "CORRECTION"):
        siblings = [j for (p, j) in conjunctions if p == parent]
        if not siblings:
            continue  # cross-turn denial: genuine content, leave it alone
        if len(siblings) > 1:
            problems.append(
                f"CORRECTION at box {corr_box} has {len(siblings)} sibling "
                "CONJUNCTIONs; repair scope is ambiguous"
            )
            continue
        repairs.append(Repair(parent=parent, corr_box=corr_box,
                               conj_box=siblings[0]))
    return repairs, problems


# ---------------------------------------------------------------------------
# gates -- the machine-checkable domain boundary
# ---------------------------------------------------------------------------

def _gate_conjunction_boundary(G, rep: Repair) -> Optional[str]:
    """Gate 1. Nothing outside the merge may be attached to the CONJUNCTION box.

    Flattening dissolves the box, so every edge touching it has to be
    retargeted onto the merge target -- and that retargeting is exactly what
    the notation treats as meaning-changing.

    Note the direction convention: box-box edges run `target_box -> new_box`,
    so a separator that *points at* the CONJUNCTION box shows up as an OUT
    edge of it. The cross-turn CORRECTION written `<1` in the Joe/Ferrari
    example is such an edge: it deliberately localises B's denial to A's
    CONJUNCTION box rather than to BOX0, and the notation states outright that
    `<1` and `<3` are different graphs scored differently. Flattening would
    collapse them. The same applies to a following `CONTINUATION`.

    A box-box out-edge opened by material *within* the conjunction span would
    in principle be safe to re-parent, but it is not reliably distinguishable
    from the deliberate case, so both are refused. In practice this is the
    line between a repair that ends its sentence and one with discourse built
    on top of it -- i.e. exactly the discourse-level challenge set.

    Foreign IN edges (e.g. a `Proposition >n` aimed at the box) are refused
    for the same reason.
    """
    hanging = [
        f"{d['token']}->{_fmt(v)}"
        for _, v, d in G.out_edges(rep.conj_box, data=True)
        if d["type"] == SBN_EDGE_TYPE.BOX_BOX_CONNECT
    ]
    if hanging:
        return (f"{_fmt(rep.conj_box)} (CONJUNCTION) is the target of "
                f"separator(s) [{', '.join(hanging)}]; flattening would "
                "retarget them onto the merge target and collapse a "
                "distinction the notation treats as meaningful")

    foreign = [
        f"{d['token']} from {_fmt(u)}"
        for u, _, d in G.in_edges(rep.conj_box, data=True)
        if not (u == rep.parent and d["token"] == "CONJUNCTION")
    ]
    if foreign:
        return (f"{_fmt(rep.conj_box)} (CONJUNCTION) has foreign inbound "
                f"edge(s) [{', '.join(foreign)}]; flattening would silently "
                "retarget them")
    return None


def _gate_dangling_into_reparandum(G, doomed: Set[SBN_ID]) -> Optional[str]:
    """Gate 2. No surviving node may point into the reparandum.

    Clarification C3 forbids inbound positive indices, and device 2 puts
    boundary-crossing edges on the INNER node pointing out, so deleting the box
    normally removes them automatically. The systematic exception is C4: an
    anaphor (`entity.n.01 ANA -2`) reaching back into the reparandum. Those
    graphs have no uncontroversial clean reading, so they are deferred pending
    a decision on the dangling-edge policy.
    """
    dangling = [
        f"{_fmt(u)} -{d['token']}-> {_fmt(v)}"
        for v in doomed
        for u, _, d in G.in_edges(v, data=True)
        if u not in doomed and d["type"] in _CONTENT_EDGE_TYPES
    ]
    if dangling:
        return (f"surviving node(s) point into the reparandum "
                f"[{', '.join(dangling)}] (clarification C4); dangling-edge "
                "policy undecided")
    return None


# ---------------------------------------------------------------------------
# dummy detection -- by structure, never by token
# ---------------------------------------------------------------------------

@dataclass
class _Dummy:
    node: SBN_ID
    kind: str                      # "inner" | "anchor"
    host: Optional[SBN_ID] = None  # inner: the EQU target
    role: Optional[Tuple[SBN_ID, Dict]] = None  # inner: (target, edge_data)


def _classify_dummy(G, node: SBN_ID, repair_boxes: Set[SBN_ID]) -> Optional[_Dummy]:
    """Return a _Dummy if `node` matches a scaffolding signature, else None.

    INNER (device 3): sits in a repair box; source of exactly one EQU whose
      target is a SYNSET (not a constant -- `time.n.08 EQU now` and
      `person.n.01 EQU speaker` are ordinary content and must not match); plus
      exactly one other role edge; no inbound content edges; and crucially
      BOTH endpoints live outside the repair boxes. That last clause is what
      the dummy is for: it exists only to carry a repaired edge between two
      concepts the speaker did commit to. Without it, `male.n.02 Name "Josh"
      EQU -1` from the subject-repair case matches too -- it also has one EQU
      to a synset plus one other role -- but its `Name` target is a constant
      inside the reparandum, so it is content, not scaffolding.

    ANCHOR: target of at least one EQU originating INSIDE a repair box; carries
      no content edges of its own. The bare `entity.n.01` that gives the
      reparandum something to attach to.

    Deliberately NOT matched, verified against the notation's own examples:
      - donkey "it" (`entity.n.01 EQU -1`): has an EQU out-edge but no second
        role, and its EQU target is reached from the parent box -> fails INNER;
        has an out-edge -> fails ANCHOR.
      - dessert "they" (`entity.n.01 ANA -2 ANA -1`): no EQU at all.
    """
    if not _is_synset(G, node):
        return None

    out_edges = _content_out(G, node)
    in_edges = _content_in(G, node)

    equ_out = [(v, d) for v, d in out_edges if d["token"] == "EQU"]
    other_out = [(v, d) for v, d in out_edges if d["token"] != "EQU"]

    # --- INNER ---------------------------------------------------------
    if (_home_box(G, node) in repair_boxes
            and len(equ_out) == 1
            and len(other_out) == 1
            and not in_edges
            and _is_synset(G, equ_out[0][0])
            and _effective_box(G, equ_out[0][0]) not in repair_boxes
            and _effective_box(G, other_out[0][0]) not in repair_boxes):
        return _Dummy(node=node, kind="inner", host=equ_out[0][0],
                      role=other_out[0])

    # --- ANCHOR --------------------------------------------------------
    equ_in_from_repair = [
        (u, d) for u, d in in_edges
        if d["token"] == "EQU" and _home_box(G, u) in repair_boxes
    ]
    if equ_in_from_repair and not out_edges:
        return _Dummy(node=node, kind="anchor")

    return None


def _find_dummies(G, repairs: List[Repair]) -> List[_Dummy]:
    repair_boxes: Set[SBN_ID] = set()
    for rep in repairs:
        repair_boxes |= box_subtree(G, rep.corr_box)
        repair_boxes |= box_subtree(G, rep.conj_box)

    dummies = []
    for node in list(G.nodes):
        if (d := _classify_dummy(G, node, repair_boxes)) is not None:
            dummies.append(d)
    return dummies


# ---------------------------------------------------------------------------
# transformations
# ---------------------------------------------------------------------------

def _delete_reparandum(G, rep: Repair) -> Set[SBN_ID]:
    """Remove the CORRECTION box, its whole subtree, and their members."""
    boxes = box_subtree(G, rep.corr_box)
    doomed = boxes | box_members(G, boxes)
    G.remove_nodes_from(doomed)
    return doomed


def _flatten_conjunction(G, rep: Repair) -> None:
    """Merge the CONJUNCTION box into its target (clarification C5).

    Members move to the parent box; boxes opened inside the CONJUNCTION box are
    re-parented onto the merge target. Gate 1 has already guaranteed that no
    foreign edge points at this box, so only the CONJUNCTION edge itself needs
    removing.
    """
    for _, v, d in list(G.out_edges(rep.conj_box, data=True)):
        if d["type"] in (SBN_EDGE_TYPE.BOX_CONNECT,
                         SBN_EDGE_TYPE.BOX_BOX_CONNECT):
            G.add_edge(rep.parent, v, **d)
            G.remove_edge(rep.conj_box, v)
    G.remove_node(rep.conj_box)


def _dissolve_dummy(G, dummy: _Dummy) -> None:
    """Remove one scaffolding dummy, preserving the semantics it stood in for."""
    if dummy.kind == "inner":
        # `d EQU X, d R Y`  =>  `X R Y`
        target, edata = dummy.role
        # networkx silently materialises a bare, attribute-less node when
        # add_edge names one that no longer exists, which surfaces much later
        # as an opaque KeyError. Fail here instead.
        for endpoint in (dummy.host, target):
            if endpoint not in G:
                raise StripError(
                    f"inner dummy {dummy.node} references deleted node "
                    f"{endpoint}; signature matched scaffolding that was not"
                )
        G.add_edge(dummy.host, target, **edata)
        G.remove_node(dummy.node)
        return

    # anchor: merge into the surviving EQU partner
    partners = [u for u, d in _content_in(G, dummy.node) if d["token"] == "EQU"]
    if len(partners) != 1:
        raise StripError(
            f"anchor dummy {dummy.node} has {len(partners)} surviving EQU "
            "partners after stripping; expected exactly 1"
        )
    survivor = partners[0]
    for u, _, d in list(G.in_edges(dummy.node, data=True)):
        if u == survivor and d["token"] == "EQU":
            continue  # the EQU that defined the identity is now vacuous
        if d["type"] in _CONTENT_EDGE_TYPES:
            G.add_edge(u, survivor, **d)
    G.remove_node(dummy.node)


def _drop_orphan_constants(G) -> None:
    """Constants are created fresh per occurrence, so a deleted owner orphans one."""
    orphans = [
        n for n in list(G.nodes)
        if G.nodes[n]["type"] == SBN_NODE_TYPE.CONSTANT and G.in_degree(n) == 0
    ]
    G.remove_nodes_from(orphans)


# ---------------------------------------------------------------------------
# post-conditions
# ---------------------------------------------------------------------------

def _assert_healthy(G) -> None:
    """Guard against the silent-truncation failure mode.

    `to_penman_string` starts its traversal at `[n for n, d in G.in_degree()
    if d == 0][0]` (sbn_smatch.py:717) -- the FIRST in-degree-0 node -- and
    serialises only what is reachable from it. A stripping bug that splits the
    graph would therefore drop content with no error at all, depressing both
    gold and prediction by unequal amounts. Fail loudly instead.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise StripError("clean graph is cyclic; Penman export impossible")

    roots = [n for n, d in G.in_degree() if d == 0]
    if len(roots) != 1:
        raise StripError(
            f"clean graph has {len(roots)} in-degree-0 nodes "
            f"[{', '.join(_fmt(n) for n in roots)}]; to_penman_string would "
            "serialise only the first"
        )
    if G.nodes[roots[0]]["type"] != SBN_NODE_TYPE.BOX:
        raise StripError(f"clean graph root {_fmt(roots[0])} is not a box")

    unreachable = set(G.nodes) - (nx.descendants(G, roots[0]) | {roots[0]})
    if unreachable:
        raise StripError(
            f"{len(unreachable)} node(s) unreachable from root: "
            f"[{', '.join(_fmt(n) for n in sorted(unreachable))}]"
        )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def strip_repair(graph) -> StripResult:
    """Strip self-repair scaffolding from an SBNGraph. Never mutates the input."""
    G = deepcopy(graph)

    repairs, problems = find_repairs(G)
    if problems:
        return StripResult(status="na", reason="; ".join(problems))
    if not repairs:
        return StripResult(status="no_repair", graph=G, n_repairs=0)

    for rep in repairs:
        if (why := _gate_conjunction_boundary(G, rep)) is not None:
            return StripResult(status="na", reason=why)

    # Dummies must be spotted while both repair boxes are still present.
    dummies = _find_dummies(G, repairs)

    for rep in repairs:
        boxes = box_subtree(G, rep.corr_box)
        doomed = boxes | box_members(G, boxes)
        if (why := _gate_dangling_into_reparandum(G, doomed)) is not None:
            return StripResult(status="na", reason=why)

    for rep in repairs:
        _delete_reparandum(G, rep)
        _flatten_conjunction(G, rep)

    # Only dummies that survived deletion still need dissolving.
    dissolved = [d for d in dummies if d.node in G]
    for dummy in dissolved:
        _dissolve_dummy(G, dummy)

    _drop_orphan_constants(G)
    _assert_healthy(G)

    return StripResult(
        status="stripped",
        graph=G,
        n_repairs=len(repairs),
        detail={
            "dummies_dissolved": [(d.kind, str(d.node)) for d in dissolved],
            "dummies_detected": [(d.kind, str(d.node)) for d in dummies],
        },
    )


def to_clean_penman_string(sbn_string: str, is_single_line: bool = True,
                           **penman_kwargs) -> Tuple[Optional[str], StripResult]:
    """Convenience wrapper: SBN string -> (clean Penman | None, StripResult)."""
    from sbn_smatch import SBNGraph

    res = strip_repair(
        SBNGraph().from_string(sbn_string, is_single_line=is_single_line)
    )
    if res.status == "na":
        return None, res
    return res.graph.to_penman_string(**penman_kwargs), res
