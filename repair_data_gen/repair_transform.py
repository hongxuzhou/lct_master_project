"""
The LARD-style repair transform on SBN, and its feasibility test.

Take a gold (nl, sbn) pair, pick one *concept* to repair, synthesise a
WordNet-neighbour reparandum for it, and splice CORRECTION / CONJUNCTION
around the pair.  Because the reparandum is produced by substitution, the
graph-level diff is known by construction -- no second parse is needed.

The difficulty is index arithmetic plus the accessibility clauses C1-C5 of
documentation/knowledge_base/repair_sbn_notation.qmd.  This module makes those
constraints executable so we can measure where the method breaks.

Three strategies, tried in order of increasing cost.  All three are attested in
the notation doc's covered cases:

  INPLACE  devices (1)/(2).  Reparandum sits where the concept was.
           -> reproduces the verb, object, adjunct and tense/aspect cases.
  ANCHOR   device (5): an `entity.n.01` anchor is inserted *before* the
           CORRECTION, every inbound edge is redirected onto it, and both
           copies hang off it with EQU.
           -> reproduces the subject self-repair case verbatim.
  HOIST    device (4): the concept's forward arguments are moved in front of
           the CORRECTION so the reparandum can point back at them.
           -> the only way to repair a prenominal modifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import networkx as nx

from sbn_lin import (
    BOX_PTR_PATTERN, IDX_PATTERN, INVERSE_OF, Concept, SBNDoc, Separator,
    serialise,
)

MAX_INDEX_MAGNITUDE = 9  # SBNSpec.INDEX_PATTERN matches a single digit only
ANCHOR_SYNSET = "entity.n.01"


class Strategy(str, Enum):
    INPLACE = "inplace"
    ANCHOR = "anchor"
    HOIST = "hoist"


class Device(str, Enum):
    DROP_ROLE = "d1_drop_role"
    INVERT_ROLE = "d2_invert_role"
    MOVEMENT = "d4_movement"
    ANCHOR_DUMMY = "d5_anchor_dummy"
    RETARGET = "retarget_to_repair"


class Blocker(str, Enum):
    DANGLING_REPARANDUM = "dangling_reparandum"
    INDEX_OVERFLOW = "index_overflow"
    HOIST_UNAVAILABLE = "hoist_unavailable"
    CYCLE = "cycle"
    SEPARATOR_COLLISION = "separator_collision"


@dataclass
class RepairResult:
    doc_id: str
    pos: int
    strategy: Strategy
    sbn: str
    devices: list[Device] = field(default_factory=list)
    blockers: list[Blocker] = field(default_factory=list)
    dropped_roles: list[str] = field(default_factory=list)
    inverted_roles: list[str] = field(default_factory=list)
    max_abs_index: int = 0
    separator_in_tail: bool = False

    @property
    def ok(self) -> bool:
        return not self.blockers


# --------------------------------------------------------------------------
# slot machinery: the new sequence is a list of slots, each naming what goes
# there.  Everything else (index recomputation) falls out of the slot layout.
# --------------------------------------------------------------------------
OLD = "old"          # ("old", old_pos)
REPARANDUM = "rep"
REPAIR = "fix"
ANCHOR = "anchor"


def _fmt(off: int) -> str:
    return f"+{off}" if off > 0 else str(off)


def _build(doc: SBNDoc, i: int, reparandum_synset: str, strategy: Strategy,
           interregnum: str | None = None) -> RepairResult:
    target = doc.concepts[i]
    edges = list(doc.edges())
    inbound = [(s, r) for (s, r, t) in edges if t == i]
    out_after = sorted({t for (s, r, t) in edges if s == i and t > i})

    devices: list[Device] = []
    blockers: list[Blocker] = []
    dropped: list[str] = []
    inverted: list[str] = []

    # ---- 1. decide the slot layout ---------------------------------------
    hoisted: list[int] = []
    if strategy is Strategy.HOIST:
        # Move the concept's forward arguments (and nothing else) in front of
        # the CORRECTION.  Only safe when those concepts are not themselves
        # depended on by material between i and them.
        hoisted = list(out_after)
        if not hoisted:
            blockers.append(Blocker.HOIST_UNAVAILABLE)
        for h in hoisted:
            for (s, r, t) in edges:
                if s != i and (s == h or t == h) and i < min(s, t) < h:
                    blockers.append(Blocker.HOIST_UNAVAILABLE)
                    break

    slots: list[tuple] = []
    for p in range(len(doc.concepts)):
        if p == i or p in hoisted:
            continue
        slots.append((OLD, p))
    # insert the repair construction just before the first old concept > i
    insert_at = len([s for s in slots if s[1] < i])
    block: list[tuple] = []
    for h in hoisted:
        block.append((OLD, h))
    if strategy is Strategy.ANCHOR:
        block.append((ANCHOR,))
    block.append((REPARANDUM,))
    block.append((REPAIR,))
    slots[insert_at:insert_at] = block

    newpos: dict[object, int] = {}
    for k, s in enumerate(slots):
        newpos[s] = k
    # old concept i resolves to the *repair* copy
    old_to_new = {p: newpos[(OLD, p)] for (kind, p) in
                  [s for s in slots if s[0] == OLD]}
    old_to_new[i] = newpos[(REPAIR,)]
    rep_pos = newpos[(REPARANDUM,)]
    fix_pos = newpos[(REPAIR,)]
    anchor_pos = newpos.get((ANCHOR,))

    # separator positions: CORRECTION opens before the reparandum,
    # CONJUNCTION between reparandum and repair.
    corr_at, conj_at = rep_pos, fix_pos

    # ---- 2. rewrite every concept ----------------------------------------
    new_concepts: list[Concept] = []
    for s in slots:
        if s[0] == ANCHOR:
            new_concepts.append(Concept(newpos[s], ANCHOR_SYNSET, target.box, []))
            continue
        if s[0] == REPARANDUM:
            new_concepts.append(Concept(rep_pos, reparandum_synset, target.box, []))
            continue
        if s[0] == REPAIR:
            new_concepts.append(Concept(fix_pos, target.synset, target.box, []))
            continue
        p = s[1]
        c = doc.concepts[p]
        np = newpos[s]
        roles: list[tuple[str, str]] = []
        for role, val in c.roles:
            if not IDX_PATTERN.match(val):
                roles.append((role, val))
                continue
            old_t = p + int(val)
            if not (0 <= old_t < len(doc.concepts)):
                roles.append((role, val))
                continue
            if old_t == i:
                if strategy is Strategy.ANCHOR:
                    roles.append((role, _fmt(anchor_pos - np)))
                    if Device.ANCHOR_DUMMY not in devices:
                        devices.append(Device.ANCHOR_DUMMY)
                    continue
                if np < corr_at:
                    # Source is a true ancestor (before CORRECTION even opens).
                    # Retargeting its edge to fix_pos always lands inside the
                    # sibling CONJUNCTION box, which C3 calls permeable -- so
                    # this is legal regardless of whether `role` has a
                    # registered "...Of" inverse. Inverting (device②) is still
                    # preferred *when available*, because it additionally
                    # gives the reparandum copy an edge from this source
                    # (fixing what would otherwise be a NON_INVERTIBLE_INBOUND
                    # false alarm: dangling-or-not is decided later, by
                    # whether rep_c ends up with any roles at all, not by
                    # whether this one edge could be duplicated).
                    inv = INVERSE_OF.get(role)
                    if inv is not None:
                        inverted.append(f"{role}->{inv}")
                        if Device.INVERT_ROLE not in devices:
                            devices.append(Device.INVERT_ROLE)
                        continue
                roles.append((role, _fmt(fix_pos - np)))
                if Device.RETARGET not in devices:
                    devices.append(Device.RETARGET)
                continue
            roles.append((role, _fmt(old_to_new[old_t] - np)))
        new_concepts.append(Concept(np, c.synset, c.box, roles))

    by_pos = {c.pos: c for c in new_concepts}
    rep_c, fix_c = by_pos[rep_pos], by_pos[fix_pos]

    # ---- 3. the two copies' own roles ------------------------------------
    for role, val in target.roles:
        if not IDX_PATTERN.match(val):
            rep_c.roles.append((role, val))
            fix_c.roles.append((role, val))
            continue
        old_t = i + int(val)
        if not (0 <= old_t < len(doc.concepts)):
            rep_c.roles.append((role, val))
            fix_c.roles.append((role, val))
            continue
        nt = old_to_new[old_t]
        fix_c.roles.append((role, _fmt(nt - fix_pos)))
        if nt < corr_at:
            # negative index out of the CORRECTION box into an ancestor: legal
            rep_c.roles.append((role, _fmt(nt - rep_pos)))
        else:
            dropped.append(role)                       # device (1)
            if Device.DROP_ROLE not in devices:
                devices.append(Device.DROP_ROLE)

    if strategy is Strategy.ANCHOR:
        rep_c.roles.append(("EQU", _fmt(anchor_pos - rep_pos)))
        fix_c.roles.append(("EQU", _fmt(anchor_pos - fix_pos)))
    else:
        for src, role in inbound:
            inv = INVERSE_OF.get(role)
            if inv is None or old_to_new.get(src, 0) >= corr_at:
                continue
            rep_c.roles.append((inv, _fmt(old_to_new[src] - rep_pos)))
            fix_c.roles.append((inv, _fmt(old_to_new[src] - fix_pos)))

    if not rep_c.roles:
        blockers.append(Blocker.DANGLING_REPARANDUM)

    # ---- 4. separators, and the *second* index space --------------------
    # Splicing in two boxes renumbers every box created after the splice, so
    # downstream `<n` pointers and box-valued role values (`Proposition >1`)
    # have to be recomputed too.  C1 warns that these are a separate index
    # space; getting it wrong parses silently and changes the reading.
    n_before = sum(1 for s in doc.separators if s.after_concept <= i)
    corr_box = n_before + 1
    conj_box = n_before + 2

    def bmap(old_box: int) -> int:
        """Old box id -> new box id."""
        if old_box == 0:
            return 0
        return old_box if old_box <= n_before else old_box + 2

    block_start = insert_at
    new_seps: list[Separator] = []
    for s in doc.separators:
        anchor_old = s.after_concept
        if anchor_old <= i or anchor_old in hoisted:
            # the box opener stays in front of the repair construction
            n_new = (newpos[(OLD, anchor_old)] if (OLD, anchor_old) in newpos
                     else block_start)
        else:
            n_new = min([newpos[sl] for sl in slots
                         if sl[0] == OLD and sl[1] >= anchor_old]
                        or [len(slots)])
        if s.target_box is None:                      # CONTINUATION <0
            ptr = s.ptr
        else:
            ptr = f"<{bmap(s.new_box) - bmap(s.target_box)}"
        new_seps.append(Separator(n_new, s.name, ptr, bmap(s.new_box),
                                  None if s.target_box is None
                                  else bmap(s.target_box)))
    # Tag identity, not name: a pre-existing separator can itself be named
    # CONJUNCTION (any doc with its own coordination/disjunction structure at
    # the repair site), and sorting by name alone can't tell it apart from
    # ours. A pre-existing opener at the same position structurally CONTAINS
    # the repair site, so it must open first regardless of what it's called.
    my_correction = Separator(corr_at, "CORRECTION", "<1", corr_box, n_before)
    my_conjunction = Separator(conj_at, "CONJUNCTION", "<2", conj_box, n_before)
    new_seps.append(my_correction)
    new_seps.append(my_conjunction)
    order = {id(my_correction): 1, id(my_conjunction): 2}
    new_seps.sort(key=lambda s: (s.after_concept, order.get(id(s), 0)))

    # live box per new concept position, by walking the new sequence
    live: list[int] = []
    cur = 0
    si = 0
    for k in range(len(new_concepts)):
        while si < len(new_seps) and new_seps[si].after_concept <= k:
            cur = new_seps[si].new_box
            si += 1
        live.append(cur)

    # Belt and braces: whatever produced `live[]`, the reparandum and repair
    # copies must actually land in *our* two boxes, or the whole point of the
    # splice (quarantine vs. merge-back) is void.
    if live[rep_pos] != corr_box or live[fix_pos] != conj_box:
        blockers.append(Blocker.SEPARATOR_COLLISION)

    old_live = {c.pos: c.box for c in doc.concepts}
    for c in new_concepts:
        for k, (role, val) in enumerate(c.roles):
            m = BOX_PTR_PATTERN.match(val)
            if not m:
                continue
            off = int(m.group(2)) * (1 if m.group(1) == ">" else -1)
            src = [s for s in slots if newpos[s] == c.pos][0]
            if src[0] == OLD:
                old_box = old_live[src[1]]
            else:
                old_box = old_live[i]
            new_off = bmap(old_box + off) - live[c.pos]
            c.roles[k] = (role, f">{new_off}" if new_off > 0
                          else f"<{-new_off}")

    # ---- 5. checks --------------------------------------------------------
    max_abs = 0
    for c in new_concepts:
        for role, val in c.roles:
            if IDX_PATTERN.match(val):
                max_abs = max(max_abs, abs(int(val)))
    if max_abs > MAX_INDEX_MAGNITUDE:
        blockers.append(Blocker.INDEX_OVERFLOW)

    # A legal-looking edge can still close a cycle through some *other* path
    # that isn't visible from any single role's perspective -- e.g. an
    # inverted edge from device② landing back on a concept that is already
    # reachable via an unrelated coordination/Sub chain. This can't be ruled
    # out in advance without a full reachability check, so verify it directly
    # on the graph we actually built.
    concept_graph = nx.DiGraph()
    concept_graph.add_nodes_from(c.pos for c in new_concepts)
    for c in new_concepts:
        for role, val in c.roles:
            if IDX_PATTERN.match(val):
                t = c.pos + int(val)
                if 0 <= t < len(new_concepts):
                    concept_graph.add_edge(c.pos, t)
    if not nx.is_directed_acyclic_graph(concept_graph):
        blockers.append(Blocker.CYCLE)

    if hoisted:
        devices.append(Device.MOVEMENT)

    line = serialise(new_concepts, new_seps)
    if interregnum:
        line = line.replace("CONJUNCTION <2", f"CONJUNCTION <2 % {interregnum}", 1)

    return RepairResult(
        doc_id=doc.doc_id, pos=i, strategy=strategy, sbn=line,
        devices=devices, blockers=sorted(set(blockers), key=lambda b: b.value),
        dropped_roles=dropped, inverted_roles=inverted, max_abs_index=max_abs,
        separator_in_tail=any(s.after_concept > i for s in doc.separators),
    )


STRATEGY_ORDER = (Strategy.INPLACE, Strategy.ANCHOR, Strategy.HOIST)


def strategy_order(pos_tag: str) -> tuple[Strategy, ...]:
    """Fallback order, by what kind of thing is being repaired.

    ANCHOR works by asserting `entity.n.01 EQU <concept>`, which only reads
    correctly when the concept denotes a *referent*.  Equating an entity with
    a property or an event is not what the notation's subject-repair case
    licenses, so for adjectives and verbs movement (device 4) is tried first
    even though it costs surface order.
    """
    if pos_tag in ("a", "r", "v"):
        return (Strategy.INPLACE, Strategy.HOIST, Strategy.ANCHOR)
    return (Strategy.INPLACE, Strategy.ANCHOR, Strategy.HOIST)


def build_repair(doc: SBNDoc, i: int, reparandum_synset: str,
                 interregnum: str | None = None,
                 strategy: Strategy | None = None) -> RepairResult:
    """Cheapest legal splice for a repair at concept position `i`."""
    if strategy is not None:
        return _build(doc, i, reparandum_synset, strategy, interregnum)
    attempts = [_build(doc, i, reparandum_synset, s, interregnum)
                for s in strategy_order(doc.concepts[i].pos_tag)]
    for a in attempts:
        if a.ok:
            return a
    return attempts[0]
