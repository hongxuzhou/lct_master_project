#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metric A: how well a parser recovers self-repair STRUCTURE.
-- Hongxu Zhou, 2026

Metric B (see repair_strip.py) asks whether the speaker's final semantics
survived. Metric A asks the prior question: did the parser notice the repair at
all, quarantine the right material, and put the merge in the right discourse
context. The two together characterise a parser on this task; they are NOT
independent -- a prediction that mislabels a cross-turn denial as an intra-turn
repair is penalised twice, once here for the misidentification and again in
Metric B because stripping then deletes content the speaker did commit to. Say
so when reporting.

Unlike the stripper, nothing here is gated. Challenge-set items -- the ones
`strip_repair` returns "na" for -- still get a Metric A score; that is the point
of splitting the two.

Three sub-metrics
-----------------
1. DETECTION (item level, binary). Does the prediction contain an intra-turn
   repair at all? Aggregated as P/R/F1 over items. This is the coarsest and
   most robust number, and the one comparable to the disfluency-detection
   literature.

2. REPARANDUM EXTENT. Which concepts got quarantined. Reported twice:
     - token: multiset of tokens, position-blind. Lenient.
     - span : {(linear concept index, token)}. Strict.
   Both, because they fail in different directions and the gap between them is
   diagnostic. Token alone cannot tell a correct parse from one that swapped
   reparandum and repair -- in the tense/aspect case both spans contain exactly
   `time.n.08` and `go.v.01`, so the token score is 1.0 either way. Span alone
   is brittle: one dropped concept earlier in the sentence shifts every index
   and zeroes a semantically correct repair. Read them as a pair -- high token
   with low span means "right material, misplaced".

3. MERGE CONTEXT. The separator path from the root box to the box the repair
   merges into, e.g. `()` for BOX0 and `("NEGATION",)` for the negated box in
   the negation example. Compared as an exact tuple. This is the failure the
   notation flags as silent: putting the repair in the matrix box instead of
   under the negation parses cleanly and means something different. A path is
   used rather than a box index so the comparison survives the two graphs
   having different box counts.

   Note a genuine asymmetry: an off-by-one on the CONJUNCTION index (`<3` where
   `<2` was meant) does NOT show up here as a wrong merge context. It makes the
   CONJUNCTION stop being a sibling of the CORRECTION, so no repair is detected
   at all and the error lands in sub-metric 1. That is the right behaviour --
   per the notation a non-sibling CONJUNCTION is not part of a repair -- but it
   means detection recall, not merge accuracy, is where index slips surface.

Aggregation is micro-averaged (sum the true positives and the two denominators
across items, then divide), which is standard for span-detection tasks and
keeps long repairs from being drowned out by short ones. A macro average over
items is reported alongside.

Usage
-----
    from sbn_smatch import SBNGraph
    from repair_metrics import score_item, aggregate

    scores = [score_item(gold_graph, pred_graph_or_None) for ...]
    print(aggregate(scores))

An unparseable prediction is passed as None and counts as "detected nothing",
i.e. it costs recall rather than being silently dropped from the denominator.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from sbn_env import ensure_on_path

ensure_on_path()
from sbn_spec import SBN_EDGE_TYPE, SBN_NODE_TYPE  # noqa: E402
from repair_strip import box_members, box_subtree, find_repairs  # noqa: E402

__all__ = [
    "RepairSignature",
    "describe_repairs",
    "ItemScore",
    "score_item",
    "aggregate",
]

SBN_ID = Tuple[Any, int]


# ---------------------------------------------------------------------------
# describing a repair
# ---------------------------------------------------------------------------

@dataclass
class RepairSignature:
    """Everything Metric A compares, for one repair."""

    merge_context: Tuple[str, ...]        # separator path root -> merge target
    tokens: Counter                       # token -> count, position-blind
    spans: FrozenSet[Tuple[int, str]]     # (linear concept index, token)
    corr_box_index: int                   # for pairing gold against pred


def _box_path(G, box: SBN_ID) -> Tuple[str, ...]:
    """Separator names from the root box down to `box`; `()` for the root."""
    path: List[str] = []
    current, seen = box, {box}
    while True:
        parents = [
            (u, d["token"])
            for u, _, d in G.in_edges(current, data=True)
            if d["type"] == SBN_EDGE_TYPE.BOX_BOX_CONNECT
        ]
        if not parents:
            break
        parent, token = parents[0]
        if parent in seen:  # defensive: malformed prediction with a box cycle
            break
        seen.add(parent)
        path.append(token)
        current = parent
    return tuple(reversed(path))


def _linear_index(G, node: SBN_ID) -> int:
    """The concept index `+n`/`-n` counts in (clarification C1).

    Synset node ids are `(SYNSET, k)` where k is the creation counter, and the
    parser resolves indices with exactly that counter (sbn_smatch.py:253-256),
    so k IS the index. Constants have a separate counter and are never index
    targets, so they borrow the index of the synset that introduced them.
    """
    if G.nodes[node]["type"] == SBN_NODE_TYPE.SYNSET:
        return node[1]
    owners = [
        u for u, _, _ in G.in_edges(node, data=True)
        if G.nodes[u]["type"] == SBN_NODE_TYPE.SYNSET
    ]
    return owners[0][1] if owners else -1


def _reparandum_elements(G, corr_box: SBN_ID) -> List[SBN_ID]:
    """Nodes quarantined by one CORRECTION, its nested boxes included.

    Constants are included deliberately. Without them the adjunct case is
    unscoreable: reparandum and repair are both `time.n.08`, and the only thing
    telling `monday` from `tuesday` is the constant.
    """
    boxes = box_subtree(G, corr_box)
    members = box_members(G, boxes)
    elements = set(members)
    for m in members:
        for _, v, _ in G.out_edges(m, data=True):
            if G.nodes[v]["type"] == SBN_NODE_TYPE.CONSTANT:
                elements.add(v)
    return sorted(elements)


def describe_repairs(graph) -> List[RepairSignature]:
    """Signature of every intra-turn repair in `graph`, in sequence order."""
    if graph is None:
        return []
    repairs, _ = find_repairs(graph)
    out = []
    for rep in repairs:
        elements = _reparandum_elements(graph, rep.corr_box)
        out.append(RepairSignature(
            merge_context=_box_path(graph, rep.parent),
            tokens=Counter(graph.nodes[n]["token"] for n in elements),
            spans=frozenset(
                (_linear_index(graph, n), graph.nodes[n]["token"])
                for n in elements
            ),
            corr_box_index=rep.corr_box[1],
        ))
    return sorted(out, key=lambda s: s.corr_box_index)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

@dataclass
class ItemScore:
    """Raw counts for one (gold, prediction) pair. Aggregate before dividing."""

    gold_n: int = 0          # repairs in gold
    pred_n: int = 0          # repairs in the prediction
    token_tp: int = 0
    token_pred: int = 0
    token_gold: int = 0
    span_tp: int = 0
    span_pred: int = 0
    span_gold: int = 0
    merge_correct: int = 0
    merge_total: int = 0     # denominator is gold repairs, so a missed repair
                             # counts against merge accuracy too
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_f1(self) -> Optional[float]:
        return _f1(self.token_tp, self.token_pred, self.token_gold)

    @property
    def span_f1(self) -> Optional[float]:
        return _f1(self.span_tp, self.span_pred, self.span_gold)


def _prf(tp: int, n_pred: int, n_gold: int) -> Tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _f1(tp: int, n_pred: int, n_gold: int) -> Optional[float]:
    """None when neither side has anything to score -- a true negative."""
    if not n_pred and not n_gold:
        return None
    return _prf(tp, n_pred, n_gold)[2]


def score_item(gold_graph, pred_graph) -> ItemScore:
    """Score one prediction. Pass `pred_graph=None` for an unparseable output.

    Repairs are paired in sequence order (by the position of their CORRECTION
    box). Unpaired gold repairs cost recall, unpaired predicted ones cost
    precision.
    """
    gold_sigs = describe_repairs(gold_graph)
    pred_sigs = describe_repairs(pred_graph)

    score = ItemScore(gold_n=len(gold_sigs), pred_n=len(pred_sigs))
    score.merge_total = len(gold_sigs)

    for g, p in zip(gold_sigs, pred_sigs):
        score.token_tp += sum((g.tokens & p.tokens).values())
        score.span_tp += len(g.spans & p.spans)
        if g.merge_context == p.merge_context:
            score.merge_correct += 1

    for g in gold_sigs:
        score.token_gold += sum(g.tokens.values())
        score.span_gold += len(g.spans)
    for p in pred_sigs:
        score.token_pred += sum(p.tokens.values())
        score.span_pred += len(p.spans)

    score.detail = {
        "gold_merge_contexts": [g.merge_context for g in gold_sigs],
        "pred_merge_contexts": [p.merge_context for p in pred_sigs],
    }
    return score


def aggregate(scores: List[ItemScore]) -> Dict[str, Any]:
    """Corpus-level Metric A. Micro-averaged, with macro alongside."""
    n_items = len(scores)

    # 1. detection, item level
    det_tp = sum(1 for s in scores if s.gold_n and s.pred_n)
    det_pred = sum(1 for s in scores if s.pred_n)
    det_gold = sum(1 for s in scores if s.gold_n)
    det_p, det_r, det_f = _prf(det_tp, det_pred, det_gold)

    # 2. reparandum extent, micro
    tok_p, tok_r, tok_f = _prf(
        sum(s.token_tp for s in scores),
        sum(s.token_pred for s in scores),
        sum(s.token_gold for s in scores),
    )
    span_p, span_r, span_f = _prf(
        sum(s.span_tp for s in scores),
        sum(s.span_pred for s in scores),
        sum(s.span_gold for s in scores),
    )

    # macro: mean over items that have something to score
    tok_macro = [s.token_f1 for s in scores if s.token_f1 is not None]
    span_macro = [s.span_f1 for s in scores if s.span_f1 is not None]

    # 3. merge context
    merge_total = sum(s.merge_total for s in scores)
    merge_correct = sum(s.merge_correct for s in scores)

    return {
        "n_items": n_items,
        "n_items_with_gold_repair": det_gold,
        "detection_p": det_p,
        "detection_r": det_r,
        "detection_f1": det_f,
        "reparandum_token_p": tok_p,
        "reparandum_token_r": tok_r,
        "reparandum_token_f1": tok_f,
        "reparandum_span_p": span_p,
        "reparandum_span_r": span_r,
        "reparandum_span_f1": span_f,
        "reparandum_token_f1_macro": (
            sum(tok_macro) / len(tok_macro) if tok_macro else 0.0),
        "reparandum_span_f1_macro": (
            sum(span_macro) / len(span_macro) if span_macro else 0.0),
        "merge_context_acc": merge_correct / merge_total if merge_total else 0.0,
        "merge_context_n": merge_total,
    }


def format_summary(agg: Dict[str, Any]) -> str:
    """Human-readable block for logs and the analysis notebook."""
    return "\n".join([
        "=" * 60,
        "METRIC A -- self-repair structure",
        "=" * 60,
        f"items: {agg['n_items']}  (with a gold repair: "
        f"{agg['n_items_with_gold_repair']})",
        "",
        f"detection          P {agg['detection_p']:.4f}  "
        f"R {agg['detection_r']:.4f}  F1 {agg['detection_f1']:.4f}",
        f"reparandum token   P {agg['reparandum_token_p']:.4f}  "
        f"R {agg['reparandum_token_r']:.4f}  F1 {agg['reparandum_token_f1']:.4f}"
        f"  (macro {agg['reparandum_token_f1_macro']:.4f})",
        f"reparandum span    P {agg['reparandum_span_p']:.4f}  "
        f"R {agg['reparandum_span_r']:.4f}  F1 {agg['reparandum_span_f1']:.4f}"
        f"  (macro {agg['reparandum_span_f1_macro']:.4f})",
        f"merge context acc  {agg['merge_context_acc']:.4f} "
        f"over {agg['merge_context_n']} gold repair(s)",
        "=" * 60,
    ])
