#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression suite for repair_metrics.py (Metric A).

Every "prediction" below parses without error -- that is the point. These are
the failure modes that produce a well-formed graph meaning something else, so
they are invisible to a parse-success check and have to be caught by the
metric. Two cases are the reason the metric reports what it reports:

  swap          reparandum and repair have identical tokens, so the
                position-blind score is a perfect 1.0 while the graph is wrong.
                Only the span score sees it.
  wrong_context the reparandum is recovered exactly, but the repair merges into
                the matrix box instead of the negated one -- the silent error
                the notation warns about.

    python3 test_repair_metrics.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbn_env import ensure_on_path  # noqa: E402

ensure_on_path()

from sbn_smatch import SBNGraph                                  # noqa: E402
from repair_metrics import score_item, aggregate, format_summary  # noqa: E402


# ── shared material ──────────────────────────────────────────────────────────

# "I didn't order a banana bread, I mean, a cherry pie" -- repair sits inside
# the NEGATION box, so its merge context is ("NEGATION",).
NEGATION = (
    'person.n.01 EQU speaker time.n.08 TPR now NEGATION <1 '
    'order.v.01 Agent -2 Time -1 '
    'CORRECTION <1 banana_bread.n.01 ThemeOf -1 '
    'CONJUNCTION <2 cherry_pie.n.01 ThemeOf -2'
)

# Same sentence with the negation dropped: identical reparandum, but the repair
# now merges into BOX0. Concept indices are unchanged because separators are
# not concepts, so the reparandum scores perfectly and only the context differs.
NO_NEGATION = (
    'person.n.01 EQU speaker time.n.08 TPR now '
    'order.v.01 Agent -2 Time -1 '
    'CORRECTION <1 banana_bread.n.01 ThemeOf -1 '
    'CONJUNCTION <2 cherry_pie.n.01 ThemeOf -2'
)

# "She will go to, well, went to the church" -- reparandum and repair are both
# {time.n.08, go.v.01, now}.
TENSE = (
    'female.n.02 CORRECTION <1 time.n.08 TSU now go.v.01 Theme -2 Time -1 '
    'CONJUNCTION <2 time.n.08 TPR now '
    'go.v.01 Theme -4 Time -1 Destination +1 church.n.02'
)

# Same tokens, but the CORRECTION quarantines the second time/go pair.
TENSE_SWAPPED = (
    'female.n.02 time.n.08 TSU now go.v.01 Theme -2 Time -1 '
    'CORRECTION <1 time.n.08 TPR now go.v.01 Theme -4 Time -1 Destination +1 '
    'CONJUNCTION <2 church.n.02'
)

FLUENT = 'person.n.01 EQU speaker time.n.08 TPR now order.v.01 Agent -2 Time -1 Theme +1 cherry_pie.n.01'

# CONJUNCTION <3 instead of <2: it stops sharing a source box with the
# CORRECTION, so it is no longer a sibling and no repair is detected.
CONJ_SLIP = (
    'person.n.01 EQU speaker time.n.08 TPR now NEGATION <1 '
    'order.v.01 Agent -2 Time -1 '
    'CORRECTION <1 banana_bread.n.01 ThemeOf -1 '
    'CONJUNCTION <3 cherry_pie.n.01 ThemeOf -2'
)

# Reparandum swallows the tense referent it should not contain.
OVER_QUARANTINE = (
    'person.n.01 EQU speaker order.v.01 Agent -1 Time +1 '
    'CORRECTION <1 time.n.08 TPR now banana_bread.n.01 ThemeOf -2 '
    'CONJUNCTION <2 cherry_pie.n.01 ThemeOf -3'
)

GOLD_OVER = (
    'person.n.01 EQU speaker order.v.01 Agent -1 Time +1 time.n.08 TPR now '
    'CORRECTION <1 banana_bread.n.01 ThemeOf -2 '
    'CONJUNCTION <2 cherry_pie.n.01 ThemeOf -3'
)


CASES = [
    # (name, gold, pred|None, expectations)
    ("perfect", NEGATION, NEGATION,
     dict(gold_n=1, pred_n=1, token_f1=1.0, span_f1=1.0, merge=(1, 1))),

    ("missed_repair", NEGATION, FLUENT,
     dict(gold_n=1, pred_n=0, token_f1=0.0, span_f1=0.0, merge=(0, 1))),

    ("spurious_repair", FLUENT, NEGATION,
     dict(gold_n=0, pred_n=1, token_f1=0.0, span_f1=0.0, merge=(0, 0))),

    ("unparseable_pred", NEGATION, None,
     dict(gold_n=1, pred_n=0, token_f1=0.0, span_f1=0.0, merge=(0, 1))),

    ("conjunction_index_slip", NEGATION, CONJ_SLIP,
     dict(gold_n=1, pred_n=0, token_f1=0.0, span_f1=0.0, merge=(0, 1))),

    # Headline 1: identical tokens, wrong position. Token blind, span catches it.
    ("swap", TENSE, TENSE_SWAPPED,
     dict(gold_n=1, pred_n=1, token_f1=1.0, span_f1=0.0, merge=(1, 1))),

    # Headline 2: reparandum perfect, merge context wrong.
    ("wrong_merge_context", NEGATION, NO_NEGATION,
     dict(gold_n=1, pred_n=1, token_f1=1.0, span_f1=1.0, merge=(0, 1))),

    # Over-quarantine: everything gold wanted is there, plus material that
    # should not have been retracted -> recall 1.0, precision below it.
    ("over_quarantine", GOLD_OVER, OVER_QUARANTINE,
     dict(gold_n=1, pred_n=1, token_recall=1.0, token_precision_lt=1.0,
          merge=(1, 1))),
]


def main() -> int:
    print(f"{'case':<26}{'gold_n':>7}{'pred_n':>7}{'tokF1':>8}{'spanF1':>8}"
          f"{'merge':>8}  note")
    print("-" * 82)

    failures, scores = [], []
    for name, gold_sbn, pred_sbn, exp in CASES:
        gold = SBNGraph().from_string(gold_sbn, is_single_line=True)
        pred = (SBNGraph().from_string(pred_sbn, is_single_line=True)
                if pred_sbn is not None else None)

        s = score_item(gold, pred)
        scores.append(s)

        tok = s.token_f1 if s.token_f1 is not None else 0.0
        span = s.span_f1 if s.span_f1 is not None else 0.0

        problems = []
        for key in ("gold_n", "pred_n"):
            if key in exp and getattr(s, key) != exp[key]:
                problems.append(f"{key}={getattr(s, key)} want {exp[key]}")
        if "token_f1" in exp and abs(tok - exp["token_f1"]) > 1e-9:
            problems.append(f"token_f1={tok:.4f} want {exp['token_f1']}")
        if "span_f1" in exp and abs(span - exp["span_f1"]) > 1e-9:
            problems.append(f"span_f1={span:.4f} want {exp['span_f1']}")
        if "merge" in exp and (s.merge_correct, s.merge_total) != exp["merge"]:
            problems.append(
                f"merge=({s.merge_correct},{s.merge_total}) want {exp['merge']}")
        if "token_recall" in exp:
            r = s.token_tp / s.token_gold if s.token_gold else 0.0
            if abs(r - exp["token_recall"]) > 1e-9:
                problems.append(f"token_recall={r:.4f} want {exp['token_recall']}")
        if "token_precision_lt" in exp:
            p = s.token_tp / s.token_pred if s.token_pred else 0.0
            if not p < exp["token_precision_lt"]:
                problems.append(f"token_precision={p:.4f} want < "
                                f"{exp['token_precision_lt']}")

        note = (f"gold ctx {s.detail['gold_merge_contexts']} "
                f"pred ctx {s.detail['pred_merge_contexts']}")
        flag = " " if not problems else "!"
        print(f"{flag}{name:<25}{s.gold_n:>7}{s.pred_n:>7}{tok:>8.4f}{span:>8.4f}"
              f"{s.merge_correct:>4}/{s.merge_total:<3}  {note}")
        if problems:
            failures.append((name, "; ".join(problems)))

    print("-" * 82)
    print()
    print(format_summary(aggregate(scores)))

    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for name, why in failures:
            print(f"  - {name}: {why}")
        return 1
    print(f"\nAll {len(CASES)} cases pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
