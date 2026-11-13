#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression suite for repair_strip.py, driven by the worked examples in
documentation/knowledge_base/repair_sbn_notation.qmd.

Each case carries the repair-aware SBN and, where an uncontroversial fluent
paraphrase exists, the NATURAL SBN a PMB-style annotator would write for the
repaired sentence. Stripping the former must yield a graph that scores 1.00
against the latter -- that is the actual claim Metric B rests on: the notation
adds repair structure without damaging the semantics underneath.

Cases with `natural=None` are reported for manual inspection only.
Cases expected outside the stripper's domain carry `expect="na"`.

    python3 test_repair_strip.py          # summary
    python3 test_repair_strip.py -v       # + clean Penman for every case
"""

import sys
import os
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sbn_env import ensure_on_path  # noqa: E402

ensure_on_path()

from sbn_smatch import SBNGraph                      # noqa: E402
from repair_strip import strip_repair, StripError    # noqa: E402
from smatchpp import Smatchpp, solvers               # noqa: E402
from smatchpp.formalism.generic.tools import GenericStandardizer  # noqa: E402


@dataclass
class Case:
    name: str
    sentence: str
    sbn: str                      # repair-aware
    natural: Optional[str] = None  # fluent-sentence SBN, or None
    expect: str = "stripped"       # stripped | no_repair | na


CASES = [
    # ── One-sentence cases ───────────────────────────────────────────────
    Case(
        "sv_sentence",
        "The boy sneezed, I mean, coughed",
        'boy.n.01 time.n.08 TPR now CORRECTION <1 sneeze.v.01 Agent -2 Time -1 '
        'CONJUNCTION <2 cough.v.01 Agent -3 Time -2',
        'boy.n.01 time.n.08 TPR now cough.v.01 Agent -2 Time -1',
    ),
    Case(
        "subject_repair",
        "Josh, no, Mary plays tennis well",
        'entity.n.01 CORRECTION <1 male.n.02 Name "Josh" EQU -1 '
        'CONJUNCTION <2 female.n.02 Name "Mary" EQU -2 '
        'play.v.01 Agent -3 Time +1 Theme +2 Manner +3 '
        'time.n.08 EQU now tennis.n.01 well.r.01',
        'female.n.02 Name "Mary" play.v.01 Agent -1 Time +1 Theme +2 Manner +3 '
        'time.n.08 EQU now tennis.n.01 well.r.01',
    ),
    Case(
        "verb_repair",
        "The cat chases, actually, hunts the rat",
        'cat.n.01 time.n.08 EQU now CORRECTION <1 chase.v.01 Agent -2 Time -1 '
        'CONJUNCTION <2 hunt.v.01 Agent -3 Theme +1 Time -2 rat.n.01',
        'cat.n.01 time.n.08 EQU now hunt.v.01 Agent -2 Theme +1 Time -1 rat.n.01',
    ),
    Case(
        "object_repair",
        "I ordered a banana bread, I mean, a cherry pie",
        'person.n.01 EQU speaker time.n.08 TPR now order.v.01 Agent -2 Time -1 '
        'CORRECTION <1 banana_bread.n.01 ThemeOf -1 '
        'CONJUNCTION <2 cherry_pie.n.01 ThemeOf -2',
        'person.n.01 EQU speaker time.n.08 TPR now '
        'order.v.01 Agent -2 Time -1 Theme +1 cherry_pie.n.01',
    ),
    Case(
        "tense_aspect",
        "She will go to, well, went to the church",
        'female.n.02 CORRECTION <1 time.n.08 TSU now go.v.01 Theme -2 Time -1 '
        'CONJUNCTION <2 time.n.08 TPR now '
        'go.v.01 Theme -4 Time -1 Destination +1 church.n.02',
        'female.n.02 time.n.08 TPR now '
        'go.v.01 Theme -2 Time -1 Destination +1 church.n.02',
    ),
    Case(
        "negation",
        "I didn't order a banana bread, I mean, a cherry pie",
        'person.n.01 EQU speaker time.n.08 TPR now NEGATION <1 '
        'order.v.01 Agent -2 Time -1 '
        'CORRECTION <1 banana_bread.n.01 ThemeOf -1 '
        'CONJUNCTION <2 cherry_pie.n.01 ThemeOf -2',
        'person.n.01 EQU speaker time.n.08 TPR now NEGATION <1 '
        'order.v.01 Agent -2 Time -1 Theme +1 cherry_pie.n.01',
    ),
    Case(
        "adjunct",
        "The class is on Monday, no, on Tuesday",
        'class.n.01 time.n.08 EQU now be.v.01 Theme -2 Time -1 '
        'CORRECTION <1 time.n.08 DayOfWeek monday TimeOf -1 '
        'CONJUNCTION <2 time.n.08 DayOfWeek tuesday TimeOf -2',
        'class.n.01 time.n.08 EQU now be.v.01 Theme -2 Time -1 Time +1 '
        'time.n.08 DayOfWeek tuesday',
    ),
    Case(
        "preposition",
        "I ran to, I mean, from the school",
        'person.n.01 EQU speaker time.n.08 TPR now run.v.01 Theme -2 Time -1 '
        'school.n.02 CORRECTION <1 entity.n.01 EQU -2 Destination -1 '
        'CONJUNCTION <2 entity.n.01 EQU -3 Source -2',
        'person.n.01 EQU speaker time.n.08 TPR now '
        'run.v.01 Theme -2 Time -1 Source +1 school.n.02',
    ),
    Case(
        "donkey_repair",
        "If a farmer owns a donkey, he beats, I mean, feeds it",
        'NEGATION <1 farmer.n.01 own.v.01 Pivot -1 Theme +1 donkey.n.01 '
        'NEGATION <1 entity.n.01 EQU -1 '
        'CORRECTION <1 beat.v.01 Agent -4 Patient -1 '
        'CONJUNCTION <2 feed.v.01 Agent -5 Patient -2',
        'NEGATION <1 farmer.n.01 own.v.01 Pivot -1 Theme +1 donkey.n.01 '
        'NEGATION <1 entity.n.01 EQU -1 feed.v.01 Agent -4 Patient -1',
    ),
    Case(
        "retracting",
        "Bill said you will put, I mean, you will drop the ball on the table",
        'male.n.02 Name "Bill" say.v.01 Proposition >1 Agent -1 Time +1 '
        'time.n.08 TPR now CONTINUATION <0 person.n.01 EQU hearer '
        'time.n.08 TSU now CORRECTION <1 put.v.01 Agent -2 Time -1 '
        'CONJUNCTION <2 drop.v.01 Agent -3 Time -2 Theme +1 Destination +2 '
        'ball.n.01 table.n.02',
        'male.n.02 Name "Bill" say.v.01 Proposition >1 Agent -1 Time +1 '
        'time.n.08 TPR now CONTINUATION <0 person.n.01 EQU hearer '
        'time.n.08 TSU now drop.v.01 Agent -2 Time -1 Theme +1 Destination +2 '
        'ball.n.01 table.n.02',
    ),
    Case(
        "forwarding",
        "Josh drove, no, Marsha drove the old car to the church",
        'entity.n.01 CORRECTION <1 male.n.02 Name "Josh" EQU -1 '
        'time.n.08 TPR now drive.v.03 Agent -2 Time -1 '
        'CONJUNCTION <2 female.n.02 Name "Marsha" EQU -4 time.n.08 TPR now '
        'drive.v.03 Agent -2 Time -1 Theme +2 Destination +3 '
        'old.a.02 AttributeOf +1 car.n.01 church.n.02',
        'female.n.02 Name "Marsha" time.n.08 TPR now '
        'drive.v.03 Agent -2 Time -1 Theme +2 Destination +3 '
        'old.a.02 AttributeOf +1 car.n.01 church.n.02',
    ),

    # ── Discourse cases: expected outside the domain ─────────────────────
    Case(
        "dessert_anaphora",
        "My favourite dessert is banana bread, actually, cherry pie. "
        "They are both good desserts",
        'person.n.01 EQU speaker favorite.a.02 Experiencer -1 Stimulus +1 '
        'dessert.n.01 time.n.08 EQU now be.v.02 Theme -2 Time -1 '
        'CORRECTION <1 banana_bread.n.01 Co-ThemeOf -1 '
        'CONJUNCTION <2 cherry_pie.n.01 Co-ThemeOf -2 '
        'CONTINUATION <1 entity.n.01 ANA -2 ANA -1 time.n.08 EQU now '
        'be.v.02 Theme -2 Time -1 Co-Theme +2 good.a.01 AttributeOf +1 '
        'dessert.n.01',
        None,
        expect="na",
    ),
    Case(
        "joe_ferrari",
        "A: Joe bought a Ferrari, I mean, a Jaguar, as a birthday gift. "
        "B: No, he bought it as a wedding gift.",
        'male.n.02 Name "Joe" time.n.08 TPR now '
        'buy.v.01 Agent -2 Time -1 Attribute +3 '
        'CORRECTION <1 car.n.01 Name "Ferrari" ThemeOf -1 '
        'CONJUNCTION <2 car.n.01 Name "Jaguar" ThemeOf -2 '
        'gift.n.01 Of +1 birthday.n.01 '
        'CORRECTION <1 male.n.02 EQU -7 '
        'buy.v.01 Agent -1 Theme -4 Attribute +1 gift.n.01 Of +1 wedding.n.01',
        None,
        expect="na",
    ),

    # ── Controls: nothing here may be mistaken for a self-repair ─────────
    Case(
        "donkey_baseline",
        "Every farmer who owns a donkey beats it",
        'NEGATION -1 farmer.n.01 own.v.01 Pivot -1 Theme +1 donkey.n.01 '
        'NEGATION -1 beat.v.01 Agent -3 Patient +1 entity.n.01 EQU -2',
        None,
        expect="no_repair",
    ),
    Case(
        # Lascarides & Asher denial proper: real discourse content, never
        # stripped. No sibling CONJUNCTION, so the structural test excludes it.
        "cross_turn_only",
        "A: Joe bought a Ferrari. B: No, he bought a Jaguar.",
        'male.n.02 Name "Joe" time.n.08 TPR now '
        'buy.v.01 Agent -2 Time -1 Theme +1 car.n.01 Name "Ferrari" '
        'CORRECTION <1 male.n.02 EQU -4 time.n.08 TPR now '
        'buy.v.01 Agent -2 Time -1 Theme +1 car.n.01 Name "Jaguar"',
        None,
        expect="no_repair",
    ),
    Case(
        # The configuration that motivated the worry: a cross-turn CORRECTION
        # DOES get followed by a genuine coordinating CONJUNCTION. It is not a
        # false positive, because that CONJUNCTION hangs off the CORRECTION's
        # own box (source BOX1) rather than sharing the CORRECTION's source
        # box (BOX0) -- so the two are parent/child, not siblings.
        "cross_turn_plus_conjunction",
        "A: Joe bought a Ferrari. B: No, he bought a Jaguar, and he loves it.",
        'male.n.02 Name "Joe" time.n.08 TPR now '
        'buy.v.01 Agent -2 Time -1 Theme +1 car.n.01 Name "Ferrari" '
        'CORRECTION <1 male.n.02 EQU -4 time.n.08 TPR now '
        'buy.v.01 Agent -2 Time -1 Theme +1 car.n.01 Name "Jaguar" '
        'CONJUNCTION <1 love.v.01 Agent -5 Theme -2 Time +1 time.n.08 EQU now',
        None,
        expect="no_repair",
    ),
]


def main() -> int:
    verbose = "-v" in sys.argv
    scorer = Smatchpp(alignmentsolver=solvers.ILP(),
                      graph_standardizer=GenericStandardizer())

    print(f"{'case':<20}{'status':<11}{'exp':<11}{'vs natural':<12}note")
    print("-" * 88)

    failures = []
    for case in CASES:
        try:
            graph = SBNGraph().from_string(case.sbn, is_single_line=True)
        except Exception as e:
            print(f"{case.name:<20}{'PARSE-ERR':<11}{case.expect:<11}{'-':<12}{e}")
            failures.append((case.name, f"parse error: {e}"))
            continue

        try:
            res = strip_repair(graph)
        except StripError as e:
            print(f"{case.name:<20}{'STRIP-ERR':<11}{case.expect:<11}{'-':<12}{e}")
            failures.append((case.name, f"strip error: {e}"))
            continue

        status_ok = res.status == case.expect
        note = res.reason if res.status == "na" else ""
        if res.detail.get("dummies_dissolved"):
            note = f"dissolved {res.detail['dummies_dissolved']}"

        f1_cell = "-"
        if res.status in ("stripped", "no_repair") and case.natural:
            try:
                clean = res.graph.to_penman_string()
                nat = SBNGraph().from_string(
                    case.natural, is_single_line=True).to_penman_string()
                f1 = scorer.score_pair(nat, clean)["main"]["F1"]
                f1_cell = f"{f1:.2f}"
                if f1 < 99.999:
                    failures.append((case.name, f"F1 vs natural = {f1:.2f}"))
            except Exception as e:
                f1_cell = "ERR"
                failures.append((case.name, f"scoring failed: {e}"))

        if not status_ok:
            failures.append(
                (case.name, f"status {res.status!r}, expected {case.expect!r}"))

        flag = " " if status_ok else "!"
        print(f"{flag}{case.name:<19}{res.status:<11}{case.expect:<11}"
              f"{f1_cell:<12}{note[:40]}")

        if verbose and res.graph is not None:
            print("    --- repair-aware ---")
            print("   ", case.sbn)
            print("    --- clean penman ---")
            for line in res.graph.to_penman_string().split("\n"):
                print("   ", line)
            if case.natural:
                print("    --- natural penman ---")
                for line in SBNGraph().from_string(
                        case.natural, is_single_line=True
                ).to_penman_string().split("\n"):
                    print("   ", line)
            print()

    print("-" * 88)
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for name, why in failures:
            print(f"  - {name}: {why}")
        return 1
    print(f"\nAll {len(CASES)} cases pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
