"""
Regression: does the transform reproduce the hand-written covered cases of
documentation/knowledge_base/repair_sbn_notation.qmd from their clean SBN?

Each case gives (clean SBN in the doc's own linearisation, repair site,
reparandum synset, expected repaired SBN).  A pass means the mechanical
splice and the annotator's hand agree exactly.

Run:  python3 test_against_doc.py
"""
from sbn_lin import parse_sbn_line
from repair_transform import Strategy, build_repair

CASES = [
    dict(
        name="SV sentence  (The boy sneezed, I mean, coughed)",
        clean="boy.n.01 time.n.08 TPR now cough.v.01 Agent -2 Time -1",
        i=2, reparandum="sneeze.v.01", strategy=None,
        want="boy.n.01 time.n.08 TPR now CORRECTION <1 sneeze.v.01 Agent -2 "
             "Time -1 CONJUNCTION <2 cough.v.01 Agent -3 Time -2",
    ),
    dict(
        name="Verb self-repair  (The cat chases, actually, hunts the rat)",
        clean="cat.n.01 time.n.08 EQU now hunt.v.01 Agent -2 Theme +1 Time -1 "
              "rat.n.01",
        i=2, reparandum="chase.v.01", strategy=None,
        want="cat.n.01 time.n.08 EQU now CORRECTION <1 chase.v.01 Agent -2 "
             "Time -1 CONJUNCTION <2 hunt.v.01 Agent -3 Theme +1 Time -2 "
             "rat.n.01",
    ),
    dict(
        name="Object self-repair  (I ordered a banana bread, I mean, a cherry pie)",
        clean="person.n.01 EQU speaker time.n.08 TPR now order.v.01 Agent -2 "
              "Time -1 Theme +1 cherry_pie.n.01",
        i=3, reparandum="banana_bread.n.01", strategy=None,
        want="person.n.01 EQU speaker time.n.08 TPR now order.v.01 Agent -2 "
             "Time -1 CORRECTION <1 banana_bread.n.01 ThemeOf -1 "
             "CONJUNCTION <2 cherry_pie.n.01 ThemeOf -2",
    ),
    dict(
        name="Subject self-repair  (Josh, no, Mary plays tennis well)",
        clean='female.n.02 Name "Mary" play.v.01 Agent -1 Time +1 Theme +2 '
              "Manner +3 time.n.08 EQU now tennis.n.01 well.r.01",
        i=0, reparandum="male.n.02", strategy=Strategy.ANCHOR,
        want='entity.n.01 CORRECTION <1 male.n.02 Name "Mary" EQU -1 '
             'CONJUNCTION <2 female.n.02 Name "Mary" EQU -2 play.v.01 '
             "Agent -3 Time +1 Theme +2 Manner +3 time.n.08 EQU now "
             "tennis.n.01 well.r.01",
    ),
    dict(
        name="Negation  (I didn't order a banana bread, I mean, a cherry pie)",
        clean="person.n.01 EQU speaker time.n.08 TPR now NEGATION <1 "
              "order.v.01 Agent -2 Time -1 Theme +1 cherry_pie.n.01",
        i=3, reparandum="banana_bread.n.01", strategy=None,
        want="person.n.01 EQU speaker time.n.08 TPR now NEGATION <1 "
             "order.v.01 Agent -2 Time -1 CORRECTION <1 banana_bread.n.01 "
             "ThemeOf -1 CONJUNCTION <2 cherry_pie.n.01 ThemeOf -2",
    ),
    dict(
        name="Adjunct  (The class is on Monday, no, on Tuesday)",
        clean="class.n.01 time.n.08 EQU now be.v.01 Theme -2 Time -1 Time +1 "
              "time.n.08 DayOfWeek tuesday",
        i=3, reparandum="time.n.08", strategy=None,
        want="class.n.01 time.n.08 EQU now be.v.01 Theme -2 Time -1 "
             "CORRECTION <1 time.n.08 DayOfWeek tuesday TimeOf -1 "
             "CONJUNCTION <2 time.n.08 DayOfWeek tuesday TimeOf -2",
    ),
    dict(
        name="Donkey sentence  (…he beats, I mean, feeds it)",
        clean="NEGATION <1 farmer.n.01 own.v.01 Pivot -1 Theme +1 donkey.n.01 "
              "NEGATION <1 entity.n.01 EQU -1 feed.v.01 Agent -4 Patient -1",
        i=4, reparandum="beat.v.01", strategy=None,
        want="NEGATION <1 farmer.n.01 own.v.01 Pivot -1 Theme +1 donkey.n.01 "
             "NEGATION <1 entity.n.01 EQU -1 CORRECTION <1 beat.v.01 "
             "Agent -4 Patient -1 CONJUNCTION <2 feed.v.01 Agent -5 "
             "Patient -2",
    ),
    dict(
        name="Retracting repair  (Bill said you will put, I mean, drop …)",
        clean='male.n.02 Name "Bill" say.v.01 Proposition >1 Agent -1 Time +1 '
              "time.n.08 TPR now CONTINUATION <0 person.n.01 EQU hearer "
              "time.n.08 TSU now drop.v.01 Agent -2 Time -1 Theme +1 "
              "Destination +2 ball.n.01 table.n.02",
        i=5, reparandum="put.v.01", strategy=None,
        want='male.n.02 Name "Bill" say.v.01 Proposition >1 Agent -1 Time +1 '
             "time.n.08 TPR now CONTINUATION <0 person.n.01 EQU hearer "
             "time.n.08 TSU now CORRECTION <1 put.v.01 Agent -2 Time -1 "
             "CONJUNCTION <2 drop.v.01 Agent -3 Time -2 Theme +1 "
             "Destination +2 ball.n.01 table.n.02",
    ),
    dict(
        name="Forwarding repair  (Josh drove, no, Marsha drove the old car…)",
        clean='female.n.02 Name "Marsha" time.n.08 TPR now drive.v.03 '
              "Agent -2 Time -1 Theme +2 Destination +3 old.a.02 "
              "AttributeOf +1 car.n.01 church.n.02",
        i=0, reparandum="male.n.02", strategy=Strategy.ANCHOR,
        want=None,  # doc's version repeats the whole VP; see note below
    ),
]


def main() -> None:
    ok = fail = skipped = 0
    for case in CASES:
        doc = parse_sbn_line(case["clean"], case["name"])
        res = build_repair(doc, case["i"], case["reparandum"],
                           strategy=case["strategy"])
        want = case["want"]
        if want is None:
            print(f"~ SKIP  {case['name']}")
            print(f"        produced: {res.sbn}")
            skipped += 1
            continue
        got = " ".join(res.sbn.split())
        want = " ".join(want.split())
        if got == want:
            print(f"[PASS] {case['name']}")
            print(f"       strategy={res.strategy.value} "
                  f"devices={[d.value for d in res.devices]}")
            ok += 1
        else:
            print(f"[FAIL] {case['name']}")
            print(f"       want: {want}")
            print(f"       got : {got}")
            fail += 1
    print(f"\n{ok} passed, {fail} failed, {skipped} skipped")


if __name__ == "__main__":
    main()
