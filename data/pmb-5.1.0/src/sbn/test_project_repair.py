"""
Validate `project_repair`:

  * 4 self-repair examples vs hand-written post-repair "standard" SBN.
    Compare via Smatch — expect F1 = 1.0 if projection is correct.

  * 3 PMB-gold universal-quantifier sentences must be untouched
    (projection is a structural no-op because there are no repair pairs).

Run:
    python3 test_project_repair.py
"""
from __future__ import annotations
import sys

try:
    from sbn_smatch import SBNGraph
    from sbn_spec import SBNError
    from smatch import score_amr_pairs
    from project_repair import (
        project_repair,
        find_repair_pairs,
        normalize_inverse_roles,
    )
except ImportError as e:
    sys.exit(f"Import error: {e}\nRun from .../pmb-5.1.0/src/sbn/")


REPAIR_CASES = [
    (
        "repair-1: 'I didn't order a banana bread, I mean, a cherry pie.'",
        # repair-aware
        "person.n.01 EQU speaker time.n.01 TPR now NEGATION <1 "
        "order.v.01 Agent -2 Time -1 Theme +1 CORRECTION <1 "
        "banana_bread.n.01 CONJUNCTION <2 cherry_pie.n.01 ThemeOf -2",
        # post-repair standard
        "person.n.01 EQU speaker time.n.01 TPR now NEGATION <1 "
        "order.v.01 Agent -2 Time -1 Theme +1 cherry_pie.n.01",
    ),
    (
        "repair-2: 'If a farmer owns a donkey, he beats, I mean, feeds it.'",
        "NEGATION <1 farmer.n.01 own.v.01 Pivot -1 Theme +1 donkey.n.01 "
        "NEGATION <1 entity.n.01 EQU -1 CORRECTION <1 "
        "beat.v.01 Agent -4 Patient -1 CONJUNCTION <2 "
        "feed.v.01 Agent -5 Patient -2",
        "NEGATION <1 farmer.n.01 own.v.01 Pivot -1 Theme +1 donkey.n.01 "
        "NEGATION <1 entity.n.01 EQU -1 "
        "feed.v.01 Agent -4 Patient -1",
    ),
    (
        "repair-3: 'The class is on Monday, no, on Tuesday.'",
        "class.n.01 time.n.08 EQU now be.v.01 Theme -2 Time -1 "
        "CORRECTION <1 time.n.08 DayOfWeek \"Monday\" TimeOf -1 "
        "CONJUNCTION <2 time.n.08 DayOfWeek \"Tuesday\" TimeOf -2",
        "class.n.01 time.n.08 EQU now be.v.01 Theme -2 Time -1 "
        "time.n.08 DayOfWeek \"Tuesday\" TimeOf -1",
    ),
    (
        "repair-4: 'John, no, Mary plays tennis.'",
        "CORRECTION <1 person.n.01 Name \"John\" AgentOf +3 "
        "CONJUNCTION <2 person.n.01 Name \"Mary\" time.n.08 EQU now "
        "play.v.01 Time -1 Agent -2 Theme +1 tennis.n.01",
        "person.n.01 Name \"Mary\" time.n.08 EQU now "
        "play.v.01 Time -1 Agent -2 Theme +1 tennis.n.01",
    ),
]


UNIVERSAL_CASES = [
    ("pmb-universal-1 (p20/d2820): Tom is liked by everyone",
     "male.n.02 Name \"Tom\" NEGATION <1 NEGATION <1 time.n.08 EQU now "
     "like.v.03 Stimulus -2 Time -1 Experiencer +1 CONJUNCTION <2 person.n.01"),
    ("pmb-universal-2 (p64/d1787): Tom managed to carry everything himself",
     "male.n.02 Name \"Tom\" NEGATION <1 NEGATION <1 "
     "manage.v.01 Agent -1 Time +1 Topic +2 time.n.08 TPR now "
     "carry.v.01 Agent -3 Theme +1 CONJUNCTION <2 entity.n.01 "
     "CONJUNCTION <2 male.n.02 EQU -5"),
    ("pmb-universal-3 (p00/d0850): She used to play tennis every Sunday",
     "NEGATION <1 NEGATION <1 female.n.02 time.n.08 TPR now "
     "play.v.01 Agent -2 Time -1 Theme +1 Time +2 tennis.n.01 "
     "CONJUNCTION <2 time.n.08 DayOfWeek sunday"),
]


def parse(sbn_str):
    return SBNGraph().from_string(sbn_str, is_single_line=True)


def smatch_f1(penman_a, penman_b):
    for p, r, f in score_amr_pairs([penman_a], [penman_b]):
        return float(p), float(r), float(f)
    return None, None, None


def run_repair_cases():
    print("=" * 88)
    print("REPAIR CASES — projected(repair) vs standard(post-repair)")
    print("=" * 88)
    all_pass = True
    for name, repair_sbn, standard_sbn in REPAIR_CASES:
        print()
        print(f"-- {name}")
        try:
            G_repair = parse(repair_sbn)
            G_standard = parse(standard_sbn)
        except SBNError as e:
            print(f"   PARSE FAILED: {e}")
            all_pass = False
            continue

        pairs = find_repair_pairs(G_repair)
        print(f"   repair pairs found: {len(pairs)}")
        for parent, c, cj in pairs:
            print(f"     parent=B{parent[1]}  correction=B{c[1]}  conjunction=B{cj[1]}")

        G_projected = project_repair(G_repair)
        G_standard_norm = normalize_inverse_roles(G_standard)

        try:
            pen_proj = G_projected.to_penman_string()
            pen_std = G_standard_norm.to_penman_string()
        except SBNError as e:
            print(f"   PENMAN FAILED: {e}")
            all_pass = False
            continue

        p, r, f = smatch_f1(pen_proj, pen_std)
        ok = f is not None and abs(f - 1.0) < 1e-6
        marker = "OK" if ok else "FAIL"
        print(f"   Smatch P={p:.4f} R={r:.4f} F1={f:.4f}   [{marker}]")
        if not ok:
            all_pass = False
            print("   --- projected penman:")
            for ln in pen_proj.splitlines():
                print(f"     {ln}")
            print("   --- standard penman:")
            for ln in pen_std.splitlines():
                print(f"     {ln}")
    return all_pass


def run_universal_cases():
    print()
    print("=" * 88)
    print("UNIVERSAL CASES — projection must be a no-op (Smatch=1.0 vs original)")
    print("=" * 88)
    all_pass = True
    for name, sbn in UNIVERSAL_CASES:
        print()
        print(f"-- {name}")
        try:
            G = parse(sbn)
        except SBNError as e:
            print(f"   PARSE FAILED: {e}")
            all_pass = False
            continue

        pairs = find_repair_pairs(G)
        print(f"   repair pairs found: {len(pairs)}  (expect 0)")
        if pairs:
            all_pass = False

        G_projected = project_repair(G)

        try:
            pen_orig = G.to_penman_string()
            pen_proj = G_projected.to_penman_string()
        except SBNError as e:
            print(f"   PENMAN FAILED: {e}")
            all_pass = False
            continue

        p, r, f = smatch_f1(pen_orig, pen_proj)
        ok = f is not None and abs(f - 1.0) < 1e-6 and len(pairs) == 0
        marker = "OK" if ok else "FAIL"
        print(f"   Smatch P={p:.4f} R={r:.4f} F1={f:.4f}   [{marker}]")
        if not ok:
            all_pass = False
    return all_pass


if __name__ == "__main__":
    ok_a = run_repair_cases()
    ok_b = run_universal_cases()
    print()
    print("=" * 88)
    print(f"REPAIR projections: {'PASS' if ok_a else 'FAIL'}")
    print(f"UNIVERSAL no-op   : {'PASS' if ok_b else 'FAIL'}")
    print("=" * 88)
    sys.exit(0 if (ok_a and ok_b) else 1)
