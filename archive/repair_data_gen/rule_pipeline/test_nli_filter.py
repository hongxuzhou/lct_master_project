"""Regression: does the NLI lower bound separate synonyms from contrast pairs?

The cases are minimal pairs whose verdict is fixed by human judgement -- the
ones recorded in FINDINGS.md §6.5 as bad samples, the notation doc's own
examples, and the antonym cases that motivated using NLI instead of a cosine.

The antonym rows are the point of the test.  A cosine would score
`cheap`/`expensive` as highly similar, because distributional similarity cannot
separate synonymy from antonymy, and would reject one of the best reparanda we
can produce.  If a future change to the filter makes those rows fail, the
filter has regressed to measuring distance instead of relation.

The last two rows are expected to pass through: they are bad samples, but bad
for a reason the *upper* bound catches, not this one.  They are here so the
division of labour stays visible.

Run:  python3 test_nli_filter.py
"""
from nli_filter import SYNONYMY_REJECT_ABOVE, judge_pairs

CASES = [
    # (clean, substituted, label, expected)
    # --- synonyms: reject ---------------------------------------------------
    ("My brother is good at playing tennis.",
     "My brother is effective at playing tennis.", "effective/good", "REJECT"),
    ("This drawer's stuck.", "This drawer's affixed.", "affixed/stuck", "REJECT"),
    ("We saw the bird when we visited Okinawa.",
     "We watched the bird when we visited Okinawa.", "watch/see", "REJECT"),
    ("You told a lie.", "You said a lie.", "say/tell", "REJECT"),
    ("I made dinner.", "I cooked dinner.", "make/cook", "REJECT"),

    # --- antonyms: keep.  A cosine would get every one of these wrong -------
    ("I can only afford the small room.",
     "I can only afford the big room.", "big/small", "KEEP"),
    ("The ticket was expensive.", "The ticket was cheap.", "cheap/expensive", "KEEP"),
    ("Sekkura is a qualified chef.",
     "Sekkura is an unqualified chef.", "unqualified/qualified", "KEEP"),

    # --- same-dimension contrast: keep --------------------------------------
    ("The green curry in that restaurant is too spicy.",
     "The green curry in that restaurant is too sour.", "spicy/sour", "KEEP"),
    ("I ordered a cherry pie.", "I ordered a banana bread.",
     "cherry_pie/banana_bread", "KEEP"),
    ("I saw him.", "I heard him.", "see/hear", "KEEP"),
    ("I ate it.", "I drank it.", "eat/drink", "KEEP"),
    ("The peacock is a beautiful butterfly.",
     "The peacock is a beautiful moth.", "moth/butterfly", "KEEP"),
    ("He threw a stone into the lake.",
     "He threw a stone into the bay.", "bay/lake", "KEEP"),

    # --- not this bound's job: the upper bound rejects these ----------------
    ("Is he taller than his brother?",
     "Is he more osseous than his brother?", "osseous/tall", "PASSES-THROUGH"),
    ("The wind is blowing east.",
     "The wind is blowing region.", "region/east", "PASSES-THROUGH"),
]


def main() -> None:
    verdicts = judge_all()
    print(f"{'pair':26s} {'expected':15s} {'synonymy':>9s} {'contra':>7s}  got")
    print("-" * 74)
    ok = fail = 0
    for (label, expected, v) in verdicts:
        got = "REJECT" if v.too_similar else "KEEP"
        if expected == "PASSES-THROUGH":
            mark = "(upper bound)"
        elif got == expected:
            mark, ok = "ok", ok + 1
        else:
            mark, fail = "FAIL", fail + 1
        print(f"{label:26s} {expected:15s} {v.synonymy:9.3f} "
              f"{v.contradiction:7.3f}  {got:6s} {mark}")
    print(f"\n{ok} passed, {fail} failed "
          f"(threshold synonymy > {SYNONYMY_REJECT_ABOVE})")
    if fail:
        raise SystemExit(1)


def judge_all():
    vs = judge_pairs([(c[0], c[1]) for c in CASES])
    return [(CASES[i][2], CASES[i][3], vs[i]) for i in range(len(CASES))]


if __name__ == "__main__":
    main()
