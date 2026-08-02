"""Draw a stratified sample from a scored pool, for calibrating the band.

What this is for.  `nli_filter.SYNONYMY_REJECT_ABOVE` is currently 0.5, a
number with no evidence behind it.  Calibration means finding where on the
score axis human judgement actually flips, by reading a sample of pairs drawn
evenly across the whole range.  The NLI model is not retrained; only the
threshold is set.

Three files are written:

  <out>_items.tsv         what you annotate.  Shuffled, and **without the
                          score**: an annotator who can see the model's answer
                          anchors on it, and the calibration measures nothing.
  <out>_key.tsv           item_id -> score and provenance.  Join this back
                          after annotating.
  <out>_instructions.md   the two questions, and why they are phrased the way
                          they are.

Two questions are asked per item because one annotation pass can serve both
bounds.  Only the first is usable now; the second is stored for when the upper
bound (a masked LM over the substituted word's context) exists.

Run:
    python3 sample_for_calibration.py --pool pool_train.tsv --out calib
    python3 sample_for_calibration.py --pool pool_train.tsv --out calib --per-bin 20
"""
from __future__ import annotations

import argparse
import collections
import csv
import random
from pathlib import Path

INSTRUCTIONS = """# Calibration annotation

{n} pairs, drawn evenly across the model's score range. The score is **not**
shown: it is in `{key}`, to be joined after you finish. Seeing it first would
make the exercise measure agreement with the model rather than judge it.

Each row gives two sentences that differ in exactly one word:

- **A** is the original PMB sentence.
- **B** is A with that word swapped for a candidate reparandum.

## Question 1 — `same_meaning` (this is what we are calibrating now)

> Reading A and B **in this context**, would the speaker be saying the same
> thing either way?

`Y` they are interchangeable here · `N` they say different things · `?` unsure

**Judge only interchangeability, not whether B sounds odd.** Some B sentences
are strange because the word does not belong in that slot at all —
*"Is he more osseous than his brother?"* — and those are **N**: `osseous` and
`tall` do not mean the same thing. Marking them `Y` because they read badly
would put an unrelated defect onto this threshold's account. Oddness is
question 2's business.

Why this question: a reparandum that means the same as the repair gives a
sample where the graph asserts a correction the sentence cannot express. A
reparandum that means something *opposite* — "the big, small room" — is one of
the best samples we can produce, so `N` is the answer we want to see often.

## Question 2 — `fits_sentence` (stored for later)

> Ignoring A entirely: is B a sentence someone could plausibly have started to
> say?

`Y` plausible · `N` the word does not belong there · `?` unsure

No model score exists for this yet. It is collected now so that the upper
bound, when it is built, can be calibrated without a second annotation round.

## Notes column

Free text. Anything that made the call hard is worth a line — those are the
cases that decide whether one threshold is enough.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="scored pool from build_pool.py")
    ap.add_argument("--out", default="calib", help="output file prefix")
    ap.add_argument("--per-bin", type=int, default=15,
                    help="items per 0.1-wide score bin (10 bins)")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(args.pool, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t")]

    scored = [r for r in rows if r.get("synonymy", "") != ""]
    if not scored:
        raise SystemExit(
            f"{args.pool} has no `synonymy` column with values. "
            "Rebuild it with:  python3 build_pool.py --nli ...")

    # One judgement per word pair: the same two synsets in two documents is the
    # same question asked twice, and would weight the sample by how often a
    # word happens to occur in PMB.
    seen: set[tuple[str, str]] = set()
    unique = []
    for r in scored:
        key = (r["repair_synset"], r["reparandum_synset"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    bins: dict[int, list] = collections.defaultdict(list)
    for r in unique:
        b = min(int(float(r["synonymy"]) * 10), 9)
        bins[b].append(r)

    picked = []
    print(f"{'bin':>10s} {'available':>10s} {'taken':>6s}")
    for b in range(10):
        pool_b = bins.get(b, [])
        take = rng.sample(pool_b, min(args.per_bin, len(pool_b)))
        picked.extend(take)
        print(f"{b/10:.1f}-{(b+1)/10:.1f} {len(pool_b):10d} {len(take):6d}")

    rng.shuffle(picked)

    items_path = Path(f"{args.out}_items.tsv")
    key_path = Path(f"{args.out}_key.tsv")
    instr_path = Path(f"{args.out}_instructions.md")

    with open(items_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["item_id", "A_original", "B_substituted",
                    "word_in_A", "word_in_B",
                    "same_meaning", "fits_sentence", "notes"])
        for i, r in enumerate(picked, 1):
            w.writerow([
                f"i{i:03d}", r["nl_clean"], r["nl_substituted"],
                r["repair_synset"].rsplit(".", 2)[0],
                r["reparandum_surface"],
                "", "", "",
            ])

    with open(key_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["item_id", "synonymy", "contradiction", "neutral",
                    "repair_pos", "repair_synset", "reparandum_synset",
                    "doc_id", "concept_pos"])
        for i, r in enumerate(picked, 1):
            w.writerow([
                f"i{i:03d}", r["synonymy"], r["contradiction"], r["neutral"],
                r["repair_pos"], r["repair_synset"], r["reparandum_synset"],
                r["doc_id"], r["concept_pos"],
            ])

    instr_path.write_text(
        INSTRUCTIONS.format(n=len(picked), key=key_path.name), encoding="utf-8")

    by_pos = collections.Counter(r["repair_pos"] for r in picked)
    print(f"\n{len(picked)} items  ({dict(by_pos)})")
    print(f"  annotate : {items_path}")
    print(f"  key      : {key_path}")
    print(f"  how to   : {instr_path}")


if __name__ == "__main__":
    main()
