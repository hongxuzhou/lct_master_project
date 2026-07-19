#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diff-based ground truth for the repair-sensitivity ablation.
-- Hongxu Zhou, 2026

Per design decision 5 in the ablation plan: the pilot dataset was built by
INSERTING a reparandum(+interregnum) into the gold sentence, so the reparandum
is always exactly the material inserted relative to gold, and the repair is
always the (untouched) gold sentence itself. Ground truth needs no new
annotation — it is recovered by diffing `nl` against each `repair_*_nl`
column.

Alignment is done on LOWERCASED whitespace tokens (SequenceMatcher), because
insertion can shift capitalization on what used to be the gold sentence's
first word (e.g. p60/d1674: gold "My house..." -> repair_head "Your house, my
house..." — "my" lowercases once it's no longer sentence-initial). Extracted
spans are then sliced from the ORIGINAL-case tokens at the aligned indices, so
the recovered reparandum keeps its real casing regardless of that shift.

For `*_interrug` conditions, the interregnum is always the fixed literal
" I mean," (verified: present in all 2508 interrug rows in
Shrikes/self_repair_parsing_pilot_data). Diffing the interrug variety directly
against `gold_nl` is fragile when gold happens to contain a token that
coincidentally collides with a token in "I mean" (observed: gold sentences
starting with "I ..." can get that "I" absorbed into the alignment's matched
block instead of the inserted one, e.g. p41/d2557, p96/d1687 — see
ground_truth_smoke.py). To avoid that, the interregnum is stripped out via a
literal string replace FIRST (reconstructing the non-interrug-equivalent
text), and reparandum is then extracted the same way as the non-interrug
conditions, against that reconstructed text.

`GroundTruth.suspect` flags rows where the diff produced more than one
non-contiguous inserted block — i.e. more than a single clean reparandum
insertion — which turned out to catch a genuine data defect (p40/d1979,
repair_tail[_interrug]: two separate insertions land in what should be a
single-repair condition, one of them an empty-string artifact). Downstream
scoring should exclude `suspect` rows from Tier-2 (localization) comparisons.
"""

import difflib
import re
from dataclasses import dataclass
from typing import Optional

_INTERRUG_LITERAL = " I mean,"


@dataclass
class GroundTruth:
    reparandum: str
    interregnum: Optional[str]   # None for non-interrug conditions
    repair: str                  # always == gold_nl (design decision 5)
    suspect: bool = False        # True if the diff looked like >1 insertion


def _tokenize(text: str):
    return text.split()


def _alignment_key(token: str) -> str:
    """Lowercased token with leading/trailing punctuation stripped, used only
    to decide token equality for alignment. The gold (`nl`) column and the
    `repair_*_nl` variant columns disagree on trailing sentence punctuation
    surprisingly often (e.g. gold "...a goal net" vs variant "...a goal
    net."), which otherwise makes the final gold token look "replaced"
    instead of "equal" and corrupts the diff with a spurious tail block."""
    return re.sub(r'^[\'"]+|[\'".,!?]+$', "", token.lower())


def _strip_edges(text: str) -> str:
    """Drop leading/trailing whitespace and a single leading/trailing comma
    left over from the insertion boundary (e.g. "RJ Reynolds," -> "RJ
    Reynolds")."""
    text = text.strip()
    text = re.sub(r",\s*$", "", text)
    text = re.sub(r"^,\s*", "", text)
    return text.strip()


def diff_inserted_span(gold_nl: str, variety_nl: str):
    """Return (inserted_text, n_blocks): the text inserted in `variety_nl`
    relative to `gold_nl`, aligning on lowercased tokens but returning
    original-case text, plus how many non-contiguous insert/replace opcodes
    contributed to it (>1 signals a likely data anomaly)."""
    gold_tokens = _tokenize(gold_nl)
    var_tokens = _tokenize(variety_nl)
    gold_key = [_alignment_key(t) for t in gold_tokens]
    var_key = [_alignment_key(t) for t in var_tokens]

    sm = difflib.SequenceMatcher(None, gold_key, var_key, autojunk=False)
    inserted = []
    n_blocks = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            n_blocks += 1
            inserted.extend(var_tokens[j1:j2])
    return " ".join(inserted), n_blocks


def extract_ground_truth(gold_nl: str, variety_nl: str, condition: str) -> GroundTruth:
    """condition: one of gold, repair_{head,mid,tail}[_interrug]."""
    if condition == "gold":
        return GroundTruth(reparandum="", interregnum=None, repair=gold_nl)

    is_interrug = condition.endswith("_interrug")
    interregnum = None
    reparandum_source = variety_nl
    if is_interrug:
        if _INTERRUG_LITERAL not in variety_nl:
            # Shouldn't happen (verified 0/2508 missing) — fail loud rather
            # than silently mis-scoring.
            raise ValueError(f"{_INTERRUG_LITERAL!r} not found in {variety_nl!r} "
                              f"(condition={condition!r})")
        interregnum = "I mean"
        # Strip ALL occurrences: a row with a genuine double-insertion defect
        # (see docstring) still collapses to its non-interrug sibling text,
        # which is exactly what we want to diff against gold.
        reparandum_source = variety_nl.replace(_INTERRUG_LITERAL, "")

    span, n_blocks = diff_inserted_span(gold_nl, reparandum_source)
    return GroundTruth(reparandum=_strip_edges(span), interregnum=interregnum,
                        repair=gold_nl, suspect=(n_blocks > 1))


if __name__ == "__main__":
    # Verification step from the plan: dry-run against known Q3 examples.
    cases = [
        ("Tom threw a rock at Mary, but it didn't hit her.",
         "Jerry, Tom threw a rock at Mary, but it didn't hit her.", "repair_head"),
        ("Tom threw a rock at Mary, but it didn't hit her.",
         "Tom threw a book, a rock at Mary, but it didn't hit her.", "repair_mid"),
        ("Tom threw a rock at Mary, but it didn't hit her.",
         "Tom threw a rock at Mary, but it didn't hit him, her.", "repair_tail"),
        ("Tom threw a rock at Mary, but it didn't hit her.",
         "Jerry, I mean, Tom threw a rock at Mary, but it didn't hit her.", "repair_head_interrug"),
        ("Tom threw a rock at Mary, but it didn't hit her.",
         "Tom threw a book, I mean, a rock at Mary, but it didn't hit her.", "repair_mid_interrug"),
        ("Tom threw a rock at Mary, but it didn't hit her.",
         "Tom threw a rock at Mary, but it didn't hit him, I mean, her.", "repair_tail_interrug"),
        ("Is it true that you are in love with me?",
         "Is it funny, true that you are in love with me?", "repair_head"),
        ("Is it true that you are in love with me?",
         "Is it true that I am, you are in love with me?", "repair_mid"),
        ("Is it true that you are in love with me?",
         "Is it true that you are in love with her, me?", "repair_tail"),
        ("My house is on the south bank of the Thames.",
         "Your house, my house is on the south bank of the Thames.", "repair_head"),
    ]
    for gold, variety, cond in cases:
        gt = extract_ground_truth(gold, variety, cond)
        print(f"[{cond:20s}] reparandum={gt.reparandum!r:30s} interregnum={gt.interregnum!r}")
