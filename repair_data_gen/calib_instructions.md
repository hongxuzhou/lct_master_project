# Calibration annotation

150 pairs, drawn evenly across the model's score range. The score is **not**
shown: it is in `calib_key.tsv`, to be joined after you finish. Seeing it first would
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
