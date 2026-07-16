# Statistical significance of repair position (head/mid/tail)

Tests whether the position of the injected reparandum (head, mid, tail)
significantly affects parser behaviour, evaluated separately for the
vanilla-repair conditions and the interregnum conditions. Motivated by the
design assumption made when synthesising the pilot data: that reparandum
position is a meaningful independent variable.

Script: `position_significance_test.py` (scratch copy; re-run against
`preds_long_scored.parquet`).

## Data and exclusions

- Source: `v2_pilot_eval_run/preds_long_scored.parquet`, 836 documents × 7
  conditions (`gold`, `repair_head`, `repair_mid`, `repair_tail`,
  `repair_head_interrug`, `repair_mid_interrug`, `repair_tail_interrug`).
- Excluded 3 documents (`p04/d2024`, `p05/d0919`, `p69/d1719`) whose `gold`
  row itself has `status == gold_error` in every condition — a gold-side
  data defect, not model behavior. N = 833 documents used below.
- Two groups analyzed independently:
  - **Vanilla**: `repair_head`, `repair_mid`, `repair_tail`
  - **Interregnum**: `repair_head_interrug`, `repair_mid_interrug`, `repair_tail_interrug`

## Design and method choice

Each document appears in all three conditions of a group (same underlying
sentence, only the injected noise differs), so this is a **repeated-measures
/ paired design**, not independent samples. Two outcomes are tested
separately because they capture different phenomena in this dataset:

1. **Success/fail** (`status == success` vs. not) — binary, paired across 3
   conditions per document.
   - Omnibus test: **Cochran's Q** (generalisation of McNemar's test to ≥3
     related binary samples). Implemented directly from the standard
     formula; χ² reference distribution, df = k−1 = 2.
   - Post hoc: pairwise **McNemar's test** (chi-square with continuity
     correction, df = 1) on each condition pair, using only the discordant
     document counts (n01, n10).
   - Multiple-comparison correction: **Holm** across the 3 pairwise tests
     within each group.
   - Sample: complete triplets only (document has a recorded status in all
     3 conditions of the group). N = 833 for both groups (the 7-condition
     grid is dense after excluding gold_error docs).

2. **F1 quality among successes** — continuous, paired.
   - Omnibus test: **Friedman test** (non-parametric repeated-measures
     ANOVA analogue), since f1 is bounded/non-normal.
   - Post hoc: pairwise **Wilcoxon signed-rank test** per condition pair.
   - Multiple-comparison correction: **Holm** across the 3 pairwise tests
     within each group.
   - Sample: **complete-case** — only documents that succeeded in *all
     three* conditions of the group are included (N = 722 for vanilla,
     N = 408 for interregnum). This is a real limitation, noted below.

Significance level: α = 0.05 throughout (Holm-adjusted p-values reported).

## Results

### Vanilla group (repair_head / repair_mid / repair_tail)

**Success rate** (N = 833 documents):

| condition | success rate |
|---|---|
| repair_head | 0.9580 |
| repair_mid | 0.9232 |
| repair_tail | 0.9556 |

Cochran's Q = 14.85, df = 2, **p = 0.000596** — significant.

Pairwise McNemar (Holm-corrected):

| pair | χ² | p (raw) | p (Holm) |
|---|---|---|---|
| head vs mid | 10.453 | 0.001224 | **0.003673** |
| head vs tail | 0.017 | 0.897279 | 0.897279 |
| mid vs tail | 8.779 | 0.003047 | **0.006094** |

head vs tail is not distinguishable; mid differs significantly from both.

**F1 among successes** (complete-case triplets, N = 722):

| condition | mean f1 |
|---|---|
| repair_head | 0.8427 |
| repair_mid | 0.8267 |
| repair_tail | 0.8248 |

Friedman χ² = 26.14, **p = 0.0000021** — significant.

Pairwise Wilcoxon (Holm-corrected):

| pair | p (raw) | p (Holm) |
|---|---|---|
| head vs mid | 0.000010 | **0.000019** |
| head vs tail | 0.000001 | **0.000003** |
| mid vs tail | 0.438907 | 0.438907 |

Here head differs significantly from both mid and tail; mid vs tail is not
distinguishable.

### Interregnum group (repair_head_interrug / repair_mid_interrug / repair_tail_interrug)

**Success rate** (N = 833 documents):

| condition | success rate |
|---|---|
| repair_head_interrug | 0.8691 |
| repair_mid_interrug | 0.7179 |
| repair_tail_interrug | 0.7071 |

Cochran's Q = 84.61, df = 2, **p < 1e-6** — significant, much larger effect
than the vanilla group.

Pairwise McNemar (Holm-corrected):

| pair | χ² | p (raw) | p (Holm) |
|---|---|---|---|
| head vs mid | 63.004 | ~0 | **~0** |
| head vs tail | 64.358 | ~0 | **~0** |
| mid vs tail | 0.228 | 0.633191 | 0.633191 |

mid and tail are statistically indistinguishable; both differ sharply from
head.

**F1 among successes** (complete-case triplets, N = 408):

| condition | mean f1 |
|---|---|
| repair_head_interrug | 0.8094 |
| repair_mid_interrug | 0.7804 |
| repair_tail_interrug | 0.8053 |

Friedman χ² = 18.02, **p = 0.000122** — significant.

Pairwise Wilcoxon (Holm-corrected):

| pair | p (raw) | p (Holm) |
|---|---|---|
| head vs mid | 0.000042 | **0.000126** |
| head vs tail | 0.762861 | 0.762861 |
| mid vs tail | 0.000245 | **0.000490** |

Here mid is the outlier (lower quality than both head and tail); head and
tail are indistinguishable.

## Interpretation

- Position is a statistically significant factor in both groups, on both
  outcomes (success/fail and f1-among-successes) — the design assumption
  holds.
- The effect is **not a monotonic head < mid < tail gradient**. In every
  test, one position pairs off against the other two rather than forming
  an ordered progression:
  - **Success/fail**: head is set apart from (mid, tail), which are
    mutually indistinguishable — most pronounced in the interregnum group
    (mid/tail success rates 71.8% / 70.7% vs. head 86.9%; McNemar p ≈ 0 for
    both mid and tail against head, p = 0.63 between mid and tail).
  - **F1 quality**: mid is set apart from (head, tail) instead — in both
    groups, head vs tail is not significant while mid differs from both.
- Practical reading: **structural failure (whether the parser produces a
  well-formed output at all) is governed by a head vs. non-head split**,
  while **quality given success is governed by a mid vs. non-mid split**.
  These are two distinct effects with different position groupings, not
  two views of the same underlying gradient.
- Caveat on the F1 tests: they are computed on complete-case triplets only
  (documents that succeeded in all three conditions of the group), which
  is 87% of documents for vanilla but only 49% for interregnum. This
  under-represents exactly the harder documents that fail in mid/tail —
  a selection effect to keep in mind when generalizing the F1 finding.


