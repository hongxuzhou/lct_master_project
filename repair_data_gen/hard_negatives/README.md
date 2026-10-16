# Hard negatives: train hold-out

66 PMB 5.1.0 train gold documents that look like a replacement self-repair
on the surface but are not one ("I am from Rio de Janeiro, Brazil."). They
are held out of every training set, so that a hard-negative test built from
them measures generalisation, not memory. Frozen 2026-10-16.

`train_holdout.tsv` sha256
`9a29eab567953f7e1df42194328e8ebbf58690314c2f118dd3612546bce91f7d`
(also `SHA256` in `holdout.py`).

## Rule for every builder

Any script that builds training data from PMB train gold reads the list with
`holdout.load()` and leaves these documents out. It does not copy the ids.
Done so far:

| builder | status |
|---|---|
| RSR source pool (`../rsr_generation/select_docs.py`) | reads `load()` |
| base RSR train set (`../rsr_generation/dataset/assemble.py`) | reads `load()`, asserts none present |
| fluent PMB part of C1 / C2 / C3 | not built yet; must read `load()` |

Train after the hold-out: 9,552 − 66 = **9,486** documents. All conditions
use this same partition.

## How the list was made

1. **Candidates.** Every train gold document whose text has a comma: 352.
2. **Labels.** Claude read all 352 and labelled each by the criteria of
   `documentation/model_experiment_pipeline.md` §7.1: keep apposition,
   asyndetic coordination and lists; leave out tag questions, fronted
   adverbials, conditional and other subordinate clauses, and dates. Also
   left out, as not surface-similar to a repair: clause coordination with
   `but`/`and`/`so`, `, too`, interjections, parentheticals, relative
   clauses, quotations and thousands separators.
3. **Held out:** every label except `excluded`.

| label | docs | example |
|---|---:|---|
| apposition | 24 | Al Gore was born in Washington, DC. |
| list | 25 | Tom is allergic to bees, hornets and wasps. |
| asyndetic | 5 | She wore a long, loose coat. |
| contrast_correction | 7 | She is not a nurse, but a doctor. |
| borderline | 5 | I am very, very sorry. / Steffi Graf defeated Conchita Martinez of Spain 6:3, 6:1. |
| **held out** | **66** | |
| excluded | 286 | Tom is angry, isn't he? |

`contrast_correction` and `borderline` are outside the §7.1 criteria. They
are held out anyway: whether they join the test set is decided later, but a
document trained on can never be used.

No vocatives ("Jason, ...") occur in train, dev or test gold.

## Not done

- The test side: selecting from these 66 plus the test_standard candidates
  (§7.1), with gold SBN, into the hard-negative test set. Dev is not used.
- One annotator (Claude), no second pass. Labels can be reviewed in
  `train_comma_labels.tsv`; a change is a new list, a new sha256 and a
  rebuild of everything that reads it.
- The design record's earlier figure of 105 candidates came from a rule that
  was not written down and could not be reproduced; this list replaces it.

## Files

| file | what |
|---|---|
| `train_comma_labels.tsv` | all 352 candidates with label (the authored record) |
| `train_holdout.tsv` | the 66 held-out documents (built from the labels) |
| `holdout.py` | `build`, `check`, and `load()` for builders |
