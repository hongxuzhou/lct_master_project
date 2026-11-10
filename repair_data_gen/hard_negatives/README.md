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
| unified fluent PMB part (`../training_dataset/build.py`) | reads `load()`, asserts 9,486 rows and no overlap |

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

The original §7.1 criteria focused on apposition, lists and asyndetic
coordination. For the diagnostic, retain all 66 held-out candidates that
have intelligible English, a defensible non-RSR reading and faithful,
scorable gold SBN. Do not rank or prune them by perceived difficulty.
Keep the original categories, including `contrast_correction` and
`borderline`, so their results can be reported separately.

No vocatives ("Jason, ...") occur in train, dev or test gold.

## Diagnostic acceptance

63 of the 66 original gold graphs pass the structural checks. The other
three are `p17/d2285` (unresolved index), `p23/d1469` (index overflow), and
`p69/d1719` (cycle and index overflow). The pass count does not establish
text and graph fidelity. Record any defects and validated fixes in a
separate evaluation record. Items not ready to score remain held out of
training.

The diagnostic export still needs text and meaning verification. A
`test_standard` supplement may be evaluated as a separate subset; dev
is not used. The 66-document training hold-out remains fixed.

## Files

| file | what |
|---|---|
| `train_comma_labels.tsv` | all 352 candidates with label (the authored record) |
| `train_holdout.tsv` | the 66 held-out documents (built from the labels) |
| `holdout.py` | `build`, `check`, and `load()` for builders |
