# NXT comparison: specification

Fixed 2026-10-17, before any number below was computed. Selection rules
(§1-2, `select_1to1.py`) frozen 2026-10-17 before the hand-check sample
was drawn: 1,858 real items, 3,249 of 3,489 synthetic. Rules are not
changed after the hand check (§3) except as §3 allows. Replaces the
2026-10-06 plan (`todo.md`, archived).

## Purpose and scope

A short descriptive comparison of the frozen train set
(`../dataset/train_rsr.base.tsv`) with real replacement self-repairs (RSR)
from NXT Switchboard. It supports a secondary claim (representativeness);
it is not a quality filter.

Scope: **one-word-for-one-word replacements only**. The lexical and
predictability measures are defined on word pairs, as in SPACER (Upadhye,
Li and Futrell, CMCL 2025). Multi-word replacements, restarts and
forwarding beyond one word may be added later as a separate step.

RSR definition used: one expression is replaced by another, with an
optional editing expression in between.

## 1. Real side (NXT)

Input: `swbd/disfluencies.tsv` and `swbd/sentences.tsv` from
`swbd_flatten.py` (reparandum, interregnum and repair already split).

Words are compared in lower case, with *a*/*an* treated as the same word.
The retrace is the longest common prefix of reparandum and repair, the
tail their longest common suffix (not overlapping the prefix). What is
left on each side is its **core**. A row is dropped, in this order, if:

1. it has an `issues` flag;
2. it is nested (`depth` > 0 or `n_children` > 0);
3. the repair is empty (abandoned);
4. reparandum and repair are identical (repetition);
5. one core is empty (insertion or deletion);
6. a core word is a fragment (starts or ends with `-`), a filled pause
   (tag UH), `MUMBLEx`, or a clitic (`'s n't 're 've 'll 'd 'm`);
7. a core is longer than one word.

The interregnum may be empty or not. Output: `swbd/real_1to1.tsv`, plus
the count left after each step (the funnel).

## 2. Synthetic side (our data)

The same rule. The reparandum span is the text removed from `nl_repair` to
give `nl_clean`, without its trailing comma. The repair span is the same
number of words read from `nl_clean` at that point (a one-to-one core
implies equal span length). Tokens follow Treebank style (`ptb_tokens`:
clitics split, punctuation dropped). Our data does not store where the
repair ends, so a repair longer than the reparandum would be cut short.
Three guards (added 2026-10-17, before any measure was computed) treat the
repair core as multi-word, dropping the item at step 7, when the core is
the last word of the window (no shared tail marks where the repair ends)
and the word after it continues the same expression:

- both are capitalised, the next word not being *I* (a name: *Paris, New
  York*);
- the core is a modal and the next word a verb (*went, will go*);
- for content word and tense items, the repair concept's WordNet lemma is
  multi-word and its second word starts the next word (*surrendered, gave
  in*; *broker, real estate agent*).

Where the repair spells out a contraction that `nl_clean` keeps (*I
bought, will buy* / *I'll buy*; 17 items), the prefix is the words
`nl_repair` and `nl_clean` share and the repair is read from `nl_repair`.

Reported: how many items pass, overall and per repair type.

## 3. Hand check (real side)

After the rules are frozen, 100 items are drawn from `real_1to1.tsv`
(seed 20261017) into `swbd/hand_check_1to1.tsv`. The user labels each
`rsr`, `not` or `unclear`, reading the sentence and the split. Reported:
precision (`rsr` / labelled, `unclear` left out) with a Wilson 95%
interval.

If precision is below 0.80, the rule is revised once, in writing, and a
new sample of 100 is drawn with a new seed. Otherwise the rules stay.

## 4. Measures (both sides, same code)

| measure | definition |
|---|---|
| PoS | Treebank tag of each core word mapped to coarse UPOS (one table for both). Real: NXT gold tags. Ours: spaCy `en_core_web_sm` on `nl_repair` (reparandum) and `nl_clean` (repair). Also open/closed class |
| WordNet relation | for core pairs with the same PoS in n/v/a: same lemma (an inflection change such as *wait*/*waited*; added 2026-10-17 before computing) > synonym (shared synset) > antonym > hypernym/hyponym (any depth) > co-hyponym (shared direct hypernym) > none; `n/a` otherwise. Recomputed on our side; the stored `wn_relation` is not used |
| Δ frequency | SUBTLEXus Zipf, repair − reparandum; out of vocabulary = missing. Zipf here is log10(SUBTLWF) + 3, without the +1 smoothing of Van Heuven et al. (2014); the difference cancels almost entirely in Δ |
| semantic distance | cosine distance of GloVe 6B 300d vectors; out of vocabulary = missing |
| phonemic distance | Levenshtein distance over CMU phones, stress removed, smallest over pronunciations; raw count |
| Δ surprisal | GPT-2 small, word surprisal (bits) with the Oh & Schuler (2024) correction; context = the fluent sentence prefix before the core (retrace included; other disfluencies, UH words and fragments removed on the real side); S(repair) − S(reparandum) |

Code: `lexical.py` and `score_lm.py`, fed by the new selection.

## 5. Report

- One table: median and IQR per side and Cliff's δ for the four continuous
  measures; proportions per side for PoS, open/closed class and WordNet
  relation.
- Sensitivity: the real-side figures again on the items labelled `rsr`.
- The funnel (§1) and our coverage (§2).

No significance tests, no thresholds for "natural", no regression.

## Expected before computing

- The real side is mostly closed class (SPACER: 344 of 576); ours is not,
  because our repair types were chosen for SBN-level changes.
- Our side has no synonyms: same-synset pairs were rejected at generation.

## Differences from SPACER

SPACER takes context from preceding turns; we use the sentence prefix only,
because our sentences have no discourse context. SPACER uses IPA
characters (`eng_to_ipa`) for phonemic distance and log10 counts for
frequency with unknown words set to 0; we use CMU phones, Zipf, and
missing values.

## Not done here

Multi-word cores, restart detection, nested disfluencies, classifiers or
LLM-based selection, a second language model, a human rating study.
`build_sets.py`'s typing is retired (its rules were tuned on synthetic
data); `swbd_flatten.py`, `lexical.py` and `score_lm.py` are reused.
