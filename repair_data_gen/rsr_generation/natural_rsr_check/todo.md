# Measuring the naturalness of LLM-generated RSR

Status: plan (2026-10-06). The Switchboard reference has been converted, and
what is compared (§2) and how the real set is selected (§5) were settled on
the same day. Nothing else has been run. Numbers marked "first look" were
computed that day from `results/meta-models_Muse-Glimmer-30B.train_full.checked.tsv`
and the converted reference, with the rough rules of §5 before their fixes.
They show the size of an effect, and will be recomputed once the rules are
fixed.

# 1. Why, and what this is not

The audit (`audit.py`) asks one annotator whether each repair is natural
(`natural`, threshold 0.80). One annotator, with no real repairs to anchor the
scale, is not enough for a paper. The plan is to compare the synthetic repairs
with real self-repairs from spontaneous English speech, and to repair the
human judgement with a design that has an anchor and several raters.

The real-speech reference is the NXT-format Switchboard corpus (Calhoun et al.
2010). It is used only as a reference. The conversion into two flat tables is
described in the docstring of `swbd_flatten.py`. Everything derived from the
corpus stays in `swbd/`, which git does not track and which never leaves this
machine.

These measures describe the data. None of them is used as a quality filter.
This keeps the earlier decision that surprisal and NLI scores are not a filter
(README, "Deferred").

# 2. What is compared

The model is given a gold sentence and a repair type. It then makes three
choices. It picks where to repair, it picks the reparandum, and it splits the
sentence into `nl_left` and `nl_right`, which sets how much of the sentence is
said again. Only these choices are compared with real repairs, and always
within one repair type.

Everything else is fixed by design. It differs from real speech for reasons
known in advance, so it is reported and not tested.

| Fixed by design | Reason |
|---|---|
| No interregnum | Inserted later by script (README, MVP) |
| No stutter, word fragment or nested repair | The task asks for one clean repair |
| One repair per sentence | Same |
| Mix of repair types | Set by quota, after the coverage principle (`DESIGN.md` §0) |
| Source sentences | PMB, closer to written text than to telephone speech |

The three choices give three questions, and a fourth asks about the whole item.

| Question | What is asked | Measures |
|---|---|---|
| Q1 Site | Where would a speaker repair? | A1, A2, B3 |
| Q2 Content | Given the repair, what could the speaker have said by mistake? | B1, B2 |
| Q3 Form | How much is said again, and how long is the reparandum? | C1, C3 |
| Q4 Overall | Can a reader or a classifier tell it from a real one? | D1, D2, D3, E |

Distribution measures (A to D) say whether the data is typical of real
repairs. Human judgement (E) says whether each item is acceptable. A repair can
be acceptable and still atypical, so both are reported.

The questions that apply depend on the type. For constant and tense the type
almost fixes the reparandum (another number, another tense), so only Q1 and
Q3 are compared. Q2 applies to content word, and in part to edge label.

Q1 is the question most affected by the difference between PMB and telephone
speech. It therefore compares how likely each part of speech is to be chosen,
given the words the sentence offers, and not the raw distribution of repaired
parts of speech.

# 3. Measures

## A. Site

**A1. Relative onset position.** Words before the repair onset divided by the
sentence length, from `repair_onset`. Source: own definition. No published
reference value exists, so the real side is computed on the reference corpus.
First look (median): content word 0.7, constant name 0.6, constant number
0.6, constant temporal 0.7, tense 0.4, forwarding 0.3.

**A2. Surprisal of the word after the repair.** `S_k = -log P(w_k | w_{k-1},
w_{k-2})` for the first word that follows the repair, given the prefix. Source:
Dammalapati, Rajkumar and Agarwal (2019), who define lexical surprisal after
Hale (2001) and find that the word following a disfluency has higher
surprisal, which they read as a sign of upcoming difficulty. They use a trigram
model trained on the Open American National Corpus. This is the origin of the
"surprisal given the prefix" idea.

## B. Content (content word first)

**B1. Four lexical features of the reparandum and the repair.** Source:
Upadhye, Li and Futrell (2025, SPACER), who compute them for single-word
substitution repairs from Switchboard.
- Word predictability, the log probability of the word given the preceding
  context, from GPT-2 small (Radford et al. 2019) through `minicons` (Misra
  2022).
- Word frequency in SUBTLEXus (Brysbaert and New 2009).
- Semantic distance, the cosine distance between GloVe vectors (Pennington et
  al. 2014).
- Phonemic distance, the Levenshtein distance between the IPA forms from the
  CMU Pronouncing Dictionary.

Word length is also recorded in SPACER. Report the difference between the
reparandum and the repair for predictability and frequency, and the distance
for the other two.

Resources used here (decided 2026-10-06, see §4): predictability from the
model of §4, not `minicons`; frequency from SUBTLEXus; semantic distance from
the 2014 GloVe `glove.6B.300d`; the CMU dictionary through NLTK. For a
multi-word core the features are taken on its head word. Words missing from
GloVe or the CMU dictionary are counted and reported per side.

Semantic distance stays on static vectors, not on a contextual embedding
model such as E5. What is measured is how close the two words are as
words, apart from any sentence. How well a word fits its context is already
measured by predictability, and a contextual measure would overlap with it.
Comparing two sentences that differ in one word mostly measures the shared
words (`archive/repair_data_gen/rule_pipeline/DESIGN.md` §5.2). Like any distributional measure, GloVe puts
antonyms close together, so it is read together with B2.

The 2014 vectors are used rather than the 2024 release (Carlson, Bauer and
Manning 2025). The newer vectors score lower on word similarity (300d,
Spearman: WordSim353 0.544 for 2014 against 0.486 for 2024 Wikipedia and
Gigaword, SimLex-999 0.371 against 0.338, MEN 0.737 against 0.690), and their
gain is on new words and named entities, which neither side needs.

**B2. WordNet relation between reparandum and repair.** `same_synset`,
`hypernym`, `hyponym`, `similar_to`, `other`, as `analyse_run.wn_relation`
already computes for the synthetic side. Source: own. On the real side there
are no sense labels, so the relation has to be taken over all senses of the
two lemmas, and the two sides are only comparable once that difference is
fixed. Shriberg (1994) is reported to describe substitution repairs as
replacing tokens that are closely related in syntax or meaning; this is to be
checked against the thesis before it is cited.

First look, with the all-senses relation on both sides (antonym and co-hyponym
added), content word, 507 real against 3,341 synthetic:

| Relation | Real | Synthetic |
|---|---|---|
| hypernym or hyponym | 13% | 14% |
| same synset | 10% | 9% |
| co-hyponym | 7% | 21% |
| antonym | 2% | 9% |
| other | 54% | 46% |

The synthetic side has three times as many co-hyponyms and five times as many
antonyms (book → magazine, small → big). The gap holds within one part of
speech: co-hyponyms are 30% of synthetic nouns against 11% of real nouns, and
antonyms 7% of synthetic verbs against none of the real ones. The real nouns
number only 90. On the real side "same synset" is mostly light verbs that
share senses in WordNet (have → make), not synonyms.

Levelt (1983) separates error repairs (the first word was wrong) from
appropriateness repairs (the first word was not wrong, only less fitting, for
example too general). The task asks for the first kind. Real substitutions mix
both, and text alone rarely tells them apart. The mix is accepted (decided
2026-10-06). As a rough check, hypernym and hyponym, the relations closest to
an appropriateness repair, have the same share on both sides in the first
look, so the mix does not seem to drive the difference.

**B3. Part of speech of the reparandum and repair.** SPACER reports the
distribution (determiners, pronouns and verbs are the most frequent). The
synthetic content-word repairs are restricted to content words, so compare
within the same part of speech. First look: 70% of the real content-word
repairs are verbs, and 69% of those have a light verb as the reparandum (have,
make, get, do). 60% of the synthetic ones are nouns. Part of this comes from
PMB, whose sentences offer more nouns, which is why Q1 conditions on the words
the sentence offers (§2).

## C. Form

**C1. Reparandum length in words.** Source for the real value: Hough and
Purver (2013), cited in Hough and Purver (2014, STIR), a mean of about 1.5
words over all repairs. That figure includes repetitions, so it is only a
rough bound. First look at the synthetic side: mean 1.18 words, 84% single
word. The real side has to be restricted to substitutions before comparing.

**C2. Interregnum rate.** Source: Hough and Purver (2013), cited in STIR, an
interregnum in about 15% of repairs. Not compared, because the synthetic base
repairs have none by design (5,984 of 5,984). The real rate is reported as
context and will matter for the later round with interregna. The NXT README
says interregna are not annotated. In fact the editing words sit at the start
of the NXT repair span, and `swbd_flatten.py` splits them off as whole INTJ or
PRN constituents or bare UH words. First look: 4,168 of 28,097 disfluencies
(14.8%) have one, close to the figure above.

**C3. Retracing and rough copy.** How much of the repair restates the
reparandum or the words just before it (for example a repeated determiner).
Source for the idea: the reparandum as a "rough copy" of the repair (Johnson
and Charniak 2004), and the observation that only 39% of repair alignment
structures occur at least twice in Switchboard (Hough and Purver 2013, cited
in STIR). The exact operationalisation, probably an alignment of reparandum
and repair, is still to be written.

**C4. Mix of repair types.** In the Switchboard training segments 65.39% are
repetitions, 18.99% substitutions and 15.62% deletions (Yang, Yang and Ma
2020). The synthetic data only has substitution-like types, so comparisons are
made inside the substitutions and the mix is reported only as context. First
look on the converted reference (top level, not nested): 59% repetition, 4%
repetition followed by more words, 6% with an empty repair, 31% the rest.

## D. Overall distinguishability

**D1. Real versus synthetic classifier.** Train a classifier to separate real
and synthetic repair windows and report the AUC. Case, punctuation and
spelling must be normalised first, because a classifier will otherwise learn
the format. Run it with and without feature groups to see what gives the
synthetic data away, and compare with a permutation baseline. This is common
practice for checking synthetic data. I found no repair-specific precedent, so
there is no source to cite for the design.

**D2. MAUVE.** Pillutla et al. (2021). Compute it on repair windows (the
reparandum, the repair and a few words around them), not on whole sentences.
Report a noise floor from two random halves of the real data. Secondary,
because it is sensitive to sample size and to the embedding model.

**D3. Detector recall.** Train a disfluency detector on real data and test it
on the synthetic repairs. Yang, Yang and Ma (2020) use generated disfluent
text for detection and judge naturalness by BLEU and sentence accuracy against
a reference, plus a human rating. A high recall can also mean the repairs are
easy copies, so this is secondary.

## E. Human judgement

- A blind mixed pool of real repairs and synthetic repairs, in the same
  normalised format, so the rater cannot use formatting cues.
- Two questions. The first is a three-level rating from Yang, Yang and Ma
  (2020): natural scores 1, unnatural 0.5, incomprehensible 0 (they used 100
  sentences and three Mechanical Turk responses each). The second is a forced
  choice between real and synthetic. The rating is the main question. The
  forced choice is secondary, because register alone can give a real item
  away (§5, Q4).
- Report the share of synthetic items judged real divided by the share of real
  items judged real. The real items set the ceiling, which the single-annotator
  audit does not have.
- At least three raters per item. Report ordinal Krippendorff's alpha, and
  model the ratings with item and rater random intercepts as SPACER does
  (Barr et al. 2013). Expect low agreement. SPACER reports a mean Cohen's kappa
  of 0.213, although for a different decision (whether to correct a word).
- Rating differences can be small. A study of LLM utterances with inserted
  disfluencies (arXiv:2412.12710) reports spontaneity of 2.58 against 2.91 on a
  five-point scale with 41 participants (p = 0.033). Plan the number of raters
  and items with that in mind.
- Text only. Real repairs are also marked by prosody, so this tests the
  transcript, not the speech. State this as a limit.

# 4. Language model and normalisation

## How surprisal is used

Surprisal is a description feature (A2, and predictability in B1). It is not a
score for a single repair.
- No direction is assumed. The papers show that disfluencies relate to
  surprisal, not whether a natural repair should be above or below a baseline.
  The real distribution decides.
- Values depend on the language model. Dammalapati et al. use a trigram model
  and SPACER uses GPT-2, so absolute values are not comparable across studies.
  Use one model for both sides, name it, and prefer differences (reparandum
  against repair) over absolute values.
- Expectation to check, with no source: LLM-written text is more predictable
  than human text, which would shift the synthetic values.

## The model only reads fluent text (decided 2026-10-06)

- B1: the reparandum word and the repair word are scored at the same
  position, given the same left context, which ends before the repair site.
  This is the SPACER design, and it keeps the context constant as
  `archive/repair_data_gen/rule_pipeline/DESIGN.md` §5.2 requires.
- A2: the word after the repair is scored given the fluent left context, with
  the reparandum removed. This measures whether the repair comes before a
  hard word (Dammalapati et al.). With the reparandum left in, it would
  measure how the model reacts to a disfluent string, which depends on how
  each side writes the disfluency.
- So how a disfluency is written never reaches the model.
- Context is the current sentence only, on both sides, since the synthetic
  items have no preceding turns. A BOS token is put first, so a repair at
  the start of the sentence can still be scored.
- Sentences differ in length (median 17 words on the real side, 7 on the
  synthetic). Compare differences at the same position, or put the length
  of the left context in the model as a covariate.

## Normalisation

Two levels. Treebank tokens are used for alignment, typing and word counts
(C1, C3), as `build_sets.py` does. The text given to the language model is
built as follows, the same way on both sides:
- contractions joined again (`do n't` → `don't`);
- no punctuation at all on either side (decided 2026-10-06). Restoring the
  transcribers' punctuation would read better for a model trained on written
  text, but the two sides would then follow different punctuation
  conventions;
- first letter of the sentence upper case, and the repair's first letter
  lower case unless it is a name or "I" (16% of synthetic repairs start
  with a capital, as in "She died, He died");
- digits written out as words on the synthetic side (320 items), since
  transcripts contain no digits (`num2words`);
- real side only: titles in capitals turned into title case (STAR TREK, 1,977
  tokens); the two encoding leftovers fixed (`appliqu3ed`, `fianc3e`);
  reparanda, interregna, UH words and word fragments of other disfluencies
  removed from the left context; items with `MUMBLEx` in the context
  dropped;
- synthetic side only: the ", " the harness puts at the repair onset is not
  part of any fluent text. It matters only for presenting items in Q4,
  which is decided with E.

## Model (decided 2026-10-06)

Main model: Pythia-160m-deduped. Check: GPT-2 small, the SPACER model. If
the two disagree, that is reported.
- Larger models fit human reading times worse, in part because they
  memorise training text (Oh and Schuler 2023). A small model is the
  better choice for this kind of measure.
- Training-data overlap is a concrete risk here. PMB sentences come mostly
  from Tatoeba. If the model has seen the source sentence, the gold word,
  which is the synthetic repair, becomes more predictable, and the
  repair-against-reparandum difference grows on the synthetic side only. An
  infini-gram check on PMB sentences of seven words or more found 12 of 47 in
  the Pile (Pythia's training data, mostly once or twice) and 40 of 48 in
  Dolma, a 2024 web corpus. Newer models trained on newer web data are
  therefore riskier. Pythia's data can be searched, GPT-2's cannot.
- Every synthetic source sentence is looked up in the Pile with infini-gram.
  Every measure that uses the model is reported on all items and again
  without the items found. The real side cannot be checked this way, since
  its sentences may not be sent to an outside service. This is stated as a
  limit.
- Not used: GPT-2 large (larger, more memorisation), recent models such as
  Qwen3 or Gemma 2 (newer web data), any model trained on Switchboard (the
  reference would leak into the scorer). A model close to the generator
  would also make the synthetic text look predictable for the wrong reason.
- Word probability: the sum over a word's subword tokens, with the
  correction for leading whitespace of Oh and Schuler (2024). Implemented
  directly with `transformers`.

## Details settled in the code

- Predictability is scored on the whole core (the sum over its words). The
  lexical features (frequency, GloVe, CMU) are taken on the core's head word.
- If a core starts with a clitic ("'s"), the word before it is scored with
  it on both sides ("that's" against "that was").
- Digits are read as cardinals, so a date comes out as "October two" rather
  than "October second". This touches only constants.
- Real side: discourse markers that the Treebank keeps as words ("you
  know") stay in the context; only UH words are removed.

## Where it runs

On the LST cluster, both sides. The university cluster counts as the
author's own space, so `swbd/` may be copied there (decided 2026-10-06).
Nothing is downloaded locally.

# 5. The real comparison set

The real set is defined by the repair types of `prompts/base/repair_notation.md`
§4, because the synthetic data is drawn from the repairs the notation can
represent. The usual Switchboard categories (repetition, substitution,
deletion) are only used to drop items. Every step selects from the converted
reference and none rewrites it: a normalised repair would be one that no
speaker produced. SPACER also drops repetitions and false starts instead of
editing them.

## Steps

1. **Structure.** Keep disfluencies at the top level that contain no nested
   disfluency, have no entry in `issues`, have no word fragment (a word ending
   in `-`) on either side, and have a non-empty repair. An interregnum does not
   exclude an item. It is kept as a covariate.
2. **Alignment.** Remove the words that reparandum and repair share at the
   start and at the end. What remains is the changed core. The shared start is
   the retrace, measured in words (C3). An empty core is a repetition or an
   insertion and is dropped. A core of more than three words is taken as a
   restart and dropped.
3. **Type.** Map the core to a notation type from its Treebank part of speech
   (gold in this corpus) and its WordNet lemma. Cores the notation cannot
   represent are counted and not compared: function words, and the
   separators of the notation's §7 (negation, quantifiers, modals,
   connectives).
4. **Check by hand.** All items of the small types, and 100 random
   content-word items. The judgement is whether the item belongs to its type,
   yes or no. It is not a naturalness judgement.

The same rules run on the synthetic side. Every synthetic item should land in
the type it was given, and those that do not are inspected. A mismatch is
either a gap in the rules or an oddity in the synthetic data.

The rules are in `build_sets.py` (built 2026-10-06), which writes
`swbd/real_set.tsv`, `swbd/hand_check.tsv` and `synth_typed.tsv`. Changes from
the first draft, from reading its output on both sides:
- contractions count as their full form, so `he is → he 's` and `did n't →
  did not` are repetitions (Switchboard tags `'s` as BES or HVS);
- "a" and "an" count as one word, so `an avalanche, a landslide` retraces;
- a cut-off word is dropped only if it is not itself a word (`coro →
  coronary` is dropped, `pen → pencil` and `memorize → memorized` are not);
- tense needs the same main verb (the last verb of each span) and only verb
  forms in the core; changes that are not clearly past, present or future
  (`make → making`, `go → 've gone`) are kept with the note `aspect`, and
  copula changes with the note `be`; the hand check decides both (§4.3: a
  copula counts with a noun predicate, which has `be.v.01`, and not with an
  adjective);
- forwarding is a new subject followed by the same verb, and is tested
  before constant (`Jim has, Tom has`);
- determiners do not decide content or constant (`the plastic stool, the
  chair`); multi-word cores of content words count (`train station → bus
  stop`, note `multiword`);
- nationality and language words (`Swedish → Finnish`) are typed as content
  words with the note `proper`, because PMB itself writes some as names and
  some as concepts;
- edge label: whether the PP attaches to a verb is left to the hand check,
  since every edge-label item is checked anyway.

The same rules type the synthetic run. The share that lands in its given type
is 93% for constant, 92% for content word, 95% for forwarding and 98% for
tense (430 of 5,984 miss). Most misses are tagger errors on short sentences
(noun against adjective) and the nationality words. A few are defects of the
synthetic data, for example "I've never been ski, skiing", "That is, building
is an old castle", and "John borrowed, I have borrowed", which mixes
forwarding with tense.

## First look

From 24,435 top-level disfluencies without issues (`swbd/real_set.tsv`):

| Outcome | Items |
|---|---|
| dropped: contains a nested disfluency, word fragment | 2,735, 1,952 |
| dropped: repetition, empty repair, insertion | 13,501, 1,355, 1,503 |
| dropped: restart, cut-off word | 371, 2 |
| outside the notation: function word, negation | 1,091, 82 |
| not yet placed: longer cores, 1:1 across parts of speech | 724, 120 |
| content word | 591 |
| tense | 187 |
| forwarding | 109 |
| edge label | 75 |
| constant | 37 |

Function-word substitutions (`the → a`, `he → his`, `but → so`) outnumber
content-word ones by almost two to one. The notation covers only part of
real substitutions, which belongs in the scope section of the paper. The
longer cores are dropped from the main analysis. A sample of 50 estimates how
many of them belong to a notation type, and the estimate is reported as a
limit.

Only the content word has enough items for distribution measures. The other
types are reported with counts and descriptive statistics. Of the 591
content-word items, 248 have a light verb as the head of the reparandum.

The hand check (`swbd/hand_check.tsv`, 558 items) holds all tense,
constant, edge-label and forwarding items, 100 random content-word items and
50 random longer cores. Each row carries the question to answer; fill `ok`
with y or n, and `comment` where needed.

## Extra conditions per question

- **Q1 to Q3** use the set above.
- **Q4** compares whole items, so the presentation must match as well. Other
  disfluencies in a real sentence would give it away, so Q4 also requires no
  interregnum, the target as the only disfluency in its sentence, and no UH
  word anywhere in the sentence. Both sides are shown as a window of a few
  words around the repair, which also reduces the length cue. This is a
  filter, not a clean-up. Its cost is a bias toward short sentences and fluent
  speakers. Register stays a cue that no filter removes, which is why the
  forced choice in E is secondary. Content word: 213 of 591 items meet these
  conditions (`q4_ok`). Enough for about 150 items per side.

# 6. Order of work

- [x] Convert the NXT reference to flat tables (`swbd_flatten.py`).
- [x] Decide what is compared (§2) and how the real set is selected (§5).
- [x] Decide how to get the real interregnum rate (C2).
- [x] Fix one normalisation shared by both sides (case, punctuation,
  contractions) and one language model (§4).
- [x] Implement the normalisation of §4 and the word-probability scorer
  (`lm_items.py`, `score_lm.py`, `lexical.py`; cluster jobs in `hpc/`).
  Tested on the laptop without model weights: a two-layer random GPT-2 with
  the real tokeniser gives the same values in batches of 32 and of 1, and
  the same as a whole-string computation of the correction.
- [ ] Look up every synthetic source sentence in the Pile (infini-gram).
- [ ] On the cluster (`hpc/README.md`): setup job (venv, Pythia-160m-deduped,
  `glove.6B.300d`, SUBTLEXus, `num2words`; GPT-2 small is already in
  `/scratch/common_models`), then the run job.
- [x] Apply the rule fixes of §5, build the real set, and run the same rules
  on the synthetic side (`build_sets.py`).
- [ ] Check by hand (§5, step 4), and sample 50 of the longer cores.
- [ ] Compute A1, A2, B1, B2, B3, C1, C3 on both sides for the content word.
  Report effect sizes with intervals, not only p-values.
- [ ] Human study (E). Proposed size is about 150 items per side with at least
  three raters each. Revisit once the effect size from the distribution
  measures is known.
- [ ] Secondary: D1, then D2 and D3 if time allows.

Left out for now: non-English corpora (DISCO, arXiv:2310.16749; the Corpus of
Spontaneous Japanese, Maruyama; Mandarin repair studies by Tseng 2003 and Chen
et al. 2012; Verbmobil for German). Their value is to show which properties
hold across languages, and none of them can stand in for the English
reference. I read none of them in full.

# 7. Limits and open points

- Domain. The synthetic sentences come from PMB, which is closer to written
  text, and the reference is telephone speech. Features are therefore taken in
  a window around the repair, and relative measures are preferred. The
  difference remains and is stated in the paper.
- Error and appropriateness repairs are mixed on the real side (B2).
- The real reference is only an approximation of what a speaker would say. The
  repairs it contains are the ones speakers made, so it holds no information on
  repairs a speaker could have made and did not.
- SPACER size: the paper reports 1,056 speaker utterances and 5,808
  comprehender responses. An earlier draft of this note gave 576 repaired
  utterances, which is to be checked against the paper.
- To verify before citing: the Shriberg (1994) statement in B2, the Levelt
  (1983) distinction in B2, which is written from memory, and the Hough and
  Purver (2013) numbers, which are taken from STIR.
- Licence. The reference corpus is CC BY-NC-SA 3.0. Nothing derived from it
  leaves this machine and the university cluster: not in git, not on Hugging
  Face, not in any cloud tool.
  Only summary statistics are reported.

# References

Barr, D. J., Levy, R., Scheepers, C., and Tily, H. J. (2013). Random effects
structure for confirmatory hypothesis testing: Keep it maximal. *Journal of
Memory and Language*, 68(3), 255-278.

Biderman, S., Schoelkopf, H., Anthony, Q., et al. (2023). Pythia: A suite for
analyzing large language models across training and scaling. *Proceedings of
ICML*.

Brysbaert, M. and New, B. (2009). Moving beyond Kucera and Francis: a critical
evaluation of current word frequency norms and the introduction of a new and
improved word frequency measure for American English. *Behavior Research
Methods*, 41(4), 977-990.

Calhoun, S., Carletta, J., Brenier, J. M., Mayo, N., Jurafsky, D., Steedman,
M., and Beaver, D. (2010). The NXT-format Switchboard Corpus: a rich resource
for investigating the syntax, semantics, pragmatics and prosody of dialogue.
*Language Resources and Evaluation*, 44(4), 387-419.

Carlson, R., Bauer, J., and Manning, C. D. (2025). A new pair of GloVes.
arXiv:2507.18103.

Dammalapati, S., Rajkumar, R., and Agarwal, S. (2019). Expectation and
locality effects in the prediction of disfluent fillers and repairs in English
speech. *Proceedings of the NAACL Student Research Workshop*, 103-109.

Hale, J. (2001). A probabilistic Earley parser as a psycholinguistic model.
*Proceedings of NAACL*, 1-8.

Hough, J. and Purver, M. (2014). Strongly incremental repair detection.
*Proceedings of EMNLP*. arXiv:1408.6788. (Cites Hough and Purver 2013 for the
figures in C1, C2 and C3.)

Johnson, M. and Charniak, E. (2004). A TAG-based noisy-channel model of speech
repairs. *Proceedings of ACL*, 33-39.

Levelt, W. J. M. (1983). Monitoring and self-repair in speech. *Cognition*,
14(1), 41-104.

Liu, J., Min, S., Zettlemoyer, L., Choi, Y., and Hajishirzi, H. (2024).
Infini-gram: Scaling unbounded n-gram language models to a trillion tokens.
arXiv:2401.17377.

Misra, K. (2022). minicons: Enabling flexible behavioral and representational
analyses of transformer language models. arXiv:2203.13112.

Oh, B.-D. and Schuler, W. (2023). Why does surprisal from larger
Transformer-based language models provide a poorer fit to human reading
times? *Transactions of the Association for Computational Linguistics*, 11.

Oh, B.-D. and Schuler, W. (2024). Leading whitespaces of language models'
subword vocabulary pose a confound for calculating word probabilities.
*Proceedings of EMNLP*.

Pennington, J., Socher, R., and Manning, C. D. (2014). GloVe: Global vectors
for word representation. *Proceedings of EMNLP*, 1532-1543.

Pillutla, K., Swayamdipta, S., Zellers, R., Thickstun, J., Welleck, S., Choi,
Y., and Harchaoui, Z. (2021). MAUVE: Measuring the gap between neural text and
human text using divergence frontiers. *Advances in Neural Information
Processing Systems 34*. arXiv:2102.01454.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I.
(2019). Language models are unsupervised multitask learners. OpenAI.

Shriberg, E. (1994). *Preliminaries to a theory of speech disfluencies*. PhD
thesis, University of California, Berkeley.

Upadhye, S., Li, J., and Futrell, R. (2025). SPACER: A parallel dataset of
speech production and comprehension of error repairs. arXiv:2503.16745.

Yang, J., Yang, D., and Ma, Z. (2020). Planning and generating natural and
diverse disfluent texts as augmentation for disfluency detection.
*Proceedings of EMNLP*, 1450-1460.

arXiv:2412.12710. Enhancing naturalness in LLM-generated utterances through
disfluency insertion.
