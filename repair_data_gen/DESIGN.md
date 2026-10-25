# Design of the repair-aware SBN data

What the data is for, what it contains, and how its quality is controlled.
Individual decisions and their reasons are in `DECISIONS.md`; how to run the
pipeline is in `rsr_generation/README.md`. The earlier design, written for a
rule-based generator, is in `archive/repair_data_gen/rule_pipeline/`.

## 1. Purpose

A semantic parser should handle replacement self-repair (RSR): a speaker says
something, withdraws it and says something else in its place.

> The cat chases, I mean, hunts the rat.

The Parallel Meaning Bank (PMB 5.1.0) has about 9,500 English training
sentences with gold SBN graphs, all fluent written text. The repair data is
therefore made from PMB: take a gold sentence, add a first version that the
speaker withdraws, and annotate the result with the `CORRECTION` /
`CONJUNCTION` construction (`documentation/knowledge_base/repair_sbn_notation.qmd`).

**Coverage, not frequency.** PMB contains no repairs, so there is no natural
frequency to copy. The data aims to cover every repair type the notation
licenses in learnable quantity, with types balanced so that differences
between them are not differences in data size. Whether the result resembles
real speech is measured separately (Switchboard comparison), not built in.

## 2. The representation

The withdrawn version goes into a `CORRECTION` box and the final version into
a `CONJUNCTION` box, side by side; removing the `CORRECTION` box (and
merging any dummy back) gives back the gold graph. Edges that would break this are resolved by five devices:

| device | used when |
|---|---|
| ① drop a role | the withdrawn version never got to an argument the repair has |
| ② invert a role | a concept before the repair points at the repaired concept |
| ③ dummy `entity.n.01` with `EQU` | a role itself is repaired (edge label) |
| ④ move concepts forward | an adjective, adverb or verb would have no edge otherwise |
| ⑤ anchor `entity.n.01` | a noun would have no edge otherwise; both versions point to the anchor |

Full rules and examples: `rsr_generation/prompts/base/repair_notation.md`.

## 3. Repair types

| type | what is replaced | example |
|---|---|---|
| content word | a concept, by another of the same part of speech (not a synonym) | A baby is sleeping in the basket, the cradle. |
| constant: name | a `Name` value | This city is located in Germany, France. |
| constant: number | a `Quantity` or number value | We have three, two boats. |
| constant: temporal | a clock time, day, month or year | I will be sixteen in April, in May. |
| tense | the tense operator of a verb's time (`TPR`, `EQU`, `TSU`) | Hebrew was, is my mother tongue. |
| forwarding | the subject, with the verb said again | John lent, I lent him a CD. |
| edge label | a role (`Source` / `Destination`) or a place (a spatial operator against a plain role) | He jumped from, into the train. / He is lying under, on the bench. |

Every type is a single-sentence repair. Repairs that need context beyond
one sentence (anaphora, a repair across speaker turns) are not generated.

## 4. Pipeline

| step | status |
|---|---|
| 1. Select PMB documents with a site for the type | done (train) |
| 2. Generate: an LLM writes the repaired sentence and SBN | done (train) |
| 3. Check every item automatically; drop failures | done (train) |
| 4. Select edge-label items by hand for naturalness | done (train) |
| 5. Align the text outside the repair with PMB; sample about 500 per type | done: `rsr_generation/dataset/train_rsr.base.tsv` |
| 6. Insert editing expressions | scripted, 2026-10-24 |
| 7. Repeat 1–6 on PMB dev and test | later |

**Why an LLM.** A rule-based generator reached content words only:
constants, tense, edge labels and forwarding each need their own operator
(`archive/repair_data_gen/rule_pipeline/FINDINGS.md`). The LLM gets the
gold sentence and SBN, the repair type and the notation, picks the site
and the first version, and writes the SBN itself, in one attempt, with a
WordNet lookup as its only tool. The model's choices are what is being
tested, so it gets no candidate list.

**Base first.** The model writes repairs without an editing expression ("The
cat, the dog barks"). Adding "I mean" does not change the graph, so a script
inserts it later (step 6), which also yields matched pairs with and without
an expression.

## 5. Quality control

Three layers, each answering a different question:

| layer | question | scope | effect |
|---|---|---|---|
| automatic checks | is the SBN well formed, and does it match the gold and the sentence? | every item | failures are dropped |
| blind human audit | is the meaning right, and is the repair natural? | samples of passing items | decides whether a type is used |
| selection | is this item natural? | every edge-label item | only "yes" items are kept |

A type goes into the data only if it meets thresholds that were set before
the first audit: checks ≥ 95%, no answer ≤ 3%, meaning ≥ 90%, natural ≥
80%. Edge label missed the naturalness threshold and was moved to
selection rather than given a lower bar. Results:
`rsr_generation/dataset/README.md`.

"Natural" means acceptable to a fluent speaker: grammatical, recognisable as
a repair, with both versions recoverable. It does not mean typical of speech.

## 6. Editing expressions

Built by `rsr_generation/dataset/insert_editing_expressions.py`:

- Markers for training: *I mean*, *no*, *no wait*, *actually*. *Sorry* is
  kept for the test set, as an unseen marker.
- Select 1,047 of 3,489 base items (30% globally, rounded): all 482
  ambiguous items, then 565 sampled proportionally to the remaining pool
  sizes by repair type. Use largest remainders and seed 20261024.
- Four markers are balanced within type and globally (counts differ by at
  most one); assignment uses seed 20261025. Test keeps paired forms.
- Items whose bare form has a fluent reading (`ambiguous`: number ranges,
  sentence-initial names, weekday lists) all receive an additional marked
  variant. Their unmarked versions remain in training. Test reports them
  as a separate stratum.

The script preserves the frozen base, writes 1,047 marked variants, and
appends them to all 3,489 base rows in `train_rsr.augmented.tsv` (4,536 rows).
This additive recipe was chosen on 2026-10-24: keep fluent data, add base
RSR, then add marked RSR with unchanged targets. The 30% denominator is
base items; marked rows are 23.1% of augmented RSR. Selected items have two
surface variants and greater training weight. The 482 ambiguous base inputs
are retained, so textual ambiguity remains a limitation. C2 and C3 use the
same inputs and sampling. `training_dataset/build.py` combines these rows
with the 9,486 eligible fluent PMB rows in one typed Parquet table.
