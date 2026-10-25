# Decisions: repair data

What was decided, and why. One line each. The round-by-round record behind
these is the playground README at git tag `rsr-train-base-log`.

## Source and scope

| date | decision | why |
|---|---|---|
| 10-01 | Test LLM generation on every single-sentence repair type, not only content words | The rule pipeline reaches content words only; the other types are where an LLM can add something |
| 10-04 | All seven types are LLM-generated; the rule pipeline is no longer a data source (archived) | Round 2 passed the thresholds for the types the rule pipeline cannot produce |
| 10-04 | Train first; dev and test later, on PMB's own split | One problem at a time; PMB's split keeps test sentences unseen in training |
| 10-04 | Tense is back in the data | It was cut for the cost of a rule operator, which LLM generation does not have |
| 10-01 | Anaphora and cross-turn repairs are left out | They need a second sentence or turn with no gold to check against |
| 10-16 | 66 surface-similar non-repair train documents are held out of all training data, RSR included (`hard_negatives/`); builders read the list through `holdout.load()` | A hard-negative test on documents seen in training measures memory; the hold-out was planned for 10-13 but never implemented |
| 10-16 | `contrast_correction` and `borderline` documents are held out too | Test use is decided later; a trained-on document can never be used |

## Generation

| date | decision | why |
|---|---|---|
| 10-01 | One attempt per item; no revise loop, no self-rating | Retries hide the error distribution; models rate themselves too generously |
| 10-01 | The only tool is a WordNet lookup; no candidate pool | A pool would encode the judgement being tested (what makes a believable slip) |
| 10-01 | Prompt inventories are PMB train gold ∩ what the SBN reader accepts; inversion is a rule (every index role has an inverse) | Data-driven, and every example must parse |
| 10-02 | Base repairs only; editing expressions are inserted later by script | Inserting one is a fixed string edit; only scripted insertion gives matched pairs |
| 10-02 | The model returns the sentence as left and right parts; the harness records `repair_onset` | Models count characters badly |
| 10-02 | Muse-Glimmer-30B (open weights) on the LST cluster | Opus 5.5 was the pilot (round 1); production runs use an open-weights model on university hardware |
| 10-04 | A prompt change makes a new prompt version; versions are never edited in place; `run_meta` records the sha256 | All types share one system prompt, so an in-place edit would change validated conditions |

## Quality control

| date | decision | why |
|---|---|---|
| 10-02 | Release thresholds: checks ≥ 95%, `no_site` ≤ 3%, meaning ≥ 90%, natural ≥ 80% | Set before round 2 and not moved; they decide use, not quality |
| 10-01 | Items that fail a check are dropped, never fixed | Keeps every item the model's own output |
| 10-01 | Same-synset reparandum fails; hypernym and hyponym are allowed | A synonym is not a repair; correcting to a more exact or more general word is natural |
| 09-27 | NLI and surprisal scores are archived, not used as a filter | Calibration showed they do not separate good from bad items here |
| 10-02 | Audits are blind, drawn from items that pass every check; `unsure` is reported apart | The audit measures what the checks cannot see |
| 10-08 | "Natural" = acceptable to a fluent speaker: grammatical, recognisable as a repair, both versions recoverable; no learner errors | Widening the speaker population would make the data unrepresentative |

## Repair types

| date | decision | why |
|---|---|---|
| 10-04 | Content word: no quota per notation device | The device follows from the site; forcing one produces awkward repairs |
| 10-04 | Names: at most 3 items per name value | Otherwise the name type is mostly "Tom" |
| 10-01 | Forwarding: pronoun subjects are valid; `NEGATION` quantifiers (nobody) and elided verbs are not | A quantifier subject is built with `NEGATION`, and an elided verb is in the SBN but not in the sentence |
| 10-08 | Forwarding: a predicate said in each version points to that version's subject | Follows the notation's own example ("Josh drove, no, Marsha drove") |
| 10-13 | Forwarding: items with a reflexive bound to the new subject are kept | Restating the whole clause is what forwarding is |
| 10-08 | Edge label is selected by hand (natural only); no verb list | 48% natural in the round-3 audit; the judgement depends on tense and context, which a list cannot capture |
| 10-05 | `be.v.03` "from"/"in" only with a person subject | "There's a cat from, on the desk" is not a repair |
| 10-10 | Place repairs ("under, on") count as edge label, recorded as `spatial_operator` | Role swaps alone gave 100–150 natural items |
| 10-12 | Place selection uses the substitution test: the first preposition with the same thing must be a sentence the speaker could mean; shared landmark only | The bare form cannot show an abandoned, different landmark |
| 10-12 | Edge label closes at 495 | Nearly every eligible train document has been run |
| 10-16 | Hold-out applied to the frozen set by drop and refill (seed 20261016): 27 items dropped, 21 refilled; edge label not refilled (489) | Equivalent to a simple random sample from the filtered pool and keeps the other 3,468 items and their audits; no other edge item is labelled natural |
| 10-15 | The 270 kept items Claude labelled were read by the user and all accepted; reported as "proposed by Claude, reviewed by the user", κ = 0.56 kept | Review is not independent labelling; κ shows how subjective the judgement is |

## The dataset

| date | decision | why |
|---|---|---|
| 10-10 | About 3,500 items on train, about 500 per type; quotas are caps | 3,500 is the middle of RSR : fluent = 1:2 to 1:4; equal types remove data size as an explanation of type differences |
| 10-10 | "Two items per document" is a reporting requirement, not a rule; documents may serve several types | It guards against overstated variety, not against an error; temporal has only 297 documents |
| 10-08 | Items with a fluent reading (`ambiguous`) are labelled, not removed | Removing "n, n+1" numbers would distort the number distribution |
| 10-13 | Text outside the repair is identical to PMB (quotes, final punctuation, apostrophes) | A difference would be a surface cue that separates RSR from fluent sentences |
| 10-13 | Simple random sample per type (seed 20261013) | Keeps each type's internal proportions; balance is between types only |
| 10-13 | `reparandum_surface` stays as the model wrote it; the full span comes from the sentence diff | The field holds the core word; analyses that need the span derive it |
| 10-13 | Comparisons with Switchboard use the frozen set, not the candidate pool | The claim is about the training data; the edge-label pool includes rejected items |
| 10-02 | `sorry` is reserved for the test set | It is the held-out editing expression for the generalisation probe |
| 10-24 | Select 30% globally: 1,047 base items; include all 482 ambiguous items, then sample 565 proportionally to remaining per-type pools (largest remainders, seed 20261024) | Per-type 30% cannot cover all ambiguous name and number items |
| 10-24 | Append 1,047 scripted marked variants to all 3,489 base rows, including ambiguous unmarked rows; archive the replacement mixture | Simple additive construction: fluent, base RSR, then marked RSR with unchanged targets. Selected items receive extra row weight; 30% is relative to base items (23.1% of augmented RSR) |
| 10-24 | Publish one unified Parquet with `editing_expression`, not the component TSV files; build the 9,486 fluent rows directly from PMB train gold through `holdout.load()` | One typed table prevents schema drift and makes the C1/C2/C3 target choice explicit; the frozen base keeps its legacy source column only to preserve its hash |
