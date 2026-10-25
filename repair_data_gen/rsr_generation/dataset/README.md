# Train RSR, base version

`train_rsr.base.tsv`: 3,489 English sentences with a replacement self-repair
(RSR), each paired with its repair-aware SBN, built from PMB 5.1.0 train
gold. "Base" means no editing expression ("The cat, the dog barks"); derived version with editing expressions ("The cat, I mean, the dog barks")
is built by `insert_editing_expressions.py` (below).

Frozen 2026-10-13; hard-negative documents removed 2026-10-16 (below).
sha256 `565ddd28c70d87d4eefa1412683ca3bbb4b62a7ddcfbfd15ae219fd2b3199ca2`
(also in `train_rsr.base.manifest.json`).

## Contents

| repair type | items | source documents | candidates | example |
|---|---:|---:|---:|---|
| content word | 500 | 500 | 3,157 | A baby is sleeping in the basket, the cradle. |
| name | 500 | 500 | 773 | Vladivostok is a city in Ukraine, Russia. |
| number | 500 | 500 | 611 | The bank lent her 300, 500 dollars. |
| temporal | 500 | 287 | 558 | I was born in 1971, in 1972. |
| tense | 500 | 500 | 777 | Hebrew was, is my mother tongue. |
| forwarding | 500 | 500 | 512 | John hates, I hate spiders. |
| edge label | 489 | 489 | 991 | He jumped from, into the train. / Tom was born near, in Boston. |
| **all** | **3,489** | **3,165** | 7,379 | |

- One repair per sentence. Median sentence length 7 words.
- The types are balanced on purpose (about 500 each), so that differences
  between types are not differences in training data size. Within a type,
  proportions follow the candidate pool (simple random sample, seed
  20261013); the manifest lists pool and sample shares side by side.
- Temporal: only 297 train documents contain a time constant, so 213 of
  them carry two items (a second site, or the same site with another
  reparandum).
- Edge label covers two kinds: a role swap (`Source` against `Destination`,
  "from, into") and a place (`location.n.01` with a spatial operator against
  a plain role, "near, in"). It is the only type selected by hand (below),
  and it falls 11 short of 500.
- No item comes from a document in the hard-negative hold-out
  (`../../hard_negatives/`, 66 documents). The set was first frozen
  without this filter; on 2026-10-16 its 27 items on held-out documents
  were dropped. Six types were refilled from their remaining candidates
  (second random sample, seed 20261016); edge label could not be (no other
  candidate is labelled natural) and went from 495 to 489. The candidates
  column counts the pool before the filter.
- PMB train has 9,552 documents; 9,486 remain after the hard-negative
  hold-out, so RSR to fluent is about 1 : 2.7.

## How it was made

1. **Generate.** An LLM (Muse-Glimmer-30B, open weights, vLLM on the LST
   cluster) gets one gold sentence, its gold SBN and a repair type, and
   writes the repaired sentence and SBN in one attempt. Its only tool is a
   WordNet lookup. The prompt is the SBN manual, the repair notation and the
   task (`../prompts/`). Details: `../README.md`.
2. **Check.** Every item must pass all automatic checks (list below).
   Failures are dropped, never repaired.
3. **Select.** Edge label only: a human (or Claude, see limitations) judged
   each passing item for naturalness; only "yes" items are kept.
4. **Fix the surface.** `apply_fixes.py` makes the text outside the repair
   identical to the PMB sentence, and lower-cases the first word of the
   repair where the model kept a sentence-initial capital (175 items changed).
5. **Sample.** `assemble.py` draws 500 per type, drops and refills items on
   held-out documents, and re-runs every check on the result; it stops if
   any row fails or a held-out document is present.

### Automatic checks

| check | what it rejects |
|---|---|
| `parse`, `dag`, `index`, `magnitude`, `edges`, `boxes` | SBN the reader rejects; cycles; an index that lands on no concept or is ≥ 10; edges lost in reading; `CORRECTION` and `CONJUNCTION` not siblings |
| `wn_ok` | a synset not in WordNet |
| `wn_relation` | reparandum and repair in the same synset (a synonym, not a repair) |
| `untouched` | a gold edge away from the repair site changed |
| `withdraw` | removing the `CORRECTION` box does not give back exactly the gold graph |
| `repair_part` | the sentence does not match the SBN: onset, rest of sentence, identical or overlapping versions |
| `site_ok`, `place_inside`, `pin` | a site outside the type's definition; one place inside the other ("Osaka, Japan"); a site other than the one given |
| gold alignment | text outside the repair differs from PMB (added by `assemble.py`) |

### Quality evidence

Pass rates of the automatic checks, on answered items:

| run | pass |
|---|---:|
| six types, main run (`train_full`) | 5,643 / 5,984 (94.3%; 96.1% without the filtered synonyms) |
| forwarding, extra documents | 179 / 184 |
| temporal, second item per document | 289 / 296 |
| edge label, role swaps (`train_edge`) | 366 / 377 |
| edge label, places (round 4 / round 5) | 121 / 348; 427 / 440 |

Blind human audits of passing items (one annotator; `unsure` left out of the
rates). Thresholds were set before the first audit and not moved: meaning ≥
90%, natural ≥ 80%.

| audit | items | meaning | natural |
|---|---:|---:|---:|
| round 2, 15 per type (`../audit/r2`) | 105 | 99% | 97% |
| main run, 15 per type (`../audit/train_full`) | 90 | 100% | 97% |
| temporal extra (`../audit/train_temporal_extra`) | 21 | 100% | 100% |
| forwarding, frozen set (`../audit/train_forwarding_reaudit`) | 15 | 100% | 100% |
| edge label, role swaps, round 3 (`../audit/r3_edge`) | 34 | – | 48% |

In the main-run audit, 11 of 15 forwarding items were `unsure` for meaning,
over which subject each version's predicate points to; the meaning rate
there rests on 4 items. Once that rule was settled (each version's predicate
points to its own subject), 15 new forwarding items from the frozen set
were audited (not blind to type; `../sample_forwarding_reaudit.py`).

Edge label failed the naturalness threshold, so it was selected item by item
instead:

| source | judged | kept |
|---|---:|---:|
| role swaps, round 3 (user) | 34 | 14 |
| role swaps, `train_edge` (user) | 363 | 112 |
| places (`../audit/train_place/selection.tsv`) | 591 | 369 |

Of the 495 kept, 6 places were later dropped for the hard-negative
hold-out, leaving 489 (126 role swaps, 363 places). Of these, the
naturalness label came from the user for 225 (26 of them after a second
look at items first marked unsure) and from Claude for 264 (3 of them
rescued unsure items, accepted by the user). Claude labelled blind, with a
written guideline (`../audit/train_place/claude_guideline.md`); on the 100
items both labelled independently, κ = 0.56 and yes/no agreement 69 of 76.
The user then read all 270 Claude-labelled items kept before the drop and
accepted them without change (by hand, not recorded in a file), so every
kept item is approved by the user.
`label_source` in the data says who gave the first label.

## Files

| file | what |
|---|---|
| `train_rsr.base.tsv` | the data |
| `train_rsr.base.manifest.json` | sha256, seed, counts, source documents, pool and sample proportions per type |
| `apply_fixes.py` | surface fixes on all passing items → `pool_fixed.tsv` (not in git, rebuilt by the script) |
| `assemble.py` | edge-label selection, sampling, final checks → the data and manifest |
| `onset_case_decisions.tsv` | reviewed case decisions for repair onsets (181 rows) |
| `fix_log.tsv` | every changed item: gold, model text, fixed text |

Rebuild from `rsr_generation/`:

    python3 dataset/apply_fixes.py
    python3 dataset/assemble.py      # must print the sha256 above

## Columns

| column | meaning |
|---|---|
| `item_id` | `rsr_train_0000` … `rsr_train_3488` |
| `repair_type` | `content_word`, `constant_name`, `constant_number`, `constant_temporal`, `tense`, `forwarding`, `edge_label` |
| `doc_id` | PMB document (`p63/d1502`) |
| `nl_clean`, `sbn_clean` | gold sentence and SBN |
| `nl_repair`, `sbn_repair` | repaired sentence and SBN |
| `repair_onset` | character offset in `nl_repair` where the repair starts; the reparandum ends 2 characters before (", ") |
| `reparandum_surface` | the model's report of the replaced words; usually the core word only ("issue" in "the issue, the problem"). For the full reparandum span, compare `nl_repair` with `nl_clean` |
| `reparandum_synset`, `reparandum_value`, `repair_synset` | the replaced concept; the new constant, tense operator, role or spatial operator where the synset stays |
| `site_index` | position of the repaired concept in `sbn_clean` |
| `devices`, `strategy` | notation devices ①–⑤, as the model reports them (not verified) |
| `wn_relation` | WordNet relation of reparandum and repair (content word) |
| `ambiguous` | the bare form has a fluent reading: `number_range` (260), `initial_name` (203), `weekday_list` (19). A label, not a filter |
| `interregnum` | `none` in this version |
| `label_source`, `audit_ref` | `auto_pass` for the six automatic types; for edge label who judged it, and the audit item |
| `source_file` | the run the item comes from (`../results/`) |
| `nl_repair_model`, `repair_onset_model`, `fixes`, `aligned` | the model's own text, what `apply_fixes.py` changed, whether the sentence could be rebuilt from gold |

## Limitations

- **One annotator.** Every audit and the edge-label selection rest on one
  person's intuition; most edge-label labels were proposed by Claude and
  reviewed, not labelled independently, by the user (above). Labels
  drifted during the first edge-label session (yes rate 31%, 20%, 32%, 40% by
  quarter).
- **Audits are samples.** Apart from edge label, no type was read in full;
  the rates above apply to items that passed the checks.
- **Errors the checks cannot see.** An index that lands on a wrong concept
  of the right kind can pass the structural checks. The round-2 audit
  labelled one such item correct; `withdraw` was added and catches it, but
  the meaning rates remain upper bounds for this kind of error.
- **Written source.** PMB sentences are short written text; the repairs are
  constructed, not observed. Naturalness here means acceptable to a fluent
  speaker, not typical of speech. A comparison with Switchboard repairs is a
  separate, later step.
- **Thresholds are this project's own**, set before the runs; they decide
  whether a type is used, not how good it is.
- **Fluent readings.** Without an editing expression some items read as a
  range, a list or a vocative ("two, three pounds"); they are labelled in
  `ambiguous`, not removed.
- **PMB limits.** A repair that needs an index of 10 or more is rejected
  (the reader truncates it), so long sentences are under-represented.

## Editing-expression derivatives

Run after rebuilding the base:

    python3 dataset/insert_editing_expressions.py

The script checks the frozen base hash and `holdout.load()`. It selects
1,047 base items (30% globally, rounded to the nearest integer): all 482
ambiguous items, then 565 from the remaining 3,007. Remaining quotas are
proportional to each repair type's non-ambiguous pool, rounded by largest
remainders (lexical type order breaks ties). Sampling uses seed 20261024.
Marker assignment shuffles selected IDs within type with seed 20261025 and
cycles through `I mean`, `no`, `no wait`, `actually` across types. Counts
per marker differ by at most one globally and within each type. `sorry`
is never inserted as an editing expression.

| file | contents |
|---|---|
| `train_rsr.marked30.tsv` | 1,047 marked variants only |
| `train_rsr.augmented.tsv` | 4,536 rows: all 3,489 base rows followed by 1,047 marked variants |
| `train_rsr.editing.manifest.json` | seeds, rules, per-type counts, source/script/output hashes |

Both targets remain byte-for-byte unchanged. `repair_onset` moves past the
inserted marker and comma-space. `editing_expression` contains the inserted
marker and is empty on base rows. New fields also include `variant_id`
(unique surface variant), `nl_repair_base`, `repair_onset_base`, `editing_start`,
`editing_end` (half-open character offsets for the marker only), and
`selection_reason`. `item_id` retains the base identity for pairing.
Model-output and audit fields retain base provenance; they are not an audit
of the inserted marker. Removing the inserted segment must recover the exact
base text. Insertion does not establish that every marker resolves every
ambiguity or sounds equally natural; marker-specific human review is pending.

The training recipe is additive: retain every base row, including all 482
unmarked ambiguous items, and append the selected marked variants. The 30%
refers to selected base items; marked rows are 23.1% of the resulting 4,536
RSR rows. This preserves the full base distribution while adding examples
whose target is invariant to the inserted editing expression. Selected items
have two surface variants and thus twice the row weight of unselected items;
the augmented type counts are not balanced. These are surface variants, not
new independent source documents. Use `variant_id` as the unique row key and
`item_id` to link versions. C2 and C3 must use the same rows and sampling.

The superseded replacement mixture is archived under
`archive/repair_data_gen/dataset_recipes/`; it is not a training input.
The augmented file does not contain the fluent PMB component. Adding the
9,486 eligible fluent rows later gives 14,022 rows per C2/C3 condition and
RSR : fluent = 4,536 : 9,486 (about 1 : 2.1).

The frozen base retains its historical `interregnum = none` column because
renaming it would change the recorded base hash. All derived files and the
unified training Parquet use `editing_expression`; the legacy column is not
published.
