# Train RSR, base version

`train_rsr.base.tsv`: 3,495 English sentences with a replacement self-repair
(RSR), each paired with its repair-aware SBN, built from PMB 5.1.0 train
gold. "Base" means no editing expression ("The cat, the dog barks"); a
version with editing expressions ("The cat, I mean, the dog barks") is a
later step.

Frozen 2026-10-13. sha256 `93733f7ecab2e61ba3063eed3438ea09e47455fc02556a760f727c5b934b9f58`
(also in `train_rsr.base.manifest.json`).

## Contents

| repair type | items | source documents | candidates | example |
|---|---:|---:|---:|---|
| content word | 500 | 500 | 3,157 | A baby is sleeping in the basket, the cradle. |
| name | 500 | 500 | 773 | Vladivostok is a city in Ukraine, Russia. |
| number | 500 | 500 | 611 | The bank lent her 300, 500 dollars. |
| temporal | 500 | 290 | 558 | I was born in 1971, in 1972. |
| tense | 500 | 500 | 777 | Hebrew was, is my mother tongue. |
| forwarding | 500 | 500 | 512 | John hates, I hate spiders. |
| edge label | 495 | 495 | 991 | He jumped from, into the train. / Tom was born near, in Boston. |
| **all** | **3,495** | **3,170** | 7,379 | |

- One repair per sentence. Median sentence length 7 words.
- The types are balanced on purpose (about 500 each), so that differences
  between types are not differences in training data size. Within a type,
  proportions follow the candidate pool (simple random sample, seed
  20261013); the manifest lists pool and sample shares side by side.
- Temporal: only 297 train documents contain a time constant, so 210 of
  them carry two items (a second site, or the same site with another
  reparandum).
- Edge label covers two kinds: a role swap (`Source` against `Destination`,
  "from, into") and a place (`location.n.01` with a spatial operator against
  a plain role, "near, in"). It is the only type selected by hand (below),
  and it falls 5 short of 500.
- PMB train has 9,552 documents; 9,447 remain after a planned hard-negative
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
5. **Sample.** `assemble.py` draws 500 per type and re-runs every check on
   the result; it stops if any row fails.

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
| edge label, role swaps, round 3 (`../audit/r3_edge`) | 34 | – | 48% |

Edge label failed the naturalness threshold, so it was selected item by item
instead:

| source | judged | kept |
|---|---:|---:|
| role swaps, round 3 (user) | 34 | 14 |
| role swaps, `train_edge` (user) | 363 | 112 |
| places (`../audit/train_place/selection.tsv`) | 591 | 369 |

Of the 495 kept, the naturalness label came from the user for 225 (26 of
them after a second look at items first marked unsure) and from Claude for
270 (3 of them rescued unsure items, accepted by the user). Claude
labelled blind, with a written guideline
(`../audit/train_place/claude_guideline.md`); on the 100 items both labelled,
κ = 0.56 and yes/no agreement 69 of 76. `label_source` in the data says
which.

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
| `item_id` | `rsr_train_0000` … `rsr_train_3494` |
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
| `ambiguous` | the bare form has a fluent reading: `number_range` (258), `initial_name` (201), `weekday_list` (19). A label, not a filter |
| `interregnum` | `none` in this version |
| `label_source`, `audit_ref` | `auto_pass` for the six automatic types; for edge label who judged it, and the audit item |
| `source_file` | the run the item comes from (`../results/`) |
| `nl_repair_model`, `repair_onset_model`, `fixes`, `aligned` | the model's own text, what `apply_fixes.py` changed, whether the sentence could be rebuilt from gold |

## Limitations

- **One annotator.** Every audit and the edge-label selection rest on one
  person's intuition; most edge-label labels are Claude's (above). Labels
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
