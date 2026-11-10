---
language:
- en
pretty_name: RepairSBN
task_categories:
- text-generation
configs:
- config_name: default
  default: true
  data_files:
  - split: train
    path: data/train.parquet
  - split: validation
    path: data/validation.parquet
  - split: test
    path: data/test.parquet
- config_name: later_reference
  data_files:
  - split: test
    path: later_reference/test.parquet
---

# RepairSBN

The `default` configuration contains **17,086 English text–meaning pairs** for semantic parsing
into Simplified Box Notation (SBN), a sequence representation of discourse meaning.
It combines original Parallel Meaning Bank (PMB) 5.1.0 examples, synthetic
replacement self-repairs, and synthetic fluent examples of plural reference.
The separate `later_reference` configuration adds **208 test rows in 104
fluent–RSR quasi-minimal pairs**, giving 17,294 rows across both configurations.

A replacement self-repair replaces an earlier word or phrase within an utterance.
A fluent group-reference example introduces two referents separately, then refers
to them together. The referents may occur in separate clauses or as the subject
and object of one clause.

## Dataset composition

| Data type | `variant` | Train: rows (%) | Validation: rows (%) | Test: rows (%) |
| --- | --- | ---: | ---: | ---: |
| PMB fluent text | `pmb_fluent` | 9,486 (66.7%) | 1,194 (68.4%) | 0 |
| Self-repair without an editing expression | `rsr_base` | 3,489 (24.5%) | 390 (22.3%) | 559 (50.0%) |
| Self-repair with an editing expression | `rsr_editing_expression` | 1,047 (7.4%) | 122 (7.0%) | 559 (50.0%) |
| Fluent group reference | `fluent_split_reference` | 200 (1.4%) | 40 (2.3%) | 0 |
| **Total** | | **14,222 (100%)** | **1,746 (100%)** | **1,118 (100%)** |

Percentages are calculated within each split. `validation` is the development
split. PMB examples follow the English train-gold, standard-development and
standard-test partitions. Related synthetic examples remain in the same
split as their source family.

Self-repairs cover lexical concepts, names, numbers, time values, tense, semantic
relations and larger phrases. Editing expressions include *I mean*, *no*,
*no wait* and *actually*. Marked examples are additional surface forms of selected
base repairs: row counts do not represent independent semantic situations.

## PMB self-repair test

The `default/test` split contains 559 eligible base repairs from 505 PMB
standard-test documents. Each base has one marked counterpart with the
same `item_id` and both SBN targets unchanged. The five editing expressions
are assigned with a fixed seed and balanced within repair type and overall.
*Sorry* is held out from the train and validation RSR marker assignments.
The marked forms were spot-checked. The split contains synthetic RSR
positives; ordinary fluent PMB evaluation and the later-reference challenge
are separate.

For the main comparison, report each repair type separately and give the
seven types equal weight in the overall score. Pair variants by `item_id`
and group comparisons by PMB source document.

## Examples

| Data type | Text |
| --- | --- |
| PMB fluent | The sky is blue. |
| Self-repair with an editing expression | The sky is grey, no, blue. |
| Fluent group reference | A boy hugged a girl. They both laughed. |

<details>
<summary>View the SBN annotations</summary>

For **The sky is grey, no, blue.**, `sbn_clean` gives the intended fluent meaning:

```sbn
sky.n.01
time.n.08 EQU now
blue.a.01 ColourOf -2 Time -1
```

`sbn_repair` also retains the abandoned contribution and its replacement:

```sbn
sky.n.01
time.n.08 EQU now
CORRECTION <1
grey.a.01 ColourOf -2 Time -1
CONJUNCTION <2
blue.a.01 ColourOf -3 Time -2
```

For **A boy hugged a girl. They both laughed.**, both target fields contain:

```sbn
boy.n.01
time.n.08 TPR now
hug.v.01 Agent -2 Co-Agent +1 Time -1
girl.n.02
CONTINUATION <1
entity.n.01 Sub -4 Sub -1 Quantity 2
time.n.08 TPR now
laugh.v.01 Agent -2 Time -1
```

The two `Sub` links construct a group from the boy and girl. `Quantity 2` records
*both*. Plain *they/them* does not receive an explicit quantity solely because
two antecedents are available.

</details>

## Fields

Each row contains one complete text and two SBN target fields. The same
24-column schema is used for both splits; unavailable metadata is null.

| Field(s) | Meaning |
| --- | --- |
| `text` | English input text |
| `sbn_clean` | SBN of the intended fluent meaning, with self-repair removed |
| `sbn_repair` | SBN retaining self-repair structure; identical to `sbn_clean` for fluent text |
| `variant`, `data_origin` | Input form and provenance: original PMB, synthetic self-repair or synthetic fluent text |
| `example_id`, `item_id`, `doc_id` | Input ID, related-item ID and source-document ID |
| `repair_type`, `editing_expression` | Repair category and optional editing expression; null for fluent text |
| `construction_type` | Main introduction construction in the fluent group-reference examples |
| `family_id`, `partition_unit` | Identifiers for related examples kept within one split |
| `source`, `label_source` | Corpus or synthesis source and annotation provenance |

These fields describe `default`. The [full schema](SCHEMA.md) defines every field, its type and applicable values,
including the repair-specific offsets and lexical annotations. Provenance fields
are metadata, not part of the input text.

## Later-reference challenge

`later_reference` is a test-only diagnostic of reference to retained repair
material. Each of 104 independently authored situations has a fluent control
and an RSR version. Both introduce the same two candidate referents in the same
order and share the exact continuation. Fluent controls introduce antecedents
in separate assertions; RSR withdraws the first argument and accepts the second.
These are quasi-minimal pairs: event count, clause structure and repeated tense
or speaker information can differ. The fluent partner is not a clean target
for the RSR partner.

Every RSR input has one editing expression: *I mean* (31), *sorry* (28), *no*
(23), *no wait* (12), or *actually* (10). Expressions were chosen editorially
for natural replacement readings, not randomly assigned or balanced. This
configuration does not estimate marker effects or held-out-marker generalisation.

The full SBN gold retains repair structure for RSR inputs; fluent controls have
ordinary SBN. Later plural reference constructs a group with two `Sub` links.
Explicit *both/two* receives `Quantity 2`; plain *they* does not. OBJ016 adopts
implicit kind reference in "They are my favourite vegetables." without a
separate kind-shift operator. English surfaces were read by the project author;
four flagged cases were subsequently adjudicated, and all 208 targets passed
the implemented structural, lexical and pair-consistency checks.

```python
from datasets import load_dataset

challenge = load_dataset("Shrikes/repair-sbn", "later_reference", split="test")
```

The [challenge card and field definitions](later_reference/README.md) describe
the separate 18-field schema, reference indices and scoring scope. There is no
`sbn_clean` target in this configuration: deleting the reparandum can remove an
antecedent required by the continuation. Evaluate member recovery separately
from repair status, and retain complete pairs in comparisons. C3 is trained for
the full repair-aware target; C1/C2 can serve as reference conditions, but their
lack of repair structure is not itself a failure to follow their training task.

The set covers controlled two-antecedent short discourses, mainly adjective or
copular continuations, with no distractors, nested groups or identity chains.
It does not establish general discourse-reference competence. Normalised full
inputs are checked against the current train/validation release; this is not
proof of semantic decontamination. The 208 rows represent 104 paired situations,
with recurring construction templates, rather than 208 independent observations.

## Scope and limitations

- The synthetic self-repairs are based on short written PMB texts and do not
  cover the full range of spontaneous spoken repair.
- Fluent group-reference examples focus on two antecedents in short, controlled
  discourse. Their text and annotations have been human-reviewed; this subset
  does not represent unrestricted dialogue or all plural-reference phenomena.

Source corpus: [Parallel Meaning Bank](https://pmb.let.rug.nl/), release 5.1.0.
Source-text licensing information is provided with the PMB distribution.
