# Later-reference challenge

This test-only configuration contains 208 English inputs in 104 fluent–RSR
quasi-minimal pairs. It tests whether a parser can construct later plural
reference involving a withdrawn referent while preserving the distinction
between withdrawn and accepted content.

## Construction and review

Situations were independently authored, not formed by extending existing RSR
rows. PMB examples informed SBN conventions rather than supplying source
sentences. Fluent controls introduce two antecedents separately. RSR partners
replace the first argument with the second. Both share the exact continuation.
Event count and clause structure differ where needed; fluent controls are not
normalised RSR meanings.

Each RSR has one editorially chosen expression: `I mean` 31, `sorry` 28, `no`
23, `no wait` 12, `actually` 10. All 104 situations remain complete pairs.
There are no unmarked RSR rows in this configuration. No marker-effect or
held-out-marker comparison follows from these assignments.

The project author read the marked English inputs. Four flagged cases were
adjudicated: OBJ003 retains its material reference with a broader metadata
classification; OBJ026 retains touch Agent/Theme, supported by PMB instances;
OBJ030 retains mend Agent/Patient, combining direct Patient evidence with an
active-agent inference. OBJ016 now uses "They are my favourite vegetables."
with a revised shared continuation graph. Its kind-reference interpretation is
implicit; no dedicated kind-shift operator is encoded. Decisions are recorded
in `adjudications.json`.

All exported graphs are rebuilt from the approved specifications and checked
for parsing, DAG structure, indices, role edges, group membership, repair
status, restricted accessibility, explicit quantity and WordNet concept
existence. Shared continuations and source preservation are checked separately.
These checks do not prove that every linguistic judgement is uniquely correct.
`validation.json` records release checks and input/output hashes.

## Fields

Each row has one input and one full SBN target. Null denotes an inapplicable
editing expression or quantity; an empty withdrawn-member list is valid for a
fluent control.

| Field | Type | Meaning |
| --- | --- | --- |
| `example_id` | string | Namespaced unique row ID, ending in `_fluent` or `_rsr` |
| `pair_id` | string | Namespaced situation ID shared by the two partners |
| `family_id` | string | Namespaced source-family ID |
| `partition_unit` | string | Pair grouping ID; shared by any future derivatives |
| `text` | string | Complete English input |
| `sbn_repair` | string | Full gold SBN, ordinary for fluent and repair-aware for RSR |
| `language_condition` | string | `fluent` or `rsr` |
| `variant` | string | `fluent_split_reference` or `rsr_editing_expression` |
| `editing_expression` | nullable string | Actual expression in RSR; null for fluent |
| `data_origin` | string | `synthetic_fluent` or `synthetic_rsr` |
| `construction_type` | string | `separate_subjects` or `separate_objects`, describing the antecedent introduction frame |
| `referent_type` | string | Source interpretation category; OBJ003 uses `materials`, OBJ016 `particular_objects_with_implicit_kind_reference` |
| `source` | string | Independent synthetic later-reference scenario provenance |
| `label_source` | string | Synthetic annotation with author surface review; item adjudication where applicable |
| `reference_group_index` | int64 | Gold node constructing the later plural group |
| `antecedent_indices` | list[int64] | Gold group members in mention order: first candidate, second candidate |
| `withdrawn_antecedent_indices` | list[int64] | First candidate for RSR; empty for fluent |
| `quantity` | nullable int64 | 2 for explicit both/two; null for plain they |

Indices count concept nodes from zero, excluding separators. Each row uses its
own graph positions: fluent and RSR indices need not match. Prediction nodes
must be aligned before scoring; applying gold positions to predicted graphs
is not valid. Metadata must not be included in the parser input.

There is no `sbn_clean` field. The fluent partner asserts two ordinary events;
the RSR partner withdraws the first argument. Removing its withdrawn referent
can destroy the antecedent needed by later reference.

## Evaluation and limits

Report reference-member recovery, repair status, graph validity and complete
graph accuracy separately. The full target directly evaluates repair-aware
parsing; fluent-only and normalising parsers can be compared on reference
recovery without interpreting their missing repair structure as failure to
follow their own training objective.

Keep both partners and any derivatives together. There are 104 paired
situations, not 208 independent observations; repeated construction templates
also limit the breadth of inference. The test has two antecedents, mainly
adjective/copular continuations and no distractors, nested groups or identity
chains. It is a controlled composition diagnostic, not general dialogue
reference resolution. Complete normalised inputs are checked against the
current train/validation Parquets; this does not establish semantic isolation.

This configuration is for final evaluation, not training or model selection.
