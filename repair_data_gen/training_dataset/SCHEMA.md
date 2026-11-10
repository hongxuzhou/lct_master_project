# RepairSBN fields

All three `default` splits use the same 24-column schema. Null means a field is unavailable or
not applicable. Each row contains one English input and its SBN representations.

## Text and meaning representations

| Field | Type | Meaning |
| --- | --- | --- |
| `text` | string | Complete English input |
| `sbn_clean` | string | SBN of the intended fluent meaning, with self-repair removed |
| `sbn_repair` | string | SBN retaining abandoned and accepted repair material; equal to `sbn_clean` for fluent inputs |

## Type, provenance and identifiers

| Field | Type | Meaning |
| --- | --- | --- |
| `example_id` | string | Unique ID for an input text and its targets |
| `item_id` | nullable string | Links base and marked forms of one repair; identifies an individual synthetic fluent item; null for PMB originals |
| `doc_id` | string | PMB source-document ID, or a namespaced ID for an independently authored synthetic document |
| `variant` | string | Input form, as defined below |
| `data_origin` | string | `pmb`, `synthetic_rsr` or `synthetic_fluent` |
| `construction_type` | nullable string | Main introduction construction for fluent group-reference inputs, as defined below |
| `family_id` | string | Source-family ID linking related examples |
| `partition_unit` | string | Grouping ID used to keep closely related examples within one split |
| `label_source` | nullable string | Origin of the annotation or review judgement |
| `source` | string | Readable corpus partition or synthesis source |

### Input forms

| `variant` | `data_origin` | Meaning |
| --- | --- | --- |
| `pmb_fluent` | `pmb` | Original PMB 5.1.0 fluent text and SBN |
| `rsr_base` | `synthetic_rsr` | Synthetic replacement self-repair without an editing expression |
| `rsr_editing_expression` | `synthetic_rsr` | Additional repair surface with an editing expression such as *I mean* or *no* |
| `fluent_split_reference` | `synthetic_fluent` | Fluent text in which a later expression constructs a group from separately introduced referents |

`variant` distinguishes input forms; `data_origin` distinguishes their provenance.
An editing-expression variant shares its `item_id` and SBN targets with its base
repair. Independently authored fluent documents have IDs beginning with
`fluent_group_v1::`, rather than PMB document IDs. PMB originals and their derived
repairs share a PMB source family.

### Fluent group-reference constructions

`construction_type` is null outside `fluent_split_reference`. It describes the
main construction; an example may also contain other linguistic features.

| Value | Meaning |
| --- | --- |
| `same_clause` | The antecedents occupy subject and object positions in one clause |
| `separate_objects` | The antecedents are introduced as objects in separate clauses |
| `existential` | An antecedent is introduced by an existential such as *there was* |
| `possession` | An antecedent is introduced through possession |
| `stative_contrast` | The antecedents are introduced through contrasting states |
| `distinguished_same_noun` | Repeated noun heads are distinguished by explicit modifiers |
| `separate_subjects` | The antecedents are introduced as subjects in separate clauses |

## Self-repair annotations

These fields are null for both fluent variants. Within repair examples, a field
can also be null when it is not applicable to that repair category.

| Field | Type | Meaning |
| --- | --- | --- |
| `repair_type` | nullable string | Replacement category, as defined below |
| `editing_expression` | nullable string | Editing expression in the text; null for base repairs |
| `ambiguity_type` | nullable string | Annotation of an alternative fluent interpretation |
| `repair_onset` | nullable int64 | Zero-based character offset where the accepted repair begins in `text` |
| `reparandum_surface` | nullable string | Abandoned word or phrase |
| `reparandum_synset` | nullable string | Abandoned lexical concept |
| `reparandum_value` | nullable string | Abandoned constant or value |
| `repair_synset` | nullable string | Accepted lexical concept |
| `site_index` | nullable int64 | Position of the repair-generation site in the source SBN |
| `strategy` | nullable string | Annotation method used to construct the repair representation |
| `wn_relation` | nullable string | WordNet relation recorded for the lexical replacement |

### Replacement categories

| `repair_type` | Meaning |
| --- | --- |
| `content_word` | Lexical concept replacement |
| `constant_name` | Proper-name value replacement |
| `constant_number` | Numerical value replacement |
| `constant_temporal` | Time or date value replacement |
| `tense` | Tense replacement |
| `edge_label` | Semantic-relation replacement |
| `forwarding` | Replacement spanning a larger phrase |

## Later-reference test configuration

`later_reference` has a separate 18-field schema defined in
[its challenge card](later_reference/README.md). It contains 104 complete pairs
in one `test` split. Core names `text`, `sbn_repair`, `example_id`, `variant`,
`editing_expression`, `data_origin`, `family_id` and `partition_unit` retain
their corresponding purposes. `pair_id` links the fluent/RSR partners.
`construction_type` describes antecedent introduction on both partners.

`sbn_clean` is absent: the fluent partner is not the normalisation of its RSR
partner. Reference indices are zero-based positions among SBN concept nodes,
excluding separators, and apply to each row's own gold SBN. They are evaluation
metadata, not input features or indices into model predictions.
