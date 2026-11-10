# Repair data

Training data for parsing replacement self-repair (RSR) into SBN: PMB
sentences with a withdrawn first version added, each with a repair-aware
SBN graph.

```
Hebrew was, is my mother tongue.
hebrew.n.01 CORRECTION <1 be.v.02 Theme -1 Time +1 time.n.08 TPR now
CONJUNCTION <2 be.v.02 Theme -3 Time +1 Co-Theme +3 time.n.08 EQU now
person.n.01 EQU speaker mother_tongue.n.01 Creator -1
```

(One line in the data; wrapped here. The withdrawn past tense sits in the
`CORRECTION` box; removing it gives back PMB's gold graph.)

## The data

| file | items | status |
|---|---:|---|
| [`rsr_generation/dataset/train_rsr.base.tsv`](rsr_generation/dataset/README.md) | 3,489 (7 repair types, ~500 each) | frozen 2026-10-13, hard negatives removed 2026-10-16; base without editing expressions |
| [`hard_negatives/train_holdout.tsv`](hard_negatives/README.md) | 66 train documents | held out of all training data |
| `rsr_generation/dataset/train_rsr.marked30.tsv` | 1,047 | scripted marked variants, including all 482 ambiguous items |
| `rsr_generation/dataset/train_rsr.augmented.tsv` | 4,536 | all base rows plus 1,047 marked variants; current RSR training component |
| [`training_dataset/data/train.parquet`](training_dataset/README.md) | 14,222 | 9,486 PMB + 3,489 base RSR + 1,047 marked RSR + 200 fluent split-reference |
| [`training_dataset/data/validation.parquet`](training_dataset/README.md) | 1,746 | 1,194 PMB + 390 base RSR + 122 marked RSR + 40 fluent split-reference; local integrated version |
| [`splits/dev/`](splits/dev/README.md) | 1,706 | unchanged PMB/RSR dev source component; the earlier published validation baseline |
| [`training_dataset/data/test.parquet`](splits/test/README.md) | 1,118 | 559 accepted PMB test repairs and their marked counterparts |

The accepted fluent split-reference supplement comes from
`fluent_group_synthesis_v1/`. Its 240 formal documents are appended through
`training_dataset/`; the earlier 6-case pilot and 60-case review rounds are not
appended. See the [unified schema](training_dataset/SCHEMA.md) for provenance,
variant labels and condition targets. Publication status is recorded in
`training_dataset/publication.json` after Hub readback; the previous snapshot is
preserved under `training_dataset/releases/`. Local build and publication commands are in
[`training_dataset/MAINTENANCE.md`](training_dataset/MAINTENANCE.md).

How it was made, in one paragraph: an LLM (Muse-Glimmer-30B) received a gold
PMB sentence, its SBN and a repair type, and wrote the repaired sentence and
SBN. Automatic checks rejected every item whose graph is malformed or does
not reduce to the gold graph when the repair is withdrawn (about 94% passed
in the main run). Blind audits of passing items found 99–100% correct
meaning and 97–100% natural repairs; edge-label repairs fell short on
naturalness and were selected by hand. The text outside the repair was
aligned with PMB, and 500 items per type were sampled.

## Where to read next

| question | file |
|---|---|
| What is in the data, how good is it, what are its limits? | [`rsr_generation/dataset/README.md`](rsr_generation/dataset/README.md) |
| Why is it designed this way? | [`DESIGN.md`](DESIGN.md) |
| Which choices were made, and why? | [`DECISIONS.md`](DECISIONS.md) |
| How do I run or rebuild it? | [`rsr_generation/README.md`](rsr_generation/README.md) |

## Layout

| path | contents |
|---|---|
| `rsr_generation/` | prompts, samples, runs, checks, audits, the dataset |
| `training_dataset/` | unified train and local dev Parquet, manifests and reproducible builders |
| `shared_pipeline/` | split-aware configuration and shared pipeline components |
| `splits/` | split-specific configuration, run records and derived data; frozen train remains in its legacy paths |
| `sbn_lin.py` | SBN reader and PMB split paths |
| `validate_with_pmb_parser.py` | structural checks used on every item |
| `repair_transform.py`, `test_against_doc.py` | rule-based repair of one site; used as a reference for content-word items |
| `reference/` | papers (not in git) |

The published Parquets and release records are kept with the builders.
Re-running the annotation and release workflows requires the local research
archive for review queues, human labels, intermediate exports and staging
packages. Raw model replies stay in their ignored local directories.

The rule-based generator that came first, and its findings, are in
`archive/repair_data_gen/`.
