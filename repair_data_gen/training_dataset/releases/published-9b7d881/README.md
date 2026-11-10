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
---

# RepairSBN

This dataset contains English PMB 5.1.0 train-gold and standard-dev data
used by the project and synthetic replacement self-repair (RSR) variants.
The 66 documents reserved for the hard-negative evaluation are absent from
every component.

| `variant` | train | validation | target in C2 | target in C3 |
|---|---:|---:|---|---|
| `pmb_fluent` | 9,486 | 1,194 | `sbn_clean` | `sbn_repair` (= clean) |
| `rsr_base` | 3,489 | 390 | `sbn_clean` | `sbn_repair` |
| `rsr_editing_expression` | 1,047 | 122 | `sbn_clean` | `sbn_repair` |
| **total** | **14,022** | **1,706** | | |

`validation` is the project's dev split, following PMB's source partition.
Test data will be added separately when built and accepted.

`editing_expression` contains `I mean`, `no`, `no wait` or `actually` for
the marked variants and is null elsewhere. `interregnum` is not a dataset
field. `example_id` identifies a surface variant; `item_id` links the base
and marked versions of one RSR item.

The marked variants are additive. All base RSR rows remain. In train, 1,047
marked versions are appended with unchanged SBN targets. These comprise
all 482 rows flagged as textually ambiguous without a marker plus 565 rows
sampled proportionally from the remaining repair-type pools.

Train Parquet and `manifest.json` are generated locally by `build.py`.
`validation.manifest.json` records the development assembly stage;
`validation.acceptance.json` records the subsequent passed acceptance for
the same Parquet checksum. No TSV sources or human labels are uploaded.

## Local development build

Run `python -m repair_data_gen.training_dataset.build_dev` from the repository
root in the `repairsbn` environment. This writes `data/validation.parquet`
and `validation.manifest.json`, preserving the existing train build.

The 1,706 development rows comprise 1,194 fluent PMB standard-dev documents,
390 base RSR rows and 122 additional marked rows. The Parquet uses the same
20-column schema and null conventions as train. Marked dev rows cover the
122 identified multiple-reading items, with no random top-up.

C1 selects fluent rows and `sbn_clean`; C2 selects all rows and `sbn_clean`;
C3 selects all rows and `sbn_repair`. Row order is PMB source order followed
by augmented RSR source order. No shuffle or condition-specific copies are
introduced during assembly.

The development build has passed local acceptance. Use the Hub commit
revision to pin the exact published snapshot for an experiment.

## Development acceptance

Run in `repairsbn` from the repository root:

```bash
python -m repair_data_gen.training_dataset.validate_dev --tokenizer-root PATH
```

`PATH` contains tokenizer-only downloads in `2B/`, `9B/` and `27B/`, using the
model revisions in `model/smoke_test/revisions.json`. Use `hf download` with
`--revision` and repeated `--include` options for `tokenizer*`, `chat_template*`,
`vocab.json` and `merges.txt`; preserve its `.cache/huggingface/` metadata.
The validator loads these files offline and checks their recorded commits.

Results are in `../splits/dev/analysis/unified_validation/`: `summary.json`,
per-row graph checks and per-example token lengths. The build manifest records
assembly; the acceptance report records the subsequent checks.

All 1,706 rows pass source, schema, identity and graph checks. All 512 RSR rows
pass withdrawal, untouched-content and new-WordNet-concept checks. Document
and exact-input overlaps with train are zero. The 122 marked pairs recover
their bases exactly. The earlier approved surface warning is retained;
context near-duplicate checks remain deferred under the existing policy.

For all three pinned tokenizers, maximum complete sequence length is 219;
no example exceeds the design's 1,024-token training limit or 512-token
generation budget. The audit uses the design's shared neutral instruction,
the pilot's instruction/newline/text formatting, Qwen's chat template and
`enable_thinking=False`. Assistant boundaries and supervised EOS are checked.
The formal trainer must use this audited format; the smoke-test instruction
is condition-specific and is not the formal instruction.

## Limitations

The RSR inputs are synthetic and based on short written PMB text. Train
ambiguity labels are heuristic; dev labels also record confirmed alternative
readings. Most train repair types were audited by sampling; train edge-label
repairs and all selected dev base repairs were reviewed individually.
The marked and base variants of an item are surface variants of the same
semantic example, not independent observations.
