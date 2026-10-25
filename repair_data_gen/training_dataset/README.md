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
---

# RepairSBN

This private working dataset contains the English PMB 5.1.0 train-gold data
used by the project and synthetic replacement self-repair (RSR) variants.
The 66 documents reserved for the hard-negative evaluation are absent from
every component.

| `variant` | rows | target in C2 | target in C3 |
|---|---:|---|---|
| `pmb_fluent` | 9,486 | `sbn_clean` | `sbn_repair` (= clean) |
| `rsr_base` | 3,489 | `sbn_clean` | `sbn_repair` |
| `rsr_editing_expression` | 1,047 | `sbn_clean` | `sbn_repair` |
| **total** | **14,022** | | |

`editing_expression` contains `I mean`, `no`, `no wait` or `actually` for
the marked variants and is null elsewhere. `interregnum` is not a dataset
field. `example_id` identifies a surface variant; `item_id` links the base
and marked versions of one RSR item.

The marked variants are additive. All base RSR rows remain, and 1,047 marked
versions are appended with unchanged SBN targets. These 1,047 items comprise
all 482 rows flagged as textually ambiguous without a marker plus 565 rows
sampled proportionally from the remaining repair-type pools.

The Parquet and `manifest.json` are generated locally by `build.py`. No TSV
source file is part of the upload directory.

## Limitations

The RSR inputs are synthetic and based on short written PMB text. The
ambiguity labels are heuristic. Most repair types were audited by sampling;
edge-label repairs were selected item by item. The marked and base variants
of an item are surface variants of the same semantic example, not independent
observations.
