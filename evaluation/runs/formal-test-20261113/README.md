# Formal test: 20261113

This batch contains 27 runs: sizes `2B`, `9B` and `27B`, conditions `C1`,
`C2` and `C3`, and seeds `24`, `42` and `4224`. The date identifies the
frozen test batch, not the download date.

```text
formal-test-20261113/
├── README.md
├── frozen-habrok.json                 # Copy from Hábrók on import
├── frozen-lst.json                    # Copy from LST on import
├── 2B/
│   ├── C1/
│   │   ├── seed24/predictions.tsv
│   │   ├── seed42/predictions.tsv
│   │   └── seed4224/predictions.tsv
│   ├── C2/                           # Same seed layout
│   └── C3/
├── 9B/                               # Same condition and seed layout
├── 27B/
└── summary.tsv                       # Create after scoring
```

Only the directories are prepared initially. Import original prediction
TSVs into the corresponding seed directories and retain both frozen
manifests unchanged. Hábrók supplies 2B and 9B; LST supplies 27B.
Add the original `run_id` mapping, test data version and inference settings
here when importing the artifacts.

Each TSV retains `example_id`, `condition`, `pred_sbn`, `tokens` and
`ended_with_eos`. C1 has 1,260 inputs (fluent PMB and hard negatives);
C2 and C3 each have 2,586 inputs (also core RSR and later reference).
On import, check the run inventory, columns, row counts, condition labels,
and unique IDs against the expected test inputs.

Create `scores/` within each seed directory when scoring starts. Use
`<slice>.basic.jsonl` and `<slice>.checks.json` for per-slice outputs.
Keep missing, invalid and non-EOS predictions in the evaluation.

## Imported Hábrók runs

| Size | Condition | Seed | Original run ID |
|---|---|---|---|
| 2B | C1 | 24 | `formal-habrok-2B-C1-s24-34265894` |
| 2B | C1 | 42 | `formal-habrok-2B-C1-s42-34265893` |
| 2B | C1 | 4224 | `formal-habrok-2B-C1-s4224-34265895` |
| 2B | C2 | 24 | `formal-habrok-2B-C2-s24-34265897` |
| 2B | C2 | 42 | `formal-habrok-2B-C2-s42-34265896` |
| 2B | C2 | 4224 | `formal-habrok-2B-C2-s4224-34265898` |
| 2B | C3 | 24 | `formal-habrok-2B-C3-s24-34265901` |
| 2B | C3 | 42 | `formal-habrok-2B-C3-s42-34265899` |
| 2B | C3 | 4224 | `formal-habrok-2B-C3-s4224-34265902` |
| 9B | C1 | 24 | `formal-habrok-9B-C1-s24-34265904` |
| 9B | C1 | 42 | `formal-habrok-9B-C1-s42-34266267` |
| 9B | C1 | 4224 | `formal-habrok-9B-C1-s4224-34265905` |
| 9B | C2 | 24 | `formal-habrok-9B-C2-s24-34265907` |
| 9B | C2 | 42 | `formal-habrok-9B-C2-s42-34265906` |
| 9B | C2 | 4224 | `formal-habrok-9B-C2-s4224-34265908` |
| 9B | C3 | 24 | `formal-habrok-9B-C3-s24-34265910` |
| 9B | C3 | 42 | `formal-habrok-9B-C3-s42-34265909` |
| 9B | C3 | 4224 | `formal-habrok-9B-C3-s4224-34265911` |

## Imported LST runs

| Size | Condition | Seed | Original run ID |
|---|---|---|---|
| 27B | C1 | 24 | `formal-lst-27B-C1-s24-127157-0` |
| 27B | C1 | 42 | `formal-lst-27B-C1-s42-127156-0` |
| 27B | C1 | 4224 | `formal-lst-27B-C1-s4224-127158-0` |
| 27B | C2 | 24 | `formal-lst-27B-C2-s24-127160-0` |
| 27B | C2 | 42 | `formal-lst-27B-C2-s42-127159-0` |
| 27B | C2 | 4224 | `formal-lst-27B-C2-s4224-127161-0` |
| 27B | C3 | 24 | `formal-lst-27B-C3-s24-127163-0` |
| 27B | C3 | 42 | `formal-lst-27B-C3-s42-127162-0` |
| 27B | C3 | 4224 | `formal-lst-27B-C3-s4224-127164-0` |

`test-hashes.json` retains the original three test-table hashes. The 27B
C1/seed42 prediction `pmb_test::p99/d0760` reached 512 tokens without EOS;
it remains unchanged in the predictions and scored panel.
