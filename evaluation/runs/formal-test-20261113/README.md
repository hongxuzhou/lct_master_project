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

## Scoped evaluation

The current analysis covers all 27 imported 2B/9B/27B runs, using the audited
`../gold-audit-20261115/retained/` panel and `scoped_sbn_v2` only. Each run has
1,259 retained C1 inputs or 2,585 retained C2/C3 inputs. The original prediction
TSVs and historical `scores/` are preserved. Per-slice scores and targeted
verdicts live in `scores-scoped-v2/`. Selected prediction copies, command logs
and per-slice verification files are rebuildable local intermediates.

From the repository root, in the `repairsbn` environment:

```bash
python evaluation/score_batch.py --workers 6
python evaluation/analyse_batch.py --output /tmp/repairsbn-analysis-review
QUARTO_PYTHON="$CONDA_PREFIX/bin/python" quarto render evaluation/formal_results.qmd
```

`score_batch.py` resumes verified existing results without replacing them.
`analyse_batch.py` requires a new output directory (default
`analysis-scoped-v2-all-sizes/`); its `items.parquet` preserves every retained run/input
observation. `summary.tsv` contains per-seed metrics and denominators. `paired_units.parquet` and
`paired_effects.tsv` retain the source-document or authored-pair observations
and within-seed bootstrap intervals. `comparisons.tsv` summarises the three
seed differences. `manifest.json` records source and derived-file hashes.

See [the report](../../formal_results.qmd) for targets, comparison boundaries,
statistical units and interpretation. Earlier two-size and standalone 27B
snapshots are kept locally and are not required by the report.
The complete snapshot contains 27 runs, 90 run/slice panels and 57,861 retained
observations. Analysis snapshots require a new output directory and are never
overwritten. The optional `--update-index` creates a local batch-root summary copy.

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
