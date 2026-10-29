# Shared split pipeline

This directory holds code shared by train, development and test data
construction. Split-specific configuration and outputs live under
`../splits/`; code is not copied into each split.

The frozen train pipeline remains under `../rsr_generation/` and
`../training_dataset/`. Do not move or regenerate it as part of the dev
work. Shared components are introduced through the dev pipeline first and
may be applied to train only after a byte-identical compatibility check.

Validate a split configuration from the repository root with:

```bash
python3 repair_data_gen/shared_pipeline/split_config.py \
  repair_data_gen/splits/dev/config.json --check-inputs
```

## Machine analysis

Use the `repairsbn` Python environment to rebuild development diagnostics:

```bash
python -m repair_data_gen.shared_pipeline.analyse_split \
  --config repair_data_gen/splits/dev/config.json \
  --logs /tmp/rsr_dev_results_20261028/logs
```

`--logs` is optional. Outputs go to the split's `analysis/first_pass/`, or
`--out`. The command reruns existing checks, verifies source and raw-response
identity, records missing requests, and lists surface and replacement-pair
diagnostics. `summary.json` records input/code hashes and runtime versions.
It does not alter synthesis results, labels, prompts or frozen train data.

`machine_pass.tsv` passes the current automatic checks; it is not an accepted
evaluation dataset. `surface_review.tsv` lists additional raw-text differences
from PMB outside the repair. `failure_review.tsv` retains failures, including
quote-normalisation holds. No hold is automatically rescued. Pair collisions
are reported in both directions with type-specific keys; their exclusion
scope must be frozen separately before creating the final audit pool.

For dev, `review_policy.json` now specifies report-only pair overlap and
permitted surface differences. Run `python -m
repair_data_gen.shared_pipeline.prepare_dev_review` to apply that policy
without changing historical checks. Run `python -m
repair_data_gen.shared_pipeline.prepare_dev_topup` to prepare the next
cluster inputs locally. Neither command submits GPU jobs or changes train.
