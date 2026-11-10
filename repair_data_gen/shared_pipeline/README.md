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

After importing follow-up TSVs and raw responses into the dev results folder:

```bash
python -m repair_data_gen.shared_pipeline.analyse_split \
  --config repair_data_gen/splits/dev/config.json \
  --out repair_data_gen/splits/dev/analysis/first_pass_rechecked
python -m repair_data_gen.shared_pipeline.analyse_split \
  --config repair_data_gen/splits/dev/config.json \
  --followup-manifest repair_data_gen/splits/dev/manifests/followup_samples.json \
  --out repair_data_gen/splits/dev/analysis/followup
python -m repair_data_gen.shared_pipeline.prepare_dev_review \
  --analysis repair_data_gen/splits/dev/analysis/first_pass_rechecked \
  --analysis repair_data_gen/splits/dev/analysis/followup
```

This checks the follow-up separately and appends eligible requests to the
existing queue. Existing item IDs, order and labels are preserved. Keep both
analysis arguments on subsequent rebuilds of this combined queue.

## Confirmed development base

```bash
python -m repair_data_gen.shared_pipeline.build_dev_base
```

This exports the confirmed 390-item selection to
`splits/dev/dataset/dev_rsr.base.tsv`. It checks selection and curation
identity, raw-response provenance, PMB sources, offsets, graph validity,
WordNet, gold recovery and source isolation. It also checks insertion and
removal of all four seen markers without writing marked data. Validation
results are in `splits/dev/analysis/base_validation/`; the output checksum
and provenance are in `splits/dev/manifests/base_build.json`.

## Development marked variants

```bash
python -m repair_data_gen.shared_pipeline.build_dev_marked
```

The current `splits/dev/marker_policy.json` covers all 122 identified
multiple-reading items, with no random top-up. The historical generation
config remains unchanged. The builder writes 122 marked rows and 512
augmented RSR rows, validates exact base recovery and unchanged targets,
and records assignments and checksums in `manifests/marked_build.json`.
