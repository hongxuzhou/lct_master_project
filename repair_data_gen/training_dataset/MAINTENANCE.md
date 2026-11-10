# Dataset maintenance

The public dataset card is `README.md`; public field definitions are in
`SCHEMA.md`. This file records the local assembly and release workflow.

## Inputs and outputs

- Current model inputs: `data/train.parquet` and `data/validation.parquet`.
- Accepted fluent source: `../fluent_group_synthesis_v1/`, pinned by
  `fluent_group.acceptance.json` (200 train and 40 validation documents).
- PMB/RSR source components and their original row order remain unchanged.
- Published baseline: `releases/published-9b7d881/`.
- Data-release commit and checksums: `publication.json`.
- Later documentation-only commits: `documentation.publication.json`.

## Rebuild and check

From the repository root in the `repairsbn` environment:

```bash
python -m repair_data_gen.training_dataset.build
python -m repair_data_gen.training_dataset.build_dev
python -m repair_data_gen.training_dataset.validate_integrated --tokenizer-root /path/to/pinned-tokenizers
```

The validator checks exact preservation of old rows, the accepted supplement,
field mapping, graph/member/quantity validity, split isolation and tokenizer
limits. Results are in `integration.acceptance.json`. Source edits require
renewed acceptance; do not edit the generated Parquets directly.

## Model protocols

C1 selects `pmb_fluent` and `fluent_split_reference`, targeting `sbn_clean`.
C2/C3 use all rows, targeting `sbn_clean` and `sbn_repair` respectively.
The neutral input text does not include provenance or variant metadata.

`model/lora/protocols/fluent_groups_v1.published.json` pins the integrated data
release. The corresponding file without `.published` records its local validation
snapshot. Previous protocols and tuning subsets retain the old published data
revision; use the preserved baseline directory to reproduce those runs.
Documentation-only commits do not require changing a pinned data revision.

## Publication

`prepare_release.py --stage-dir PATH` stages an explicit nine-file allowlist.
`publish_release.py --stage-dir PATH` checks its parent commit, creates an atomic
commit and verifies file/Parquet readback. Parent guards and acceptance records
must be prepared for each new data release. The existing `publication_stage/`
is the historical data-release package, not a second editable data source.

For a card/schema edit with unchanged data, run:

```bash
python -m repair_data_gen.training_dataset.update_card
```

This publishes only `README.md` and `SCHEMA.md`, checks the current parent,
and verifies that the other published files remain unchanged. Public writes
require authorisation. Raw generation outputs and audit labels are never staged.

## Later-reference test preparation

`prepare_later_reference.py` validates the marked working copy, builds the
separate 18-field `later_reference/test.parquet`, checks pair/reference indices,
train/dev isolation, exact Parquet and local HF Datasets loading, then stages
an explicit seven-file allowlist. It does not upload or rebuild train/dev.

```bash
python -m repair_data_gen.training_dataset.prepare_later_reference --stage-dir /tmp/repairsbn-later-reference-release
```

The staged root card adds a `later_reference` configuration with a `test` split.
The existing `default` split paths and bytes are preserved. Challenge review
evidence, checksums and field definitions accompany the Parquet. Before an
authorised upload, verify the current Hub parent and protected default files;
the stage manifest records only the last locally verified documentation revision
until this remote check is performed. Do not use the historical train/dev
publisher, whose parent and file list describe a different release.

## PMB RSR test

`release_test.py stage --stage-dir PATH` maps the accepted paired test TSV to
the existing 24-field schema and stages `data/test.parquet`,
`test.acceptance.json`, and the card and schema updates. The `publish`
action checks the recorded Hub parent, commits that allowlist, and reads
back all files. It also checks exact loading through HF Datasets. Train,
validation and the separate later-reference configuration keep their data.
`test.publication.json` records the verified revision.
