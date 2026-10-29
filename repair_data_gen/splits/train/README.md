# Train split: frozen legacy lineage

The train data is frozen and remains in its existing locations:

- generation records and checks: `../../rsr_generation/`
- base and marked RSR TSVs: `../../rsr_generation/dataset/`
- unified Parquet and manifest: `../../training_dataset/`

This directory is an index, not a second copy of the data. Do not move or
regenerate train artifacts while building dev. Any future migration must
re-run the legacy builders and reproduce the frozen base SHA-256 exactly.
