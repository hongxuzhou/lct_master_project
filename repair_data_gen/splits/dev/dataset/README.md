# Development base RSR

`dev_rsr.base.tsv` contains the 390 confirmed, unmarked dev repairs:
60 each for content word, name, number, tense and forwarding; 47 temporal;
43 edge label. Temporal has one candidate per selected source site.

Rebuild from the repository root in the `repairsbn` environment:

```bash
python -m repair_data_gen.shared_pipeline.build_dev_base
```

`../manifests/base_selection.json` records the selected IDs.
`../manifests/base_build.json` records the exported checksum and provenance.
`../analysis/base_validation/summary.json` records independent validation.

The two SBN targets retain their source values. Surface normalisation is
reproduced from the review policy, with model text and offsets retained.
`ambiguous` combines machine flags and confirmed alternative readings;
`marker_priority` is a candidate preference, not a marker assignment.
The base build does not create marked variants or condition-specific files.

`dev_rsr.marked.tsv` contains 122 additional versions, covering all currently
identified multiple-reading base items (31.3% of base). There is no random
top-up. `dev_rsr.augmented.tsv` contains all 390 base and 122 marked rows.
The four seen markers are balanced globally and within each repair type;
`sorry` remains held out. Rebuild with:

```bash
python -m repair_data_gen.shared_pipeline.build_dev_marked
```

The current policy is `../marker_policy.json`. Exact assignments, counts,
checks and output hashes are in `../manifests/marked_build.json`.

## Unified development Parquet

`../../../training_dataset/data/validation.parquet` combines all 1,194 fluent
PMB dev documents with the 512 augmented RSR rows. It uses the train schema
and retains both SBN targets. Rebuild from the repository root with:

```bash
python -m repair_data_gen.training_dataset.build_dev
```

`../../../training_dataset/validation.manifest.json` records source hashes,
row counts and assembly checks. Review and provenance fields beyond the
train schema remain in the source TSVs and their manifests.
The unified split has passed source, schema, graph, pairing and token-length
acceptance. Results are in `../analysis/unified_validation/summary.json`.
The accepted snapshot is published as `validation` in `Shrikes/repair-sbn`
at commit `9b7d8812c49b4d7c6f922fd3ce996d019dc18a5c`. Exact remote readback
is recorded in `../../../training_dataset/validation.publication.json`.
