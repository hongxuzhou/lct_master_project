# Evaluation runs

Store results by test batch, model size, condition and seed:

```text
<batch>/<size>/<condition>/seed<seed>/predictions.tsv
<batch>/<size>/<condition>/seed<seed>/scores/<slice>.basic.jsonl
<batch>/<size>/<condition>/seed<seed>/scores/<slice>.checks.json
```

Keep each prediction TSV unchanged, including `tokens` and `ended_with_eos`.
Create `scores/` when scoring starts. Slice names follow
[the evaluation workflow](../README.md): `pmb_fluent`, `core_rsr`,
`later_reference` and `hard_negative`.

Keep the original frozen checkpoint manifests and a short README at the
batch root. Record the source `run_id` for each size, condition and seed
when importing predictions. Save aggregate scores as `<batch>/summary.tsv`
after scoring; directories group artifacts, while this table supports
comparisons across runs.

The current batch is [formal-test-20261113](formal-test-20261113/README.md).
