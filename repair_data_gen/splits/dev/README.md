# Development split workspace

Status: first synthesis batch analysed; formal human-review pool prepared.
No development dataset has been frozen.

The source is all 1,194 documents in PMB 5.1.0 English standard dev. The
working design caps accepted base RSR items at 60 per repair type, retains all
accepted base inputs, and appends marked variants for 30% of accepted sites.
The four seen markers are `I mean`, `no`, `no wait` and `actually`; `sorry`
is excluded for the held-out test probe.

Directory roles:

- `samples/`: frozen candidate manifests sent to the cluster
- `results/`: parsed generation results and retained run metadata
- `audit/`: blind human-review inputs and append-only labels
- `dataset/`: derived base, marked and unified dev data
- `manifests/`: seeds, hashes, counts, token statistics and leakage checks
- `hpc/`: dev-specific HTCondor submit files only

Shared code belongs in `../../shared_pipeline/`, not in this directory.
Validate the working config from the repository root with:

```bash
python3 repair_data_gen/shared_pipeline/split_config.py \
  repair_data_gen/splits/dev/config.json --check-inputs
```

The config is the historical first-generation configuration. Keep it unchanged
because its hash is recorded in the run metadata. Review decisions are stored
separately in `review_policy.json`: replacement-pair overlap is diagnostic
only and never excludes an item; context near-duplicate checks are deferred.
The historical `pair_exclusion_sources` field identifies the comparison data,
not an active exclusion rule under this review policy.

## Formal human review

Use the `repairsbn` Python environment from the repository root:

```bash
python -m repair_data_gen.shared_pipeline.prepare_dev_review
python repair_data_gen/rsr_generation/audit.py serve \
  repair_data_gen/splits/dev/audit/dev_review_v1 --annotator hz --port 8766
```

The policy retains nonsemantic surface variants, including contraction and
tense-auxiliary realisation, for formal meaning/naturalness review. Structural,
gold-content and repair-direction failures remain exclusions. Normalisation
reuses train alignment helpers and preserves model text, offsets and every
semantic target. Raw checks are retained; policy eligibility is a separate
field. Items that cannot be aligned automatically are explicitly recorded.

`analysis/dev_review_v1/` contains the review pool, exclusions, reclassified
items and surface changes. This pool is not the final per-type capped dataset.
The frontend uses stable request-derived IDs. Once labels exist, reruns keep
the existing order and append fresh items without editing labels. The earlier
`audit/first_pass/` queue is preliminary; use `audit/dev_review_v1/`.

Prepare follow-up inputs locally with:

```bash
python -m repair_data_gen.shared_pipeline.prepare_dev_topup
```

This creates timeout retries and temporal pass 2 under new tags, plus
`hpc/run_dev_followup.dag` and `manifests/followup_samples.json`. It does not
submit jobs. Original samples, results and prompt versions remain unchanged.

## Cluster layout

The submit files assume the established cluster checkout:

```text
/nethome/honzhou/thesis_zone/
  rsr_generation/
  splits/dev/
```

Copy the generated samples, config and submit files without replacing legacy
train results. The submit files run the shared legacy harness but set
`RESULTS_DIR=../splits/dev/results`, so dev outputs cannot enter
`rsr_generation/results/`. They are prepared only; submission remains a
separate human-authorised step.

`hpc/run_dev_all.dag` submits the four stages in canary-first order. Each
child is released only after its parent exits successfully, so a failed
edge-place canary cannot trigger the remaining GPU work overnight.
