# PMB test synthesis

The initial batch contains 507 unmarked generation requests from PMB 5.1.0
English test standard. Generate on LST, import results locally, then review
all machine-eligible base repairs for meaning and naturalness. Discourse
challenges and non-RSR hard negatives are separate tasks.

## Frozen first-batch inputs

| Stage | Requests | Prompt |
| --- | ---: | --- |
| Base: content word, name, number, tense, forwarding | 372 (75/75/72/75/75) | `base` |
| Temporal: first eligible site per document | 68 | `temporal_r2` |
| Edge: direction | 29 | `edge_r3` |
| Edge: place only, first eligible site | 38 | `place_r5` |

`config.json` and `manifests/candidate_samples.json` identify the sources,
seeds, samples and prompt hashes. After a run uses them, do not edit them;
prepare any follow-up under a new manifest and tag. No automatic retries or
second temporal pass are part of this batch. Empty/transport failures may
receive one separately recorded retry; answered failures and `no_site` are
not rerolled.

Candidate selection found no train/test or dev/test document-ID or exact
sentence overlap. Gold document `p32/d1803` fails the existing structural
checks and is excluded from synthesis. This does not change the fluent PMB
evaluation partition. Replacement-pair overlap is report-only against train
and dev, never an exclusion. Near-duplicate diagnostics do not establish
absence of semantic overlap.

`review_policy.json` records the first-round checks and surface policy.
Apply [the final selection policy](selection_policy.json) to both rounds:
review unresolved labels against the train/dev standard, keep every eligible
base with no repeated source/type/site, and treat 60 per type as a sampling
target. An alternative reading alone does not disqualify a natural repair
with a faithful graph. Report per-type results and their equal-weight mean;
group variants by source document for comparisons.

The test measures recovery of the intended synthetic repair from text.
Preference between readings with and without prosody, pauses, or gesture
is a separate research question. Preserve raw responses, run TSVs and
original labels; record adjudications separately.

`marker_policy.json` specifies 2N test rows: every selected base plus exactly
one scripted marked version, with unchanged targets. Balance `I mean`, `no`,
`no wait`, `actually` and held-out `sorry` within types and globally. The author
spot-checked the assigned forms and waived exhaustive naturalness review.
Train/dev RSR marker assignment continues to exclude `sorry`. A later
expanded sorry probe is a separate release.

The [local PMB RSR test partition](dataset/README.md) contains 559 bases and
559 marked counterparts. The acceptance manifest pins these exports.

## Local preparation

From the repository root, using the `repairsbn` Python environment:

```bash
python -m repair_data_gen.shared_pipeline.select_candidates repair_data_gen/splits/test/config.json
python -m repair_data_gen.shared_pipeline.build_test_submission --out /tmp/test_synthesis_v1.tar.gz
```

The archive uses an explicit allowlist: generation harness, WordNet helper,
four immutable prompt versions, test samples, policies and submit files. It
contains no train/dev results, full PMB corpus or Switchboard material. The
legacy `run.sh` is copied with only its CODE directory changed; the original
is untouched. The archive includes source and packaged checksums.

## LST upload and submission

This is an isolated runtime, leaving the existing cluster train/dev checkout
untouched. The archive builder refuses to overwrite an existing archive.
Upload from the laptop:

```bash
scp /tmp/test_synthesis_v1.tar.gz honzhou@login.lst.uni-saarland.de:/nethome/honzhou/test_synthesis_v1.tar.gz
ssh honzhou@login.lst.uni-saarland.de
```

On the login host, extract once. Refuse an existing runtime rather than
replacing code beneath a resumed job:

```bash
test ! -e /nethome/honzhou/thesis_zone/test_synthesis_v1 && tar -xzf /nethome/honzhou/test_synthesis_v1.tar.gz -C /nethome/honzhou/thesis_zone
ssh submit
```

On the submit host:

```bash
cd /nethome/honzhou/thesis_zone/test_synthesis_v1
bash splits/test/hpc/preflight.sh
condor_submit_dag splits/test/hpc/run_test_all.dag
condor_q -submitter honzhou
```

Run submission only after preflight passes. The DAG runs place, temporal,
direction and base sequentially, using one GPU per job, with no automatic
retry. Dependencies gate process exit status, not semantic acceptance.
Resources match the dev jobs: 8 CPUs, 64 GB host RAM, one >=80 GB GPU,
24-hour wall time per stage. The model is Muse-Glimmer-30B with the same
harness defaults as train/dev: reasoning high, temperature 1.0, top-p 0.95,
top-k 64, generation seed 13, and 16 workers.

## Import and review

Copy only this runtime's `splits/test/results/` into a fresh local staging
directory, together with the expanded `test_*.out/.err/.log` files from
`/scratch/hongxuzhou/logs/llm_gen/`. Preserve raw replies and every run_meta
file. Verify packaged checksums and run metadata before importing; do not
copy the legacy train/dev results directories.

Next local tasks, after generation: validate response/sample/raw identity;
rerun structural, WordNet, untouched and gold-recovery checks; apply the
recorded surface policy; report train/dev pair overlap; prepare stable-ID
blind review; export the accepted base; then derive the paired forms.
The [test build](dataset/README.md) is separate from the preparation commands
above.

## Machine quality checks

Read results directly from the downloaded runtime; do not overwrite raw run
files. From the repository root in the `repairsbn` environment:

```bash
python -m repair_data_gen.shared_pipeline.analyse_split \
  --config repair_data_gen/splits/test/config.json \
  --results-dir /path/to/test_synthesis_v1/splits/test/results \
  --logs /path/to/download/logs \
  --out repair_data_gen/splits/test/analysis/first_pass
python -m repair_data_gen.shared_pipeline.analyse_test_review \
  --analysis repair_data_gen/splits/test/analysis/first_pass \
  --out repair_data_gen/splits/test/analysis/review_policy_v1
```

`first_pass/` retains raw checks. `review_policy_v1/` applies the recorded
policy and writes review candidates, exclusions, no-site replies, surface
changes, incomplete alignments and train/dev pair diagnostics. Pair overlap
never excludes a candidate. Incomplete alignment retains the original model
text; it needs surface review rather than guessed punctuation fixes.
These tables are machine diagnostics, not human labels or final test data.
