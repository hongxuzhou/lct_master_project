# v2 cluster execution

| Size | Cluster / GPU | Host resources | Job limit |
| --- | --- | --- | --- |
| 2B | Hábrók A100 | 4 CPUs, 64 GB RAM | 12 hours |
| 9B | LST A100 40 GB | 4 CPUs, 64 GB RAM | 24 hours |
| 27B | LST H100 80 GB | 8 CPUs, 128 GB RAM | 72 hours |

Time limits are headroom for a complete nine-trial condition, not measured
runtime guarantees. By default one command submits all three conditions, with
nine sequential trials in each GPU job. Jobs run independently as GPUs become
available. There is no mandatory fit check, first-trial pause or manual gate.
After each agent finishes, it automatically writes a condition-specific result
summary. No environment installation, automatic retry or training resume is
performed; only actual failures or a time-limit interruption require recovery.

Use a separate uploaded code directory, such as `lora_tuning_v2`, while old jobs
are active. Transfer code, protocols and subset manifests; exclude local results,
checkpoints, credentials and unrelated corpus files. Do not overwrite an active
round's code. Cluster paths and GPU matching still need live verification.

## Prepare in a permitted CPU allocation

The uploaded LST bundle can perform preparation automatically in a CPU DAG
parent, followed by all six GPU agents. On the submit host, after staging data:

```bash
python3 tuning/lst/build_round_v2.py
condor_submit_dag tuning/lst/round-v2/round.dag
```

The DAG's CPU worker runs `lst/prepare_uploaded_v2.sh`: offline tokenisation
preflight and one-time registration for both 9B and 27B. It then releases all
six GPU jobs automatically. It reuses existing model caches and environments;
no login/submit-host analysis, environment installation or manual handoff after
preparation is required. This concrete uploaded workflow fixes honzhou's paths
in `lst/environment_uploaded_v2.sh`. The generic manual preparation commands
below remain available for individual model submissions.

From the uploaded code root, reuse the existing environment:

```bash
# Hábrók CPU/preparation allocation
source tuning/habrok/environment_v2.sh
bash tuning/prepare_v2.sh habrok 2B

# LST CPU allocation: never run preparation on login/submit
source tuning/lst/environment_v2.sh
bash tuning/prepare_v2.sh lst 9B
bash tuning/prepare_v2.sh lst 27B
```

Defaults use dataset directory `repair-sbn-d1ed13ec` and result root
`qwen35_lora_tuning/lr-rank-v2`. Explicit environment overrides are preserved:
check that inherited `DATASET_DIR` and `TUNING_RUN_ROOT` do not point to v1.
Preparation downloads the pinned model/data incrementally, runs complete
tokenisation preflight and validates all condition-specific subset schedules.
It creates no sweep or GPU job. It requires the `hf` CLI in the existing
environment; override `HF_CLI` if necessary.

Register each size once, on a permitted preparation node with W&B access:

```bash
# On Hábrók
"$PYTHON" tuning/register_sweeps.py --plan lr-rank-v2 --model-size 2B \
  --create --registry tuning/registrations/lr-rank-v2-2B.json

# On LST
"$PYTHON" tuning/register_sweeps.py --plan lr-rank-v2 --model-size 9B \
  --create --registry tuning/registrations/lr-rank-v2-9B.json
"$PYTHON" tuning/register_sweeps.py --plan lr-rank-v2 --model-size 27B \
  --create --registry tuning/registrations/lr-rank-v2-27B.json
```

Keep partial registries if a request fails. Do not repeat registration to obtain
replacement sweep IDs. Workers check registry size, condition, plan, protocol
and subset hashes before accepting assignments.

## Optional GPU fit checks

These commands only preview scheduler commands. Append `--check` to validate
with Slurm/HTCondor without submitting, then `--submit` to request the GPU job.
Run the LST commands from the submit host. Scheduler checks do not establish GPU
availability or model fit. Inspect `condor_status -af Name GPUs_Capability
GPUs_GlobalMemoryMb` and the site's resource listing before submission.

```bash
bash tuning/habrok/submit_v2.sh --condition C3 --fit-check
bash tuning/lst/submit_v2.sh --model-size 9B --condition C3 --fit-check
bash tuning/lst/submit_v2.sh --model-size 27B --condition C3 --fit-check
```

Each fit check is a separate calibration: three optimiser steps at rank 64 /
alpha 128, using the new subset and full condition dev, followed by best-adapter
reload validation. It uses no sweep assignment and logs no W&B run. Results go
under `lr-rank-v2/fit-checks/`. These calibrations are additional to the 81
planned tuning runs. Check `summary.json` for `development_passed` and inspect
peak GPU memory. These checks are optional diagnostics, not prerequisites for submission.
An A100 40 GB failure does not automatically trigger an H100 or A100 80 GB job.

The LST submit template restricts 9B to capability 8.0 / 39,000–41,000 MB and
27B to capability 9.0 / 79,000–85,000 MB in the local shared-filesystem pool.
The worker additionally checks the actual allocated name for A100 or H100.
These use the site's saved `GPUs_*` attributes; confirm their live values and
run the scheduler dry run. See the [HTCondor GPU matching reference](https://htcondor.readthedocs.io/en/main/users-manual/submitting-a-job.html#jobs-that-require-gpus).

## Submit the complete tuning round

After data preparation and one-time registration, run these from the uploaded
code root. Each command submits C1/C2/C3 with nine trials per condition; it does
not wait for one condition to finish before submitting the others.

```bash
# Hábrók: 27 trials for 2B
bash tuning/habrok/submit_v2.sh --submit

# LST submit host: 27 trials each for 9B and 27B
bash tuning/lst/submit_v2.sh --model-size 9B --submit
bash tuning/lst/submit_v2.sh --model-size 27B --submit
```

Omit `--submit` for a preview, or use `--check` for a scheduler dry run. Neither
is a required manual stage. Registry paths default to
`tuning/registrations/lr-rank-v2-SIZE.json`; override with `--registry` if needed.
Use `--condition C2` to target one condition and `--count N` only for a partial
batch or recovery. The W&B agent chooses candidates; count does not choose a
particular LR/rank. Do not resubmit a healthy active sweep.

Condition summaries are written automatically to
`lr-rank-v2/summaries/SIZE-CONDITION.json`. Best-run selection still requires
nine distinct successful candidates with matching provenance; a failed or
incomplete condition is reported without inventing a winner. After a failure,
reuse its registered sweep and recover missing configurations explicitly.

The historical v1 `agent.sbatch` and old `prepare.sh` remain unchanged; use only
the v2 entry points for this round. No commands here have been live-validated or
submitted to either cluster as part of local wrapper preparation.
