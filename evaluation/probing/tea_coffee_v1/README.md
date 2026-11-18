# Tea–coffee restart probe

One exploratory family, 16 unique inputs, and the selected 9B C3 adapters
from seeds 24, 42 and 4224: 48 independent generations. `stimuli.json` records
exact whitespace, punctuation, parent updates, shared path membership and
cross-branch comparisons. Start at `I ordered tea`; do not parse `I` or
`I ordered` separately. Every request contains only its current cumulative
input and the original parsing instruction. Path names describe stimulus
construction, not unique gold readings.

## Validate and generate

For the prepared LST deployment, see [Submit one LST task](#submit-one-lst-task).

Run from the repository root. These commands do not submit cluster jobs.
Generation uses the existing PyTorch/Transformers/PEFT CUDA environment,
BF16, SDPA, greedy decoding, disabled thinking and the original training
configuration's generation limit (512 tokens in the selected formal runs).
The base model must already be cached locally. Adapters can be prepared
automatically in the same compute job before model loading.

```bash
python evaluation/probing/tea_coffee_v1/run.py --stimuli-only
```

### Prepare adapters in the inference job

Use `--adapter-store` to check the persistent scratch copy and download only
missing files for the current seed from `Shrikes/repair-sbn-formal-adapters`.
The selected config, adapter and tokenizer hashes must match the existing
frozen manifest. Existing mismatched files cause a failure and are preserved.
No other seed or training condition is downloaded.

On first use, the script resolves the repository's current commit and saves
it atomically as `revision.json`. Later runs reuse that pin without contacting
the Hub when all files exist. Concurrent seeds share the first stored pin.
Use `--adapter-revision FULL_COMMIT_SHA` to select another immutable revision;
this uses a separate revision directory and leaves the default pin unchanged.

Run this inside an allocated compute job, not on login/submit. The job needs
Hub network access and existing read authentication for the private repository
only when files are missing. Do not set `HF_HUB_OFFLINE=1` for a first download.
The base model remains offline/local-only. Downloading before inference needs
no separate interactive allocation.

```bash
python evaluation/probing/tea_coffee_v1/run.py \
  --manifest evaluation/runs/formal-test-20261113/frozen-habrok.json \
  --seed 24 \
  --adapter-store /scratch/hongxuzhou/thesis/repair-sbn-formal-adapters \
  --model-cache /scratch/common_models/HuggingFace \
  --output /scratch/hongxuzhou/thesis/restart-probe/seed24-attempt1
```

Repeat for seeds 42 and 4224 with new output paths, using the same adapter
store. Each job prepares its own adapter. Rerunning or changing stimuli does
not require downloading that adapter again. If preparation fails, files already
downloaded remain in the store for the next attempt; prediction output starts
only after preparation succeeds. Download details are printed in the job log,
and successful preparation's repository/revision are saved in `run.json`.

### Use an existing checkpoint directly

Alternatively, `--checkpoint` overrides only the location recorded in the
manifest. `--config` can locate its unchanged training config separately.
Do not combine these arguments with `--adapter-store`.

```bash
python evaluation/probing/tea_coffee_v1/run.py \
  --manifest evaluation/runs/formal-test-20261113/frozen-habrok.json \
  --seed 24 --checkpoint /LST/PATH/seed24/checkpoint-1602 \
  --config /LST/PATH/seed24/config.json \
  --model-cache /LST/PATH/model-cache \
  --output /LST/PATH/probe/seed24-attempt1
```

All three selected adapters are `checkpoint-1602`; their file hashes differ.
`MODEL_CACHE` is also supported. Add `--check` to either mode for local-only
validation without loading a model or downloading files. It fails if the
adapter-store pin or files are missing. Hash verification reads weight files;
run it on a compute node. It does not check GPU or base-model cache completeness.

The output directory must be new. It contains stimulus and training-config
snapshots, `run.json` provenance and flushed `predictions.jsonl` records.
Records retain the prompt, input/generated token IDs, decoded output with
special tokens, formal-style `pred_sbn`, EOS status and generation length.
Failures retain completed rows and mark the attempt failed; rerun in a new
directory. Shared checkpoints are generated once per seed. No prior SBN or
conversation history enters a request.

## Inspect locally

Use the existing `repairsbn` evaluation environment and canonical project
SBN reader; set `SBN_SRC` if needed. Inspection is separate from GPU inference.

```bash
python evaluation/probing/tea_coffee_v1/review.py \
  --run /PATH/probe/seed24-attempt1 --output /PATH/review/seed24-attempt1
```

`manual_review.tsv` retains every checkpoint, raw separator counts, validity,
paired C–C count, output text and blank manual fields. `graphs.jsonl` exposes
resolved node IDs, labels, written boxes, box links and relations with written
and effective scope. IDs are local to each graph. `CONJUNCTION` conditions
merge into their target scope, following the existing reader. Raw separator
counts are lexical evidence only, including in invalid outputs.

Manually identify tea, coffee, milk and the ordering event before assigning
reading preferences. Record missing, withdrawn, non-withdrawn, duplicated or
unresolved objects, and inspect their semantic links separately. Record
marker content and additions unsupported by the supplied prefix. Preserve
alternative readings. Invalid/missing outputs have no inferred repair status;
an incomplete C–C configuration remains available in the raw output. No gold
graph scores, automatic referent matching or preference aggregates are used.

Compare parent–child checkpoints and the explicit branch pairs in the manifest.
Keep seeds separate. This one family describes behaviour on these inputs;
it does not estimate prevalence on the formal test panels or establish
internal commitment or causal revision.

Local contract checks (no model inference):

```bash
python -m unittest discover -s evaluation/probing/tea_coffee_v1 -p 'test_*.py'
```

## Submit one LST task

Build the inference-only archive locally:

```bash
python3 evaluation/probing/tea_coffee_v1/build_bundle.py \
  --output /tmp/repairsbn-restart-probe-v1.tar.gz
```

The archive contains the inference code, original prompt helper, the three
frozen 9B C3 entries and file hashes. It contains no weights, training/test
datasets, credentials or Switchboard material. Graph review runs locally in
the full repository after results are retrieved; the canonical SBN reader is
not included in the cluster bundle.

Upload the archive to `/nethome/honzhou/`, then unpack and submit on LST.
Extraction is small and does not start Python environments or model imports.
The destination must not already exist; keep a running job's bundle unchanged.

```bash
test ! -e /nethome/honzhou/repairsbn-restart-probe-v1 && \
  tar -xzf /nethome/honzhou/repairsbn-restart-probe-v1.tar.gz -C /nethome/honzhou

condor_submit /nethome/honzhou/repairsbn-restart-probe-v1/evaluation/probing/tea_coffee_v1/lst.sub
```

One A100 (40 or 80 GB), four CPUs, 64 GB host RAM, three-hour wall limit.
The three seeds run sequentially inside the same job; there is no separate
interactive/download job. No Conda activation is used. The interpreter is
`/nethome/honzhou/miniconda3/envs/thesis-qwen35-opt/bin/python`; token discovery
uses the explicit shared path `/nethome/honzhou/.cache/huggingface/token`.
No submit-host environment or credentials are printed/inherited.

The worker first verifies the small bundle hashes and base-cache file inventory,
imports the existing runtime and checks the visible A100. Adapter preparation
then validates Hub access as needed, downloads missing files and hashes the
selected adapters. All model imports, downloads and weight verification happen
on the compute node. No package installation is attempted.

Logs are `logs/probe.JOB_ID.{out,err,log}` inside the uploaded directory.
The output prints each input ID, generation length and seed completion.
No W&B run or email notification is configured. Monitor with
`condor_q -submitter honzhou` and read those logs using the actual job number.

Results are retained under
`/scratch/hongxuzhou/thesis/restart-probe/tea_coffee_v1/job-JOB_ID-0/`:
`preflight.json`, separate `seed24/`, `seed42/`, `seed4224/` directories and a
job-local Triton cache. Each seed has 16 prediction records. A seed failure
stops the worker and retains earlier outputs. Resubmission gets a new job
directory and reuses the persistent adapter store; it reruns all three seeds.

Copy seed directories back and run `review.py` from the full local repository.
The runtime and Hub access have not been live-verified by local packaging tests.
