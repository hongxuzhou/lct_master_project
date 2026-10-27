# LST Qwen3.5 LoRA smoke test

This directory is the HTCondor adapter for the shared smoke-test Python code.
It uses the existing `thesis` environment, the shared model cache, and pinned
revisions from `../revisions.json`.

## Remote layout

Synchronize the complete local `model/smoke_test/` directory to:

```text
/nethome/honzhou/thesis_zone/smoke_test/
```

Do not upload the Python and wrappers separately. The wrappers resolve the
shared files relative to their own location.

Before submission, create the log and result parents and restore executable
bits if the transfer tool removed them:

```bash
mkdir -p /scratch/hongxuzhou/logs /scratch/hongxuzhou/thesis/qwen35_lora_smoke
chmod +x /nethome/honzhou/thesis_zone/smoke_test/lst/*.sh
```

## CPU preparation and preflight

The LST wiki says not to run analysis or environment preparation on login or
submit hosts. Submit the CPU-only preparation job from the `lst/` directory:

```bash
condor_submit prepare.sub
```

It checks Hugging Face authentication using the token in the LST home
directory without changing `HF_HOME`, populates the pinned dataset cache, then
runs offline tokenization preflight for all three pinned model revisions. Do
not request a GPU until it passes.

## GPU canary, then smoke matrix

After the preparation job exits successfully:

```bash
condor_submit qwen35_lora_smoke_canary.sub
```

Confirm that the 2B repair canary exits successfully and that its copied
`summary.json` reports `"status": "passed"`. Then submit the remaining five
jobs:

```bash
condor_submit qwen35_lora_smoke_test.sub
```

The canary plus the five remaining jobs cover all six model-size/condition
combinations. HTCondor may keep jobs queued under the student one-GPU policy.
The submit files prevent flocking to a pool that lacks the LST `/nethome` and
`/scratch` paths.

GPU jobs run fully offline. When `_CONDOR_SCRATCH_DIR` is available, results
are written locally during the run and copied to `/scratch/hongxuzhou` on
exit. A run passes only when the Condor job exits successfully and its copied
`summary.json` reports `"status": "passed"`.
