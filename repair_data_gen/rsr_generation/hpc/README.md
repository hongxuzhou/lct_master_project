# Running generation on the LST cluster

HTCondor jobs, vanilla universe. One job = one vLLM server on one GPU +
`run_llm.py` against it (`run.sh`); the server listens on 127.0.0.1 only and
stops with the job.

Paths (edit the variables at the top of the `.sh` files if yours differ):

| what | where |
|---|---|
| code | `/nethome/honzhou/thesis_zone/rsr_generation/` |
| venv | `/nethome/honzhou/venvs/vllm/` |
| weights | `/scratch/common_models/Muse-Glimmer-30B/` (`MODEL_DIR` in `setup.sh` and `MODEL_PATH` in `run.sh` must agree) |
| logs | `/scratch/hongxuzhou/logs/llm_gen/` |

Batch jobs may run up to one week (`+MaxWallTime = 604800`); 24 h is the
interactive limit. A student account runs one job at a time.

## Setup (once)

```bash
ls /scratch/common_models/ ; test -w /scratch/common_models && echo writable
condor_submit hpc/setup.sub
# setup.*.out must show "vllm 0.30.0+cu129" and
# "MuseGlimmerForConditionalGeneration supported: True"
```

If `common_models` is not writable, `setup.sh` skips the download; ask the
admins to add the model.

## One run

Each `.sub` file sets `SAMPLE` (relative to the code folder), `DOCS` (prompt
version), `TAG` (results suffix) and the parser options. `RESULTS_DIR` may
isolate another split; it defaults to `results/`. `CONFIG` records and hashes
that split's config in `run_meta`. New split samples carry `request_id`, which
is the resume key and permits separate requests from one source document.
Legacy samples without it continue to resume by `doc_id`.

Results go to `<RESULTS_DIR>/meta-models_Muse-Glimmer-30B.<TAG>.tsv` and
`<RESULTS_DIR>/raw/…<TAG>/`; an interrupted run resumes.

```bash
# laptop, in repair_data_gen/rsr_generation/: copy only what is new
L=honzhou@login.lst.uni-saarland.de:thesis_zone/rsr_generation
scp samples/sample_x.tsv $L/samples/
scp hpc/run_x.sub $L/hpc/
scp -r prompts/<version> $L/prompts/        # if the prompt version is new
md5 hpc/run.sh                               # the job prints the same sum

# ssh to the login node, then `ssh submit`
cd thesis_zone/rsr_generation && condor_submit hpc/run_x.sub
condor_q

# back on the laptop: only the new run's files, through a staging folder
mkdir -p incoming
scp "$L/results/meta-models_Muse-Glimmer-30B.x.tsv" incoming/
scp -r "$L/results/raw/meta-models_Muse-Glimmer-30B.x" incoming/
mv -n incoming/*.tsv results/ && mv -n incoming/meta-models_Muse-Glimmer-30B.x results/raw/ && rmdir incoming
```

Never copy the whole `results/` back: it once overwrote a newer local file
with the cluster's older copy. Check the prompt hash in the new
`run_meta_*.json` against `prompts/README.md`.

## Moving the cluster copy to the current layout (once)

The cluster copy still has the layout from before 2026-10-13
(`thesis_zone/llm_gen_playground/`, flat samples, `prompt_docs*`). Keep its
`results/` and refresh the rest:

```bash
# login node
mv thesis_zone/llm_gen_playground thesis_zone/rsr_generation
# laptop, in repair_data_gen/rsr_generation/
L=honzhou@login.lst.uni-saarland.de:thesis_zone/rsr_generation
scp -r run_llm.py wn_lookup.py samples prompts hpc $L/
# submit node
chmod +x thesis_zone/rsr_generation/hpc/*.sh
```

## If something fails

- **Old vLLM version in the log**: the job ran an old `setup.sh`; compare the
  `script:` md5 at the top of `.out` with `md5 hpc/setup.sh`.
- **Architecture not supported**: vLLM too old for `muse_glimmer` (needs ≥
  0.28; `VLLM_VERSION` in `setup.sh`).
- **`torch sees no GPU`**: check the two `CUDA_VISIBLE_DEVICES` lines at the
  top of `.out` (HTCondor gives GPU UUIDs; `run.sh` renames them to indices).
- **No tool calls** (one turn per item, tool syntax in the text): set
  `TOOL_MODE=auto` with the parser named in the setup log, or
  `TOOL_MODE=text`.
- **No row and no raw file for a document**: a request error; rerun those
  documents with a small sample of their own.
