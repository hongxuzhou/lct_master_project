# HTCondor jobs: open-model runs (LST cluster, vanilla universe)

Paths assume the playground lives at
`/nethome/honzhou/thesis_zone/llm_gen_playground/`, the venv at
`/nethome/honzhou/venvs/vllm/`, caches and logs under `/scratch/hongxuzhou/`.
The weights follow the cluster rule and live in the shared
`/scratch/common_models/Muse-Glimmer-30B/` (`MODEL_DIR` in `setup.sh`,
`MODEL_PATH` in `run.sh`; the two must agree). Edit the variables at the top
of the `.sh` files if yours differ.

Before the first setup, check how the other models there are named and
whether you can write to it:
```bash
ls /scratch/common_models/ ; test -w /scratch/common_models && echo writable
```
If it is not writable, setup.sh skips the download and says so; ask the
admins to add the model, then go on from step 2.

## Copy and prepare
Files go through the login node with scp (LST wiki, "File Transfer"); the
home folder is shared, so the submit node sees them. Only what the jobs need
is copied, so the cluster's `results/` is never overwritten:
```bash
# on the laptop, in repair_data_gen/llm_gen_playground/
scp -r run_llm.py wn_lookup.py \
    sample_train_full.tsv sample_r3_edge.tsv prompt_docs prompt_docs_edge_r3 hpc \
    honzhou@login.lst.uni-saarland.de:thesis_zone/llm_gen_playground/
md5 hpc/setup.sh hpc/run.sh        # the jobs print the same sums in .out

# then: ssh honzhou@login.lst.uni-saarland.de, then: ssh submit
chmod +x /nethome/honzhou/thesis_zone/llm_gen_playground/hpc/*.sh
mkdir -p /scratch/hongxuzhou/logs/llm_gen
```
Results come back the same way, **only the new run's files**: copying the
whole `results/` once overwrote a newer local `claude-opus-5-5.checked.tsv`
with the cluster's older copy (2026-10-04).
```bash
R=honzhou@login.lst.uni-saarland.de:thesis_zone/llm_gen_playground/results
scp "$R/meta-models_Muse-Glimmer-30B.train_full.tsv" results/
scp -r "$R/raw/meta-models_Muse-Glimmer-30B.train_full" results/raw/
```

## Run order
```bash
# 1. SETUP (CPU, once): venv, WordNet, weights, and a report
condor_submit setup.sub
#    read setup.*.out: "vllm 0.30.0+cu129" and
#    "MuseGlimmerForConditionalGeneration supported: True"?
#    muse_glimmer parser files listed? (else TOOL_MODE=text in the .sub files)

# 2. SMOKE: 3 items -> results/meta-models_Muse-Glimmer-30B.smoke.tsv
condor_submit run_smoke.sub
#    check: parse=ok, turns >= 2 on content_word (the lookup ran), and the raw
#    transcript in results/raw/meta-models_Muse-Glimmer-30B.smoke/

# 3. FULL: 70 items -> results/meta-models_Muse-Glimmer-30B.tsv (resumable)
condor_submit run_full.sub

# 4. pull results/ back and score locally with analyse_run.py etc.
```

Monitor with `condor_q`, then read the `.out`/`.err` files under
`/scratch/hongxuzhou/logs/llm_gen/`.

## If something fails

- **Still `vllm 0.24.0`**: the job ran an old `setup.sh`. Compare the
  `script:` md5 at the top of `.out` with `md5 hpc/setup.sh` locally.
- **Architecture not supported**: the installed vLLM is too old for
  `muse_glimmer` (needs >= 0.28; `VLLM_VERSION` in `setup.sh`).
- **`torch sees no GPU` or CUDA errors**: first check the two
  `CUDA_VISIBLE_DEVICES` lines at the top of `.out` (the UUID → index
  renaming). If those look right, the cu129 build does not run on the 12.8
  driver after all; ask the admins whether a newer driver is planned.
- **No tool calls in the transcript** (`turns=1` on every content-word item,
  ATEM text in `result`): try `TOOL_MODE=auto` with a parser from the setup
  log, or check what the raw `content` looks like before changing the regex.

## Full run on train and round 3 (2026-10-04)
```bash
condor_submit hpc/run_train_full.sub   # six types, prompt_docs/
condor_submit hpc/run_r3_edge.sub      # edge label, prompt_docs_edge_r3/
```
Both can run at once (one GPU each). Each `run_meta_*.json` records `docs`
and `prompt_sha256`; locally `python3 -c "import run_llm; print(run_llm.prompt_hash())"`
gives the hash of `prompt_docs/` (`6b17847a…`; edge round 3 `1886b812…`).

## Edge label on train (2026-10-08)
```bash
# laptop, in llm_gen_playground/: only the new sample and the job file
scp sample_train_edge.tsv honzhou@login.lst.uni-saarland.de:thesis_zone/llm_gen_playground/
scp hpc/run_train_edge.sub honzhou@login.lst.uni-saarland.de:thesis_zone/llm_gen_playground/hpc/
# submit node, in thesis_zone/llm_gen_playground/
condor_submit hpc/run_train_edge.sub
# back on the laptop, via a staging folder as before
mkdir -p incoming
R=honzhou@login.lst.uni-saarland.de:thesis_zone/llm_gen_playground/results
scp "$R/meta-models_Muse-Glimmer-30B.train_edge.tsv" incoming/
scp -r "$R/raw/meta-models_Muse-Glimmer-30B.train_edge" incoming/
mv -n incoming/*.tsv results/ && mv -n incoming/meta-models_Muse-Glimmer-30B.train_edge results/raw/ && rmdir incoming
```

## Edge label, places (2026-10-10)
New: the sample, the job file, and the r4 prompt documents (sha256
`24c14008…`). `select_docs.py` and `analyse_run.py` changed too, but run
locally only.
```bash
# laptop, in llm_gen_playground/
L=honzhou@login.lst.uni-saarland.de:thesis_zone/llm_gen_playground
scp sample_train_place.tsv $L/
scp hpc/run_train_place.sub $L/hpc/
scp -r prompt_docs_edge_r4 $L/
# submit node, in thesis_zone/llm_gen_playground/
condor_submit hpc/run_train_place.sub
# back on the laptop, via a staging folder as before
mkdir -p incoming
R=$L/results
scp "$R/meta-models_Muse-Glimmer-30B.train_place.tsv" incoming/
scp -r "$R/raw/meta-models_Muse-Glimmer-30B.train_place" incoming/
mv -n incoming/*.tsv results/ && mv -n incoming/meta-models_Muse-Glimmer-30B.train_place results/raw/ && rmdir incoming
```

## Temporal, second item per document (2026-10-10)
New: the sample, the job file, and `prompt_docs_temporal_r2/` (sha256
`8d49b9ac…`). Runs alongside the place job (one GPU each).
```bash
# laptop, in llm_gen_playground/
L=honzhou@login.lst.uni-saarland.de:thesis_zone/llm_gen_playground
scp sample_train_temporal_extra.tsv $L/
scp hpc/run_train_temporal_extra.sub $L/hpc/
scp -r prompt_docs_temporal_r2 $L/
# submit node, in thesis_zone/llm_gen_playground/
condor_submit hpc/run_train_temporal_extra.sub
# back on the laptop, via a staging folder as before
mkdir -p incoming
R=$L/results
scp "$R/meta-models_Muse-Glimmer-30B.train_temporal_extra.tsv" incoming/
scp -r "$R/raw/meta-models_Muse-Glimmer-30B.train_temporal_extra" incoming/
mv -n incoming/*.tsv results/ && mv -n incoming/meta-models_Muse-Glimmer-30B.train_temporal_extra results/raw/ && rmdir incoming
```

## Forwarding, extra documents (2026-10-10)
New: the sample and the job file only; `prompt_docs/` is the full run's.
```bash
# laptop, in llm_gen_playground/
L=honzhou@login.lst.uni-saarland.de:thesis_zone/llm_gen_playground
scp sample_train_forwarding_extra.tsv $L/
scp hpc/run_train_forwarding_extra.sub $L/hpc/
# submit node, in thesis_zone/llm_gen_playground/
condor_submit hpc/run_train_forwarding_extra.sub
# back on the laptop
mkdir -p incoming
R=$L/results
scp "$R/meta-models_Muse-Glimmer-30B.train_forwarding_extra.tsv" incoming/
scp -r "$R/raw/meta-models_Muse-Glimmer-30B.train_forwarding_extra" incoming/
mv -n incoming/*.tsv results/ && mv -n incoming/meta-models_Muse-Glimmer-30B.train_forwarding_extra results/raw/ && rmdir incoming
```

## Edge label, places, round 5 (2026-10-11)
New: two samples, two job files, and `prompt_docs_place_r5/` (sha256
`cf0cb470…`). Planned: pilot first (48 documents), the full job (458)
after. Done (2026-10-11): pilot skipped, `run_place_r5.sub` submitted
directly; for its results, replace `place_r5_pilot` with `place_r5` in the
commands below. `sample_place_r5.py` and `check_place_r5.py` run
locally only.
```bash
# laptop, in llm_gen_playground/
L=honzhou@login.lst.uni-saarland.de:thesis_zone/llm_gen_playground
scp sample_place_r5_pilot.tsv sample_place_r5.tsv $L/
scp hpc/run_place_r5_pilot.sub hpc/run_place_r5.sub $L/hpc/
scp -r prompt_docs_place_r5 $L/
# submit node, in thesis_zone/llm_gen_playground/
condor_submit hpc/run_place_r5_pilot.sub
# back on the laptop
mkdir -p incoming
R=$L/results
scp "$R/meta-models_Muse-Glimmer-30B.place_r5_pilot.tsv" incoming/
scp -r "$R/raw/meta-models_Muse-Glimmer-30B.place_r5_pilot" incoming/
mv -n incoming/*.tsv results/ && mv -n incoming/meta-models_Muse-Glimmer-30B.place_r5_pilot results/raw/ && rmdir incoming
# check
python3 analyse_run.py results/meta-models_Muse-Glimmer-30B.place_r5_pilot.tsv \
  --out results/meta-models_Muse-Glimmer-30B.place_r5_pilot.checked.tsv
python3 check_place_r5.py results/meta-models_Muse-Glimmer-30B.place_r5_pilot.checked.tsv
```

## Edge label, places, round 5 rerun (2026-10-12)
The 48 pilot documents plus the 2 round-5 documents with no reply
(`p01/d2312`, `p28/d3262`): `sample_place_r5_rerun.tsv` (50),
`hpc/run_place_r5_rerun.sub`, same prompt (`prompt_docs_place_r5/`, already
on the cluster), so the results pool with round 5. Edge label needs about 24
more selected items.
```bash
# laptop, in llm_gen_playground/
L=honzhou@login.lst.uni-saarland.de:thesis_zone/llm_gen_playground
scp sample_place_r5_rerun.tsv $L/
scp hpc/run_place_r5_rerun.sub $L/hpc/
# submit node, in thesis_zone/llm_gen_playground/
condor_submit hpc/run_place_r5_rerun.sub
# back on the laptop
mkdir -p incoming
R=$L/results
scp "$R/meta-models_Muse-Glimmer-30B.place_r5_rerun.tsv" incoming/
scp -r "$R/raw/meta-models_Muse-Glimmer-30B.place_r5_rerun" incoming/
mv -n incoming/*.tsv results/ && mv -n incoming/meta-models_Muse-Glimmer-30B.place_r5_rerun results/raw/ && rmdir incoming
# check (check_place_r5.py reads both r5 samples by default; the rerun's
# documents are all in them)
python3 analyse_run.py results/meta-models_Muse-Glimmer-30B.place_r5_rerun.tsv \
  --out results/meta-models_Muse-Glimmer-30B.place_r5_rerun.checked.tsv
python3 check_place_r5.py results/meta-models_Muse-Glimmer-30B.place_r5_rerun.checked.tsv
```
