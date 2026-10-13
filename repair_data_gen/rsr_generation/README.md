# RSR generation

Code and records that produced the train RSR data
(`dataset/train_rsr.base.tsv`). An LLM receives a gold PMB sentence, its gold
SBN and a repair type, and writes the repaired sentence and SBN; scripts
check every item; a sample of the passing items becomes the dataset.

- **What the data is and how good it is:** `dataset/README.md`.
- **Why it was built this way:** `../DESIGN.md` and `../DECISIONS.md`.
- **This file:** how the pipeline runs.

This directory was called `llm_gen_playground/` until 2026-10-13; job files
and `run_meta` records written before then use the old name. The full
round-by-round log is this README at git tag `rsr-train-base-log`.

## Layout

| path | contents |
|---|---|
| `prompts/` | the prompt versions (system prompt = SBN manual, repair notation, task); see `prompts/README.md` |
| `samples/` | which gold documents each run used (`select_docs.py` output) |
| `results/` | one TSV per run, plus `.checked.tsv` with every check; raw replies in `results/raw/` (only `run_meta_*.json` is in git) |
| `audit/` | human labels (`labels.jsonl`), merged views, and the edge-label `selection.tsv` |
| `dataset/` | surface fixes, assembly, the final data |
| `hpc/` | job files for the LST cluster |
| `natural_rsr_check/` | comparison with Switchboard repairs (in progress; Switchboard data stays local) |

## Pipeline

| step | script | input → output |
|---|---|---|
| 1. select documents | `select_docs.py` | PMB train gold → `samples/*.tsv` (one row per document and type, with every eligible site) |
| 2. generate | `run_llm.py` (on the cluster: `hpc/run.sh` + a `.sub` file) | sample + prompt version → `results/<model>.<run>.tsv`, raw replies, `run_meta` |
| 3. check | `analyse_run.py` | run TSV → `.checked.tsv` (all checks, `ambiguous` label) |
| 3b. check the given site | `check_temporal_extra.py`, `check_place_r5.py` | runs that fix the site in the prompt → `.checked.pin.tsv` |
| 4. audit / select | `audit.py` (`sample`, `serve`, `stats`) | passing items → `audit/<name>/labels.jsonl` |
| 5. fix surface | `dataset/apply_fixes.py` | all passing items → `dataset/pool_fixed.tsv` |
| 6. assemble | `dataset/assemble.py` | fixed pool + edge-label selection → `dataset/train_rsr.base.tsv`, manifest |

A run on the cluster, end to end:

    python3 select_docs.py --quota forwarding=185 --per-type 0 --seed 13 \
        --exclude samples/*.tsv --out samples/sample_x.tsv
    # copy samples/ and prompts/ to the cluster, submit hpc/run_x.sub (see hpc/README.md)
    python3 analyse_run.py results/meta-models_Muse-Glimmer-30B.x.tsv \
        --out results/meta-models_Muse-Glimmer-30B.x.checked.tsv
    python3 audit.py sample results/meta-models_Muse-Glimmer-30B.x.checked.tsv --out audit/x
    python3 audit.py serve audit/x            # http://127.0.0.1:8765/

## Runs in the dataset

All with Muse-Glimmer-30B, reasoning high, temperature 1.0, top-p 0.95,
top-k 64, seed 13 (settings in each `run_meta`).

| run | prompt | sample | documents | items in the dataset |
|---|---|---|---:|---:|
| `r2` | base | `pinned_sample_r2.tsv` | 350 | 194 |
| `train_full` | base | `sample_train_full.tsv` | 5,993 | 2,376 |
| `train_forwarding_extra` | base | `sample_train_forwarding_extra.tsv` | 185 | 174 |
| `train_temporal_extra` | temporal_r2 | `sample_train_temporal_extra.tsv` | 297 | 256 |
| `r3_edge` | edge_r3 | `sample_r3_edge.tsv` | 50 | 14 |
| `train_edge` | edge_r3 | `sample_train_edge.tsv` | 403 | 112 |
| `train_place` | edge_r4 | `sample_train_place.tsv` | 627 | 71 |
| `place_r5` | place_r5 | `sample_place_r5.tsv` | 458 | 279 |
| `place_r5_rerun` | place_r5 | `sample_place_r5_rerun.tsv` | 50 | 19 |

Earlier runs (Opus 5.5 pilot, Muse round 1, smoke test) contribute nothing
and are in `archive/repair_data_gen/rsr_generation/`.

## The generation call

- **Input:** `prompts/<version>/item_template.md` filled with the gold
  sentence, gold SBN and repair type (and, for `temporal_r2` and
  `place_r5`, the site to repair). The document id is not shown.
- **Output:** one JSON object (`task.md`): the site, the reparandum, the
  sentence as `nl_left` and `nl_right`, the repaired SBN, the devices used,
  or `no_site`. The harness joins the sentence with ", " and records
  `repair_onset`.
- **Tool:** `wn_lookup.py`, WordNet 3.0 only. `senses <lemma> <pos>` lists
  senses with definitions, `check <lemma.pos.NN>` says whether a concept
  exists. Names are rebuilt PMB-style from the lemma, never from
  `synset.name()` (1,654 of 5,932 gold names differ). No hypernyms or
  synonyms are shown.
- **One attempt.** No retry, no self-check. `no_site` is logged, not retried.
- **Excluded documents:** the 28 gold documents used as prompt examples
  (`select_docs.EXAMPLE_DOCS`), and any whose gold SBN the reader cannot
  read correctly (`select_docs.gold_is_sound`).

Two harnesses, both in `run_llm.py`:

- `--backend openai` (production): an agent loop against a vLLM server. One
  tool, which runs `wn_lookup.py` without a shell and refuses anything else;
  at most 20 turns per item.
- `--backend claude` (Opus pilot): headless Claude Code in an empty temp
  directory, locked down to the lookup. `--setting-sources project` is
  required: without it the user's read permissions let the model read the
  repository (the first dry run did).

## Checks

`analyse_run.py` writes one column per check; `audit.auto_pass` requires all
of them. What each rejects is listed in `dataset/README.md`. The checks live
in:

| module | columns |
|---|---|
| `../validate_with_pmb_parser.py` | `parse`, `dag`, `index`, `magnitude`, `edges`, `boxes` |
| `wn_lookup.py` | `wn_ok` |
| `check_untouched.py` | `untouched` |
| `check_withdraw.py` | `withdraw` |
| `check_repair_part.py` | `repair_part` |
| `analyse_run.py` | `wn_relation`, `site_ok`, `place_inside`, `ambiguous`, `oracle` (content word against `../repair_transform.py`; reported, not required) |

`check_manual_coverage.py` checks the prompt, not the output: every kind of
token in train gold must be documented in `prompts/base/sbn_manual.md`.

## Audits

`audit.py serve` shows one item at a time, blind: sentence, gold sentence and
SBN with indices resolved, no type and no check result. Each click appends a
line to `labels.jsonl`; the last line per item counts. `--questions natural`
shows the naturalness question only (used for edge-label selection).
