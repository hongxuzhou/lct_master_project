# Project map

MA thesis: parsing replacement self-repair (RSR) into SBN with a
`CORRECTION`/`CONJUNCTION` extension; data built from PMB 5.1.0.

| path | what | start with |
|---|---|---|
| `repair_data_gen/` | the RSR training data and the pipeline that made it | `repair_data_gen/README.md` |
| `repair_data_gen/rsr_generation/dataset/` | frozen train set (`train_rsr.base.tsv`, 3,489 items) | its `README.md` (data card) |
| `repair_data_gen/hard_negatives/` | 66 train documents held out of all training data | its `README.md` |
| `repair_data_gen/DECISIONS.md` | every data decision with its reason | |
| `documentation/model_experiment_pipeline.md` | experiment design (Chinese): conditions, training, test, scoring | §5 for data |
| `documentation/knowledge_base/repair_sbn_notation.qmd` | the notation (authored record) | |
| `SBN_Annotation_Working_Manual.md` | base SBN manual (authored record) | |
| `evaluation/` | scoring code | `evaluation/README.md` |
| `colloquium_prep/` | pilot study and its data | |
| `archive/` | superseded code, data and docs (rule-based generator, early runs) | `archive/repair_data_gen/README.md` |
| `data/pmb-5.1.0/` | PMB (not in git) | |

## Rules

- **Switchboard / NXT data stays local.** Anything derived from it lives in
  `repair_data_gen/rsr_generation/natural_rsr_check/swbd/` (gitignored). Never
  commit, push, upload or send it anywhere.
- **Hard-negative hold-out is code, not a promise.** Every script that builds
  training data from PMB train gold (RSR or fluent) leaves out
  `repair_data_gen/hard_negatives/holdout.load()`. A data decision is done
  only when a script enforces it.
- **Do not edit run outputs or labels**: `rsr_generation/results/*.tsv`,
  `audit/*/labels.jsonl`. Derived files are rebuilt by scripts.
- **Prompt versions are immutable** once a run has used them; a change makes
  a new `prompts/<version>/`. Their hashes are listed in `prompts/README.md`.
- **The authored originals** (`SBN_Annotation_Working_Manual.md`,
  `repair_sbn_notation.qmd`) are not edited for prompt purposes.
- **After moving or changing anything the dataset depends on**, run
  `python3 dataset/apply_fixes.py && python3 dataset/assemble.py` in
  `repair_data_gen/rsr_generation/`; the sha256 must match the data card.
- Static docs are in English, short and direct; the working log goes to the
  user's Notion Research Log, not into the repo.
