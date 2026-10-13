# Prompt versions

Each version is a directory of four files. The system prompt is
`sbn_manual.md` + `repair_notation.md` + `task.md`, in that order; each
request is `item_template.md` filled in for one document. A change makes a
new version; no version is edited after a run has used it. `run_llm.py
--docs` picks the version and `run_meta` records the sha256 of the prompt it
built. All five hashes were recomputed on 2026-10-13 and match their runs.

| version | was | sha256 | runs | what changed |
|---|---|---|---|---|
| `base` | `prompt_docs/` | `6b17847a…` | r2, train_full, train_forwarding_extra | the version validated in round 2 |
| `edge_r3` | `prompt_docs_edge_r3/` | `1886b812…` | r3_edge, train_edge | §4.4 (edge label) rewritten for role swaps, two negative examples |
| `edge_r4` | `prompt_docs_edge_r4/` | `24c14008…` | train_place | §4.4 adds place repairs (spatial operators), examples S–U, three negatives |
| `place_r5` | `prompt_docs_place_r5/` | `cf0cb470…` | place_r5, place_r5_rerun | the item gives the site; §4.4 covers places only; example V |
| `temporal_r2` | `prompt_docs_temporal_r2/` | `8d49b9ac…` | train_temporal_extra | the item gives the site and the reparandum already used |

`sbn_manual.md` and `repair_notation.md` are rewritten copies of
`SBN_Annotation_Working_Manual.md` and
`documentation/knowledge_base/repair_sbn_notation.qmd`. The originals are
the authored record and are not edited for prompt purposes. In the copies:

- inventories (roles, operators, constants) are those in PMB 5.1.0 train
  gold that the SBN reader accepts, with gold frequencies; inversion is
  stated as a rule (every index role has an inverse);
- everything is on one line, without `%` comments, as the output must be;
- every example, positive and negative, was run through the SBN reader;
- `check_manual_coverage.py` confirms that every kind of gold token is
  documented.
