---
license: cc-by-nc-sa-4.0
language:
- en
task_categories:
- text-generation
pretty_name: Self-Repair SBN Parsing Pilot Data
tags:
- PMB
- SBN
- DRS
- semantic-parsing
- meaning-representation
- speech-repair
- disfluency
size_categories:
- n<1K
dataset_info:
  features:
  - name: id
    dtype: large_string
  - name: nl
    dtype: large_string
  - name: mr
    dtype: large_string
  - name: repair_head_nl
    dtype: large_string
  - name: repair_mid_nl
    dtype: large_string
  - name: repair_tail_nl
    dtype: large_string
  - name: repair_head_interrug_nl
    dtype: large_string
  - name: repair_mid_interrug_nl
    dtype: large_string
  - name: repair_tail_interrug_nl
    dtype: large_string
  splits:
  - name: train
    num_bytes: 598081
    num_examples: 836
  download_size: 327040
  dataset_size: 598081
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---

# Self-Repair SBN Parsing Pilot Data

A small, static pilot set for studying how a semantic parser (NL → SBN) behaves when
the input natural language carries **self-repair disfluencies**. Each row is one gold
sentence from the Parallel Meaning Bank (PMB) together with synthetically generated
repair variants. The dataset holds **inputs only** — parser predictions are stored
elsewhere so this set stays fixed across modelling stages.

- **Source:** PMB 5.1.0 (gold), English single-sentence items.
- **Size:** 836 rows.
- **Task:** sequence-to-sequence semantic parsing (NL → SBN), with disfluent inputs.

## Experimental design

Each source sentence yields **7 parser-input conditions**:

| condition | column | description |
|---|---|---|
| gold (reference) | `nl` | original fluent sentence |
| head repair | `repair_head_nl` | reparandum+repair at the sentence start |
| mid repair | `repair_mid_nl` | reparandum+repair mid-sentence |
| tail repair | `repair_tail_nl` | reparandum+repair near the end |
| head + interregnum | `repair_head_interrug_nl` | head repair with editing phrase inserted |
| mid + interregnum | `repair_mid_interrug_nl` | mid repair with editing phrase inserted |
| tail + interregnum | `repair_tail_interrug_nl` | tail repair with editing phrase inserted |

`mr` is the **gold SBN** for `nl`, used as the reference meaning representation for
all comparisons. The dataset deliberately contains **no parser-output columns**: at
parse time, melt the wide table to long `(id, condition, nl_input)`, run the parser,
and store predictions separately keyed by `(id, condition)`.

## Fields

| field | type | content |
|---|---|---|
| `id` | string | PMB item id, e.g. `p17/d0758` |
| `nl` | string | gold natural language (fluent) |
| `mr` | string | gold SBN (reference meaning representation) |
| `repair_head_nl` / `repair_mid_nl` / `repair_tail_nl` | string | disfluent input, repair at head/mid/tail |
| `repair_head_interrug_nl` / `repair_mid_interrug_nl` / `repair_tail_interrug_nl` | string | same, with an interregnum editing phrase |

## How it was built

- **Gold items** drawn from PMB 5.1.0, restricted to coherent single-sentence
  examples (29 multi-sentence / direct-speech items were removed so that
  position-based repair insertion is methodologically stable).
- **Repair variants** generated on HPC via prompt engineering with Qwen3-27B. The
  head and mid variants were manually checked; the tail variant was not.
- **Interregnum variants** inserted by rule on top of each repair, placing the
  editing phrase between reparandum and repair.

## Known limitations

This is a **pilot** set. Please read these before using it for evaluation:

1. **Single editing phrase.** The interregnum is always `"I mean,"`. No lexical or
   prosodic variation is modelled.
2. **Tail repairs are not manually validated.** Only head and mid repair variants
   were human-checked; tail repairs come straight from the model.
3. **Three items lack a head repair.** For `p90/d2399`, `p67/d2005`, `p16/d1602`
   the model did not produce a head-repair variant, so `repair_head_nl` equals `nl`
   and the head condition is degenerate for those rows.
4. **A few interregnum insertions are skipped.** Where a repair has no comma-marked
   reparandum boundary, the rule-based inserter declines rather than mis-place the
   phrase (~2 head, 1 mid rows); for those, the `*_interrug_nl` value equals the
   plain repair. (An earlier version silently collapsed ~23% of head rows this way;
   the current build reduces it to <1%.)

## Loading

```python
from datasets import load_dataset
ds = load_dataset("Shrikes/self_repair_parsing_pilot_data")  # split: "train"
```

## License

Derived from the Parallel Meaning Bank; released under **CC BY-NC-SA 4.0** to match.
Underlying sentences originate from Tatoeba. Non-commercial research use.