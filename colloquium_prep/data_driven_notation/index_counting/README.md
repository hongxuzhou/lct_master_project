# SBN index counting domain: LINEAR vs REGISTER

Settles which counting domain SBN's `Role ±n` indices use, on PMB 5.1.0 `en/train/gold.sbn`.

## The question

Two readings of Bos (2021) were in play:

- **LINEAR** — `-n` counts back `n` concepts in the raw token sequence; boxes are irrelevant.
- **REGISTER** — every box archives/loads its own index register, so concepts sitting in a box
  that is not on the current box's load path do **not** occupy a countable position.

They differ by one or more positions after any separator that returns to an outer box, so the
choice governs every synthesised `CORRECTION` + `CONJUNCTION` example.

## Verdict: LINEAR

Separators do not reset, restrict, or partition the counting domain. Concepts in already-closed,
non-ancestor sibling boxes still occupy positions and still count.

| Evidence | LINEAR | REGISTER |
| --- | --- | --- |
| Cases where the models disagree (n=92) | correct 92/92 | correct 0/92 |
| Negative indices that fail to resolve (n=19873) | 1 (0.01%) | 77 (0.39%) |

The single LINEAR failure is itself a gold annotation error: `p82/d3242` "We must keep a diary
every day.", where `keep.v.08 Agent -2` should be `-1`.

The reference implementation agrees: `data/pmb-5.1.0/src/sbn/sbn_smatch.py:253-256` computes
`target_idx = _active_synset_id + idx`, and that synset counter is a single global counter that
box creation never touches.

### Cleanest single witness — `p00/d2473` "These cups are all broken."

```
cup.n.01 NEGATION <1 NEGATION <1 time.n.08 EQU now CONJUNCTION <2 entity.n.01 SubOf -2 CONJUNCTION <2 broken.a.01 Time -2 AttributeOf -1
```

`broken.a.01 Time -2` is a type check: a `Time` role must land on a time concept. Only LINEAR
delivers `time.n.08`, and it gets there by counting *through* a sibling `CONJUNCTION` box.

## Secondary findings

**Angle brackets index BOXES, hyphen/plus index CONCEPTS — two separate notations.**
`Proposition >1` / `<1` are box pointers resolved against the box counter
(`sbn_smatch.py:300-302`). Gold has 172 `Proposition >1`, 12 `<1`, 1 `<2`. This, not any
counting-domain subtlety, is why such targets appear to "sit several positions forward" while
the index stays at 1.

**Positive indices do cross box boundaries in gold.** 80 edges (70 documents, 13 roles,
magnitudes +1..+5) carry a positive index whose filler lands in a sibling box — and in 100% of
them that box was opened by `CONJUNCTION`, never by any other separator. So "a positive index
may not leave its box" is not an absolute of gold practice. Caveat: all 80 arise from the
universal-quantification template, not from repair, so this licenses the *notation shape*
rather than any particular repair reading.

**Two bugs in the vendored toolchain:**

- `sbn_spec.py:217` — `INDEX_PATTERN = r"((-|\+|\<|\>)\d)"` matches a single digit, so `-11`
  and `-10` are silently truncated to `-1` and `+11` to `+1`. No error raised; the edge just
  connects to the wrong concept. 8 gold edges affected. Fix: `\d+`. Note this shifts Smatch
  baselines.
- `CauserOf` occurs in gold but is missing from `SBNSpec.ROLES`, so a tokeniser built strictly
  from the spec rejects those records.

## Reproducing

Scripts take no arguments and locate the corpus relative to the repo root.

```bash
cd script
python3 probe_index_domain.py   # replay gold under both models -> output/divergent_cases.json
python3 show_divergent.py       # readable dump -> output/all_92_divergent_cases.txt
python3 aggregate_test.py       # corpus-level resolvability + type-agreement comparison
python3 pos_test.py             # positive indices that cross box boundaries
```

`output/all_92_divergent_cases.txt` prints, for each divergent case, the sentence, the SBN, the
box tree, the numbered concept list, and what each model claims the index points at — so the
targets can be checked by eye.
