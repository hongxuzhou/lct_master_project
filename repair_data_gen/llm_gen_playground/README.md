# What is it

Alongside the finished rule-based pipeline for RSR NL-SBN data, this
playground asks whether an LLM can generate the same kind of data reliably.
The model is given the same formal constraints and lexicon-level tool access,
and has to do the construction itself: pick the repair site, pick the
reparandum, insert it, build the `CORRECTION`/`CONJUNCTION` structure, and
reindex.

The LLM side covers every single-sentence repair type in `DESIGN.md` §2–3,
not only the content-word substitution the rule pipeline implements today.
The types the rule pipeline cannot yet produce (constant, tense, edge label)
or needs a human for (forwarding) are where an LLM could add the most, so
they are tested from the start.

The core question is the legality of the meaning representation the model
writes. Lexical choice is left free, and WordNet membership is enforced by a
tool, because an off-inventory node makes a sample unusable and says nothing
about the compositional edit we want to study.

# Approach

Start with a minimal version, look at what it produces, and add only what the
first results show is needed. The current scope is the MVP below. Ideas from
earlier drafts that are deferred, with the reason, are listed in the last
section.

# MVP

## 1. Generate

- **Input per item**: one gold `(nl, sbn)` pair from
  `data/pmb-5.1.0/split/en/train/gold.sbn`, the **repair type** to produce
  (content word, constant, tense, edge label, forwarding; see
  `prompt_docs/repair_notation.md` §4), and the agent-facing reference
  material (§3). The type is given, not chosen by the model, so that every
  type is covered and each can be scored on its own.
- **No interregnum** (decided 2026-10-02, before the Muse Glimmer run). The
  model writes the base repair only ("The gnat, the fly buzzes."), as
  `DESIGN.md` steps 3 and 5 order it for the rule pipeline. An editing
  expression does not change the SBN, so adding one is a fixed string edit
  at the repair onset, which a script does better than a model; and only a
  scripted insertion gives the RP's matched test pairs (same repair, with
  and without an expression). Insertion is deferred until the quality of
  the base data has been assessed. The `interregnum` column of
  `pinned_sample.tsv` (assigned for the Opus trial) is not used; results
  carry `none`. The prompt documents' examples keep their interregna, since
  the notation is unchanged; `task.md` tells the model to leave them out,
  and `check_repair_part.py` flags any it writes.
- **Repair onset.** The model returns the sentence in two parts, `nl_left`
  (to the end of the reparandum) and `nl_right` (from the start of the
  repair). The harness joins them with ", " into `nl_repair` and records
  `repair_onset`, the character offset where the repair starts: the
  interruption point, and the place an editing expression will go. The
  split is asked for instead of an offset because models count characters
  badly.
- **Tool**: `wn_lookup.py`, a WordNet 3.0 dictionary and the only tool.
  `senses <lemma> <pos>` lists every sense with its definition; `check
  <lemma.pos.NN>` says whether a concept exists. Points that matter:
  - PMB names a concept by lemma and the lemma's sense number, which is not
    WordNet's own synset name (`time.n.08` is WordNet's
    `fourth_dimension.n.01`; 1654 of 5932 gold names differ). The tool
    rebuilds every name from the queried lemma, never from `synset.name()`.
  - Satellite adjectives (`s`) are listed and accepted as `a`, as in PMB.
  - `check` is exact: `chased.v.01` is rejected with a pointer to
    `chase.v.01`.
  - SBN placeholder concepts (`time.n.08`, `entity.n.01`, …) get a note on
    how SBN uses them; the 8 gold names missing from WordNet (`event.v.01`
    and 7 others) are reported as PMB-only.
  - Only definitions and examples are shown. No hypernyms, synonyms or
    sister terms, since those would point to where a reparandum could be
    found.
  - No VerbNet/FrameNet pool. The pool encodes the judgement of what makes a
    believable slip, which is what the model is tested on; and SBN legality
    depends on WordNet alone. The pool's near-synonym exclusion is run
    offline as a diagnostic (Deferred).
  - Tested against train gold: `check` accepts all 5924 gold names found in
    WordNet and rejects the 8 that are not; for every gold name, `senses`
    lists it with the same definition.
- **No revise loop and no self-check**: the model gets one attempt per item.
  Its first output is what gets evaluated, so the error distribution stays
  visible.
- **Output per item**: one JSON object (format in `prompt_docs/task.md`):
  the repair site, the reparandum, the repaired sentence, the repaired SBN,
  and the model's own report of the devices it used. A model may answer
  `no_site` when the sentence offers no site of the requested type; that is
  logged, not retried.

## 2. Store

One TSV, using the column names of `generate_repairs.py` so that the same
downstream code reads both pipelines and the two can be joined on `doc_id`:

| column | source |
|---|---|
| `doc_id`, `nl_clean`, `sbn_clean` | gold input |
| `repair_type` | input: `content_word` / `constant_name` / `constant_number` / `constant_temporal` / `tense` / `edge_label` / `forwarding` |
| `interregnum` | `none` in this round (base repairs only) |
| `status`, `reason` | model: `ok` or `no_site` |
| `site_index` | model: position of the repaired concept in the gold SBN (what `build_repair` takes as `i`) |
| `repair_synset` | model's `site_synset` |
| `reparandum_value` | model: the new constant, tense operator or role, where the synset does not change |
| `nl_repair`, `sbn_repair` | model (`nl_repair` joined from `nl_left` and `nl_right`) |
| `repair_onset` | harness: `len(nl_left) + 2`, the character offset of the repair in `nl_repair` |
| `reparandum_synset`, `reparandum_surface` | model |
| `repair_pos` | derived offline from `repair_synset` |
| `devices` | model self-report, `\|`-separated: `d1_drop_role`, `d2_invert_role`, `d3_dummy_equ`, `d4_movement`, `d5_anchor_dummy` |
| `strategy` | derived offline from `devices` (`d5` → `anchor`, `d4` → `hoist`, else `inplace`), not asked of the model: the three names are internal to `repair_transform.py` and do not appear in the prompt documents |
| `model` | exact model id used for the run |
| `raw_output` | the model's full response, for items that fail to parse into columns |

For constant and tense repairs, `reparandum_synset` equals `repair_synset`
(the value or the tense differs, not the concept). For forwarding, the two
columns hold the subject concepts.

The device column is self-reported. It may differ from what the SBN actually
does, and any mismatch seen in the first round is itself a finding.

## 3. Reference material given to the model

The system prompt is three files from `prompt_docs/`, in this order:
`sbn_manual.md`, `repair_notation.md`, `task.md` (instructions and output
format, placed last so they sit closest to the answer). Each request is
`item_template.md` filled in for one document; the document id is not shown
to the model.

The two reference documents are rewritten copies:

- `sbn_manual.md`, from `SBN_Annotation_Working_Manual.md` (base SBN);
- `repair_notation.md`, from `documentation/knowledge_base/repair_sbn_notation.qmd`
  (the `CORRECTION`/`CONJUNCTION` extension).

These are the task specification, the same material a human annotator works
from. **Do not edit the originals in place.** The originals are the authored
project record.

How the copies were made:

- Inventories (roles, operators, separators, constants) are what occurs in
  PMB 5.1.0 English train gold **and** is accepted by the SBN reader, with
  gold frequencies. One exception, by design: inversion is stated as a rule
  (Bos 2021), not listed from gold. Every role that can take an index has an
  inverse `ROf`; symmetric operators (`EQU`, `NEQ`, `APX`, `SY1`) can be
  moved to the other concept, and ordering operators replaced by their
  converse (`TPR`/`TSU`, `TIN`/`TCT`, `LES`/`MOR`, `TOP`/`BOT`, `SZN`/`SZP`,
  `SXP`/`SXN`). Operator inversion needs no new tokens.
- Everything is single-line, no `%` comments, matching gold and the
  expected output.
- Each document has positive and negative examples. Every example was run
  through the SBN reader. Content-word examples A–K in `repair_notation.md`
  equal `build_repair`'s output for the same site and reparandum, except J
  (operator inversion): the rule pipeline cannot invert operators yet and
  uses ⑤ there. Examples L–Q (constant, tense, edge label, forwarding) were
  built by hand, since the rule pipeline has no operator for them; each was
  checked edge by edge.
- Senses in examples A, B, G and I (taken from the original qmd, whose
  senses were placeholders) were chosen with `wn_lookup.py senses` and
  checked against the senses PMB annotators used for the same lemmas in
  gold; all 14 lemmas agree with gold's majority sense. Two fixes followed,
  both in G: `order.v.01` ("give instructions") became `order.v.02` ("make
  a request for something"), and `cherry_pie.n.01`, which is not in
  WordNet, became `apple_pie.n.01`.

Gold documents used as examples are excluded from the sample, so the model
never repairs a sentence it has seen worked through: `p62/d1762`,
`p37/d1861`, `p35/d3246`, `p81/d1985`, `p70/d3477`, `p18/d2190`,
`p98/d0867`, `p01/d0850`, `p49/d1783`, `p30/d2997`, `p48/d3446`,
`p00/d2270`, `p37/d0872`, `p96/d0785`, `p71/d2819`, `p30/d2724`,
`p72/d2409`, `p80/d1457`, `p68/d3216`, `p74/d1927`, `p04/d3042`,
`p76/d2001`, `p47/d1417`, `p14/d2976`, and `p18/d1394` (example R, added
after the first Muse Glimmer round).

Seven further gold documents are excluded because their gold SBN is itself
broken as the SBN reader reads it: `p17/d2285` and `p82/d3242` have an index
pointing outside the sequence (`keep.v.08 Agent -2` at position 1), and
`p07/d1959`, `p23/d1469`, `p04/d2024`, `p42/d1891`, `p69/d1719` have a
two-digit index, which the reader truncates (two of them become cyclic).

## 4. Evaluate

- **Structural validity**: `validate_with_pmb_parser.py --tsv FILE
  [--out FILE]` runs `check()` on every row (rows with an empty SBN, i.e.
  `no_site`, are skipped) and writes one result row per input row. Both
  pipelines are scored by the same code. Checks:
  - `parse`: the SBN reader accepts it;
  - `dag`: the graph is acyclic;
  - `index`: every index lands on a concept. The reader does not reject an
    out-of-range index; it turns `Theme +3` into an edge to a constant node
    "+3", so the old edge-count check never caught it;
  - `magnitude`: no index of 10 or more (silently truncated to one digit);
  - `edges`: distinct edges written equal edges in the graph (counted by
    concept pair, since two roles between one pair collapse into one edge,
    as in gold `p12/d3490`);
  - `boxes`: `CORRECTION` and `CONJUNCTION` are siblings.
  Tested on all 17 examples and 11 negative cases of
  `prompt_docs/repair_notation.md` (each failure caught by the intended
  check; the two negatives that are wrong but structurally legal pass, as
  expected) and on 374 rule-pipeline rows, 105 of them with `%` interregnum
  comments.
- **Per type.** Every figure is reported per `repair_type`.
- **Device distribution** (content word only): counts of `devices`, set
  against the rule-based pipeline's distribution on the same documents. The
  other types have no rule-pipeline counterpart yet, so for them only the
  structural checks apply.

## 5. Harness

`run_llm.py run --model MODEL` sends one headless Claude Code call
(`claude -p`) per row of `pinned_sample.tsv` and appends a row to
`results/<model>.tsv`; the full reply of every call is kept in
`results/raw/<model>/`. Rows already done are skipped, so a run can be
resumed. `run_llm.py probe` sends a free-form prompt in the same environment.

Each call runs in its own empty temporary directory outside the repository,
holding only a copy of `wn_lookup.py`, with:

| flag | removes |
|---|---|
| `--safe-mode` | CLAUDE.md, skills, plugins, hooks, MCP servers, custom agents, output styles (login and billing unchanged, so the subscription works; `--bare` would need an API key) |
| `--setting-sources project` | user settings, and with them the user's permission allow rules (see below) |
| `--system-prompt` | the default Claude Code prompt, replaced by the three prompt documents |
| `--tools Bash` + `--allowedTools "Bash(python3 wn_lookup.py *)"` + `--permission-mode dontAsk` | every tool and command except the lookup; others are refused, not asked about |
| `--strict-mcp-config`, `--disable-slash-commands` | MCP servers and skills, a second time |
| `--no-session-persistence` | session records |

Checked by dry run (Haiku 4.5, before any real run):
- the session's `init` event lists only `Bash`; no MCP servers, skills or
  slash commands; the work dir is a system temp dir;
- asked what it knows, the model reports only the prompt documents plus
  what Claude Code injects regardless: the user's e-mail, the working
  directory, the date, and commit-attribution guidance (no project content);
- `cat` of a repository file and `ls ~/.claude/projects` are refused
  (`permission_denials`); `wn_lookup.py check` and `senses` run;
- no directory is left under `~/.claude/projects/`, so nothing is
  remembered between calls.

**The first dry run leaked.** Without `--setting-sources project`, `cat`
read `repair_transform.py` and `ls` listed the memory directory: `--safe-mode`
keeps permission settings, and the user's settings allow
`Read(//Users/<user>/**)`, under which read-only Bash commands run unasked.

One full item (`p04/d2647`, content word, Opus 5.5): one lookup, valid JSON,
all structural checks pass, and the SBN equals `build_repair`'s output for
the same site and reparandum. About 22k prompt tokens read from cache and
1.3k output tokens per item.

Record the model id and the harness for every run; a result supports a
claim about that model with that harness, not about LLMs in general.

### Open models

`run_llm.py run --backend openai --base-url URL --model MODEL` runs the same
items against an OpenAI-compatible server (vLLM on the LST cluster; job files
in `hpc/`). Claude Code is replaced by a small agent loop in `run_llm.py`:

- the same system prompt (the three documents, unchanged) and item prompt;
- one tool, `bash`, which runs `python3 wn_lookup.py ...` without a shell and
  refuses any other command, as the Claude Code allow rule did. There is
  nothing else to lock down, since the model has no other tool;
- at most 20 model turns per item; the reply is stored in Claude Code's JSON
  shape, so `result_row()` and the scoring scripts read both.

Differences from the Claude Code harness, to keep in mind when comparing:
the tool description is short (Claude Code's Bash description is long), and
nothing is injected besides the prompt documents (no e-mail, date or
attribution text). For Muse Glimmer the chat template adds one line,
`Reasoning strength: <level>`, set with `--reasoning`. Sampling follows the
model card (temperature 1.0, top-p 0.95, top-k 64, seed 13). Each run writes
`run_meta_<time>.json` with these settings and the server version next to
the raw replies.

### Prompt documents changed after the first Muse Glimmer round (2026-10-02)

Reading the smoke-run reasoning showed the model guessing at `?` in
`country.n.02 Name ?`. The rewritten `sbn_manual.md` had dropped it from the
constants table (the original manual lists it, with a looser gloss), and no
example had a question. `check_manual_coverage.py` classifies every token of
train gold and reports any kind the manual does not document; it found `?`
(437 occurrences, the slot a wh-question asks about) and two rare constant
forms, scores `3-0` (13) and fractions `3/4` (5). All three are read by the
SBN reader. Added: three rows to §2.2 and a §4.6 on questions (including that
yes/no questions have no `?` and embedded wh-clauses do). The script now
reports full coverage, and removing the `?` row makes it fail again.

`repair_notation.md` gained a rule, decided here: a reparandum in a question
slot carries the same `?` as the repair, since the question word precedes
both (§4.1, example R, an error row in §6). Example R was checked like the
others (reader, untouched edges, repair part). `select_docs.py` no longer
counts `Name ?` as a `constant_name` site; no pinned row was affected.

The first Muse Glimmer round (70 rows) ran with the earlier documents; 2 of
its rows have `?` in gold (`p04/d2647`, `p33/d1381`). Results from later
rounds are not directly comparable on those rows.

`--tool-mode text` extracts Muse Glimmer's ATEM tool calls
(`<atem:invoke ...>`) from the reply text, for a vLLM without a matching
tool-call parser. Tested against a mock server in both modes: the lookup
runs, `cat` is refused, and `check x; ls` reaches `wn_lookup.py` as plain
arguments. Not yet run against the real model.

## 6. First round

`select_docs.py` writes `pinned_sample.tsv`: 10 gold documents per repair
type, 70 in all, with the interregnum fixed per row (seed 13). The aim is to
see what the errors look like before scaling up, so statistical power is
not the goal yet. The content-word documents are also run through
`generate_repairs.py` for comparison.

A document is eligible for a type when its gold SBN offers a site of that
type; every eligible site is written to `eligible_sites`, so a `no_site`
answer can be checked against what was there. Eligible documents in the
pool of 9521:

| type | site | eligible |
|---|---|---|
| `content_word` | a concept that is not a placeholder | 9501 |
| `constant_name` | a `Name` constant (not `Name ?`, since 2026-10-02: 3154) | 3254 |
| `constant_number` | a plain number (`Quantity 3`, `EQU 100`), not a year or time | 799 |
| `constant_temporal` | `ClockTime`, `DayOfWeek`, `DayOfMonth`, `MonthOfYear`, `YearOfCentury` | 311 |
| `tense` | a verb with `Time` to an adjacent `time.n.08 TPR/EQU/TSU now` in the same box | 7290 |
| `edge_label` | a verb with an index-valued `Destination`/`Source`/`Location` | 1871 |
| `forwarding` | noun subject, its verb, and the verb's tense `time.n.08`, adjacent, as in example Q; since 2026-10-02 not a quantifier subject built with `NEGATION` (nobody, everyone) and not a verb the sentence elides ("so do I", "Neither Tom nor Mary goes"): 4522. Pronoun subjects stay | 4574 |

Selection: no length filter (lengths follow gold, 2–17 words in this
sample, so a length preference of the model shows up in `no_site` and error
rates); each document used for one type only; constant types take at most
one document per constant value, so the name cell is not mostly "Tom".

## 7. To build

1. ~~The WordNet lookup.~~ Done: `wn_lookup.py`.
2. ~~The reference copies (§3).~~ Done.
3. ~~The prompt: task instructions and the output format.~~ Done:
   `prompt_docs/task.md`, `prompt_docs/item_template.md`.
4. ~~The refactor of `validate_with_pmb_parser.py` (§4).~~ Done.
5. ~~The per-type document selection (§6).~~ Done: `select_docs.py`,
   `pinned_sample.tsv`.

6. ~~The harness.~~ Done: `run_llm.py`; dry run passed (§5).

# First round: results (Opus 5.5, 2026-10-01)

`results/claude-opus-5-5.tsv` (70 rows), `results/claude-opus-5-5.checked.tsv`
(with every check), raw replies in `results/raw/claude-opus-5-5/`. Scripts:
`analyse_run.py` (structural checks, WordNet, oracle, per-type counts),
`check_untouched.py` (every gold edge away from the repair site survives,
after normalising role and operator inversion), `check_repair_part.py`
(R5, reparandum differs from repair, interregnum as assigned, gold words
kept in order; for runs with `repair_onset`, also a single ", " at the
onset and `reparandum_surface` ending the text before it). Each of the two added checks was first shown to catch
deliberately broken inputs.

| type | n | answered | no_site | structural checks | concepts in WordNet | untouched gold edges kept |
|---|---|---|---|---|---|---|
| content word | 10 | 10 | 0 | 10 | 10 | 10 |
| constant: name | 10 | 10 | 0 | 10 | 10 | 10 |
| constant: number | 10 | 10 | 0 | 10 | 10 | 10 |
| constant: temporal | 10 | 10 | 0 | 10 | 10 | 10 |
| tense | 10 | 10 | 0 | 10 | 10 | 10 |
| edge label | 10 | 10 | 0 | 10 | 10 | 10 |
| forwarding | 10 | 8 | 2 | 8 | 8 | 8 |
| **all** | 70 | 68 | 2 | 68 | 68 | 68 |

Every reply parsed. No answered row failed any automatic check; the only
flag from `check_repair_part.py` is a false alarm ("Tom's" written out as
"Tom is").

**Oracle (content word).** 9 of 10 equal `build_repair`'s output for the
same site and reparandum. The tenth (`p10/d0798`) differs because the model
inverted a symmetric operator (`EQU`) as `prompt_docs/sbn_manual.md` §2.4
allows, where `build_repair` cannot and points over the `CORRECTION` box,
leaving the reparandum unlinked to "a lot of". The model's version follows
the current specification more closely. Until the rule pipeline is
upgraded, an oracle mismatch has to be read case by case.

**no_site.** Both in forwarding, both with a pronoun subject: "I"
(`person.n.01 EQU speaker`) and "Nobody" (a `NEGATION`-built quantifier,
which §7 rules out). The second is a selection error: `select_docs.py`
accepts any noun subject, including placeholders. Whether a pronoun
subject should count for forwarding is open; the model also accepted "He"
→ "she" twice.

**One answer is wrong in the sentence but not in the SBN.** `p31/d1485`,
"Neither Tom nor Mary goes to the store": the gold SBN has a `go.v.01` in
each clause, but the sentence elides the first "goes", so the
subject-then-verb pattern exists in the SBN only. The model's SBN is
correct; its sentence ("Neither Tom nor Sue goes, Mary goes to the store")
is not. It noted the awkwardness and went on rather than answering
`no_site`. Selection should also check the surface sentence.

**Device use.** Self-reports match the structure in every row read: ⑤ for
subjects and for `quantity.n.01` numbers (correct per the toolkit, since
no edge reaches them from before the repair), ② for objects and dates, ③
on all ten edge-label repairs, ① on tense and forwarding.

**Site choice (content word).** All ten chosen sites needed only ① or ②;
of the 24 eligible sites in those ten documents, 4 would have needed ⑤ or
④. With uniform choice, picking no hard site in ten tries happens about 16%
of the time, so this is a hypothesis to test, not a finding.

**Tool use.** One WordNet lookup per content-word item (median); none for
the other types, which introduce no new concept except `entity.n.01`.

**What the automatic checks cannot see.** Whether a reparandum is a
believable slip, and whether the sentence reads naturally. Reading all 68:
most are natural; weaker ones include "I'm from, on a boat" and "Rod
Stewart, Cat Stevens is not a cat", where the name swap breaks the joke of
the next sentence.

# First round: results (Muse Glimmer 30B, 2026-10-02)

Same 70 documents, base repairs only (no interregnum, `nl_left`/`nl_right`),
prompt documents as before the `?` additions. Harness: `--backend openai`,
vLLM 0.30.0+cu129 on one H100, BF16, reasoning `high`, temperature 1.0,
top-p 0.95, top-k 64, seed 13, 8 workers, 16384 output tokens per turn
(`results/raw/meta-models_Muse-Glimmer-30B/run_meta_*.json`). Files:
`results/meta-models_Muse-Glimmer-30B.tsv`, `.checked.tsv`, raw replies.

| type | n | answered | no_site | failed | structural checks | concepts in WordNet | untouched gold edges kept |
|---|---|---|---|---|---|---|---|
| content word | 10 | 10 | 0 | 0 | 10 | 10 | 10 |
| constant: name | 10 | 10 | 0 | 0 | 10 | 10 | 10 |
| constant: number | 10 | 9 | 0 | 1 | 9 | 9 | 9 |
| constant: temporal | 10 | 9 | 0 | 1 | 9 | 9 | 9 |
| tense | 10 | 10 | 0 | 0 | 10 | 10 | 10 |
| edge label | 10 | 10 | 0 | 0 | 10 | 10 | 10 |
| forwarding | 10 | 10 | 0 | 0 | 10 | 10 | 10 |
| **all** | 70 | 68 | 0 | 2 | 68 | 68 | 68 |

**Failed.** `p83/d2955` (number) and `p31/d1973` (temporal) hit the
16384-token limit in the first turn, inside the reasoning, with no tool call
and no answer (62–66k characters of reasoning against a median of 28k for
answered items). Both circle: one over which temporal constant to change
although the sentence has `YearOfCentury 1705`, the other over where
`nl_left` should end. They count as failures of this round, not to be
retried; a production run would raise the limit and report how many items
needed it.

**Oracle (content word).** 9 of 10 equal `build_repair`. `p10/d0798` differs
for the same reason as Opus: the model moves the symmetric `EQU`, which
`build_repair` cannot.

**Rest of the sentence.** `check_repair_part.py` now also checks that the
repair part is the end of the gold sentence and the gold text before it
opens the reparandum part (possible since the split). It flags `p31/d1485`,
"Neither John goes, Tom goes to the store nor Mary goes to the store",
which repeats material the gold sentence elides; the SBN is correct. Opus
failed on the same document in a different way, so the selection error
noted above is confirmed. The other flag is `p23/d1567` ("Tom's" written
out as "Tom is"), a false alarm.

**Found by reading, not by any check.**

- *A repair inside a quantifier.* `p99/d2900`, "He appreciated, nobody
  appreciated his sacrifice". "Nobody" is built with `NEGATION`, which §7
  rules out; Opus answered `no_site` here. The `CORRECTION` box hangs off
  the negated box, so the withdrawn "he appreciated" is represented as
  negated. The SBN passes every check and means the wrong thing.
- *Synonyms as reparanda.* `rhythm.n.01` → `beat.n.03` (`p27/d1965`) is one
  WordNet synset: no change of meaning, which `task.md` rules out. Close to
  it: `apparatus` → `equipment` and `believe` → `trust` (hypernyms), `late`
  → `overdue`. Opus's ten content-word pairs include none of these (sisters
  or less related). `wn_lookup.py` shows definitions only, so a same-synset
  pair has two identical definitions, which the model did not act on.
- *Never `no_site`.* Muse answered every item, including the two Opus
  declined (`p99/d2900` "Nobody", `p29/d1811` "I"). Forwarding replaced a
  pronoun subject in 5 of 10 items ("He loves, I love…"), plus "Nobody".
- *Surface.* "This Tuesday, This Wednesday" (capital in mid-sentence);
  weaker readings as in the Opus round: "I'm from a boat, on a boat", "an
  accident from the freeway", "John, Cat Stevens is not a cat". Edge-label
  repairs use `Source` in 7 of 10.

**Site choice (content word).** As with Opus, none of the ten sites needs ④
or ⑤ (devices: ① 4, ② 4, none 2). Two models now chose no hard site on the
same ten documents, which strengthens the hypothesis in the Opus section,
though the two draws are not independent. In the smoke run, at the same
temperature, Muse chose a subject (⑤) on `p04/d2647` and `p10/d0798`, so
site choice varies between runs; per-request seeds do not make it
reproducible under batching.

**Reasoning length.** Median 28k characters of reasoning per answered item
(interquartile 24k–35k; median 7.8k output tokens), longest for tense (33k)
and shortest for content word (22k). The three smoke items (11k–17k) were
not representative. The two failures also have a slightly higher rate of
self-checking phrases ("wait", "let me recount", …: 1.3 against 1.1 per
1000 characters).

**Tool use.** Median four lookups per content-word item (Opus: one), two
for forwarding, one for tense, none for the other types.

**Decided after this round (2026-10-02).**

- *Token limit* raised to 32768 per turn (`run_llm.py --max-tokens`), with
  vLLM's `--max-model-len` at 98304 to leave room for the prompt and earlier
  turns. Each raw reply now records `turn_output_tokens`, so the next round
  can report how many items went past the old 16384.
- *Synonyms are filtered.* `analyse_run.py` adds `wn_relation` for content
  words and counts `same_synset` as a failure. `hypernym`, `hyponym` and
  `similar_to` are recorded only: correcting to a more exact or more general
  word ("apparatus, equipment") can be a natural repair, even if it is close
  to a shortcut from the generator's side. On this round: `same_synset` 1,
  `hypernym` 1, `hyponym` 1, `other` 7 (Opus: `other` 10).
- *Fixed in `analyse_run.py`:* the run's `parse` column (did the reply parse)
  shared its name with the SBN reader check `parse`, so unanswered rows were
  left out of the failure list. The reply status is now kept as `reply`. The
  counts in both result sections are unchanged.

# Round 2 (planned, 2026-10-02)

Round 1 showed what the errors look like; round 2 is to estimate how often
they occur, per type, before any larger synthesis. All seven types take
part: the RP's main corpus is content words only because the rule pipeline
had only reached that type, not because the others were ruled out.

- `pinned_sample_r2.tsv`: 50 documents per type (350), seed 13, drawn with
  `select_docs.py --exclude pinned_sample.tsv`, so none from round 1 and none
  used as a prompt example; 28 have `?` in gold. No interregnum column.
- Everything changed after round 1 applies: base repairs with
  `nl_left`/`nl_right`, `?` in the manual and the notation, the selection
  rules for `Name ?` and forwarding, 32768 tokens per turn.
- Job: `hpc/run_r2.sub` (16 workers), results in
  `results/meta-models_Muse-Glimmer-30B.r2.tsv`.
- Human audit of about 100 items, two questions each: does the SBN mean what
  the sentence says (the `p99/d2900` kind of error), and does the repair read
  naturally.

## Human audit (decided 2026-10-02)

`audit.py` with `audit_page.html`, run locally:

    python3 analyse_run.py results/meta-models_Muse-Glimmer-30B.r2.tsv \
        --out results/meta-models_Muse-Glimmer-30B.r2.checked.tsv
    python3 audit.py sample results/meta-models_Muse-Glimmer-30B.r2.checked.tsv --out audit/r2
    python3 audit.py serve audit/r2        # http://127.0.0.1:8765/
    python3 audit.py stats audit/r2        # also writes audit/r2/merged.tsv

- **Pool.** Only items that pass every automatic check (`auto_pass` in
  `audit.py`; an oracle difference is not a failure). Since 2026-10-04 this
  includes `untouched` and `repair_part`, which `analyse_run.py` now writes
  (before, the two scripts ran by hand and the pool did not see them). Items that fail a check
  are already counted as errors, so auditing them adds nothing; the audit
  measures what the checks cannot see. The 90% and 80% thresholds therefore
  apply to items that passed the checks.
- **Size.** 15 per type, 105 in all, seed 13. The overall rate is the main
  figure (95% interval about ±6 points). With 15 items per type the interval
  is about ±15 points, so a per-type rate only screens: a type is clearly
  below the threshold only if the upper bound of its interval is below it.
- **Second stage.** A type whose rate is under its threshold but whose
  interval still reaches it gets 15 more items
  (`audit.py sample … --per-type 30 --types <type>`). Items already drawn
  are kept. The second stage is decided by this rule, not by looking at the
  items.
- **Blind.** The page shows the utterance, the original sentence and the
  SBN with indices resolved, but not the repair type, the self-reported
  devices or any check result. Items are shuffled across types.
- **Storage.** Every click appends one line to `audit/<name>/labels.jsonl`
  (`audit_id`, `meaning`, `natural`, `note`, `annotator`, `ts`). The last line
  per item counts, so earlier answers stay on record. `unsure` is left out of
  the rates and reported separately.
- **Prototype.** `audit/r1_proto/` holds the 67 round-1 items that pass the
  checks (all of them; the pool is smaller than 15 for every type).

Reference thresholds for a type to go into larger synthesis, written down
before the run so the results do not move them, but to be revisited once
real output exists: automatic checks pass on ≥ 95% of answered items; no
answer on ≤ 3%; SBN meaning correct on ≥ 90% of audited items; natural on
≥ 80%.

Larger synthesis does not mean every PMB document: the RP samples to a quota
with at most two sites per document. Selection runs locally and only the
sample TSV goes to the cluster, so PMB itself never needs to be there.

# Round 2: results (Muse Glimmer 30B, 2026-10-04)

350 documents, 50 per type, settings as planned above (`hpc/run_r2.sub`).
Files: `results/meta-models_Muse-Glimmer-30B.r2.tsv`, `.r2.checked.tsv`, raw
replies in `results/raw/meta-models_Muse-Glimmer-30B.r2/`.

| type | n | no answer | answered | pass all checks | pass % | no answer % |
|---|---|---|---|---|---|---|
| content word | 50 | 0 | 50 | 44 | 88.0 | 0 |
| constant: name | 50 | 0 | 50 | 48 | 96.0 | 0 |
| constant: number | 50 | 0 | 50 | 49 | 98.0 | 0 |
| constant: temporal | 50 | 1 | 49 | 48 | 98.0 | 2.0 |
| tense | 50 | 0 | 50 | 48 | 96.0 | 0 |
| edge label | 50 | 8 | 42 | 39 | 92.9 | 16.0 |
| forwarding | 50 | 2 | 48 | 48 | 100 | 4.0 |
| **all** | 350 | 11 | 339 | 324 | 95.6 | 3.1 |

"Pass all checks" is `auto_pass` in `audit.py`: structural checks, WordNet,
no `same_synset`, `untouched`, `repair_part`. No item failed for lack of an
answer: every reply was JSON, and all eleven gaps are `no_site`.

**Token limit.** Two items went past the old 16384 output tokens per turn
(longest 18370, `p52/d2122`); median 5.8k against 7.8k in round 1. The limit
of 32768 is enough.

**Failures (15).**

- *Synonyms* (4, content word): `compose`/`write`, `sweat`/`perspiration`,
  `teller`/`bank_clerk`, `purchase`/`buy`. These are filtered, so they cost
  items but do not reach the data. Without them content word passes 44 of 46
  (95.7%).
- *Reparandum and repair the wrong way round* (2): "the field of biology,
  physics" (`p10/d2508`), "Thirteen Americans, Canadians" (`p25/d3091`). The
  final reading is no longer the gold one. `wn_relation` reports
  `biology`/`biology` as `same_synset` because the model wrote the gold
  concept in both synset fields.
- *Gold edges lost* (3): `p48/d2738` drops `person.n.01` ("famous sculptor,
  painter"), `p17/d0919` drops `Value`, `p10/d3126` rewrites `meet.v.02` and
  loses three edges. `p19/d3040` reports a `site_index` outside the gold SBN
  (10 of 6 concepts), so the check cannot run; counted as a failure.
- *Broken repair boxes* (3): no `CORRECTION` box (`p01/d2927`, tense), an
  empty reparandum box (`p71/d2730`, tense), a reparandum box with no edge
  to the rest (`p22/d2265`, "Paris, Rome is burning").
- *Sentence changed* (2, edge label): "Tom handed from, to Mary a cup",
  "Susan shined to, from your father's shoes". Both come from documents
  whose role has no preposition in the sentence (see below).

**`no_site` is mostly a selection error.** Of the eight edge-label
`no_site` answers, seven are right: `site_edge_label` in `select_docs.py`
takes any verb with an index-valued `Destination`/`Source`/`Location`, without
looking at the sentence. PMB uses these roles where no preposition is
written ("He shined his shoes", `Source` on the object; "We ran out of food",
`Source` on the subject; "shear 150 sheep") and for prepositions built with a
spatial operator ("under" = `SZN`, "near" = `APX`, through `location.n.01`).
The eighth, "sitting on the desk", has a site. Of the two forwarding
`no_site` answers, "What is the teacher's last name?" is right (`entity.n.01
EQU ?`); "That is an epic novel" is a miss, since pronoun subjects are valid
forwarding sites (decided 2026-10-02) and the model treated "That" as a
placeholder. So the 16% for edge label
measures the selection, not the model; the selection needs a surface check
before a larger run, as for `p31/d1485` in round 1.

**Oracle (content word).** 41 of 50 equal `build_repair`. The nine others
go to the audit if they pass the checks; they include role kept in the
reparandum (`spread, buttered`), a dummy anchor for `Name ?` (`town, city`)
and inverted roles (`dog, the cat`).

**Self-reported devices.** Content word now uses ⑤ in 5 and ④ in 1 of 50,
so the round-1 impression that the model avoids hard sites does not hold
at this sample size.

**Checks changed in this round.** `check_untouched.py` and
`check_repair_part.py` are now functions called by `analyse_run.py`, which
writes `untouched` and `repair_part`; `auto_pass` requires both.
`check_repair_part.py` no longer stops on an SBN without a `CORRECTION` box,
compares sentences without quote marks and with contractions written out
(the round-1 "Tom's" false alarm is gone), and requires the reparandum part
to contain `reparandum_surface` rather than end with it ("He dug, They dug";
"was from the table, on the table"). Round 1 is unchanged by this except for
`p31/d1485`, which was already known.

**Against the reference thresholds.** Checks ≥ 95%: all types except content
word (88%, 95.7% without filtered synonyms) and edge label (92.9%). No
answer ≤ 3%: all except edge label (16%, of which 2% are model misses) and
forwarding (4%, one miss and one correct `no_site`). The audit (`audit/r2`, 105 items, 15 per type) is in the next section.

## Round 2: human audit (2026-10-04)

105 items labelled by one annotator (`audit/r2/labels.jsonl`, `merged.tsv`).
Two items lack one answer: `a0017` (natural) and `a0038` (meaning, left out
because the sentence is absurd). `audit.py stats`: meaning 95 yes, 1 no,
8 unsure (99% of yes+no, interval 0.94–1.00); natural 96 yes, 3 no, 5 unsure
(97%, 0.91–0.99). No type falls under a threshold, so the second stage does
not run.

**A missed error, and a new check.** `a0057` ("spread, buttered", marked
unsure) is wrong: the repair box copies `butter.v.01 Agent -2` from gold,
but after the inserted `spread.v.09` that index lands on `sister.n.01`, not
`person.n.01`. `check_untouched.py` cannot see this, because it exempts
every edge at the repair site. `check_withdraw.py` removes the CORRECTION
box, merges the ③ and ⑤ dummies back, and requires exactly the gold edges,
with no exemption. Over all three runs it flags only real errors: in round
2, besides those already failing, `a0057`, `p02/d2102` (the ③ dummy's `EQU`
lands on `time.n.08`) and `p05/d2605` (`Theme` renamed `Agent` in the
repair box); in Muse round 1, `p12/d2070` (`Time` to "the late 1960s"
dropped). `analyse_run.py` writes it as `withdraw` and `auto_pass` requires
it. A node is compared as its synset plus its constant-valued roles
(`person.n.01[EQU=speaker]`), since two concepts of one synset hid an index
error otherwise: `p18/d3496` "Jean Reno was, is my favorite actor" points
`Co-Theme` at the speaker instead of the actor, and was labelled correct in
the audit (`a0070`); Opus round 1 `p31/d1973` sends `ANA` to the wrong
`male.n.02`. With it, 320 of 339 answered round-2 items pass all checks
(94.4%), against 324 in the table above; Opus round 1 67 of 68, Muse
round 1 65 of 68. Since the annotator missed `a0070`, the audit's meaning
rate is an upper bound for index errors between same-synset concepts.

Box structure was added to the same check on 2026-10-04: every box operator
other than the pair, and every box-valued role, is compared with gold as an
edge between boxes named by their concepts. `check()` only looked at the
CORRECTION/CONJUNCTION pair, so a renumbered pointer that makes no cycle
passed. It found Muse round 1 `p60/d3120` ("John, Cat Stevens is not a cat.
He's a singer."): gold `CONTINUATION <1` was raised to `<3` and now hangs
off the withdrawn CORRECTION box. Round 2 is unchanged by it. `a0057` leaves the audit pool.

**The unsure meaning labels, read again with the indices resolved.** All
are correct against gold; `a0065` (marked no) too:
- `a0065` "Tom was dozing off, is dozing off": the gold `time.n.08 EQU now`
  stays before the CORRECTION box; the reparandum has its own
  `time.n.08 TPR now`; the repair verb points back to the gold time.
- `a0008` "18 May, June": PMB writes a date as two `time.n.08` joined by
  `EQU` (day 18, month 6); both months carry `FinishOf` ("until") and `EQU`
  to the day.
- `a0018` "next Thursday, Friday": gold has `see.v.08 Theme` to the day (a
  gold oddity; `Time` would be expected); the model keeps it, inverted.
- `a0023`, `a0039`, `a0040`, `a0051` (`SY2` = "between"), `a0102`: correct;
  in `a0039` the rest of the sentence falls into the repair box, as with
  every in-place repair.
- `a0043` "Carlos killed, Who killed …": the SBN is right; the item fails on
  the sentence (below).

The labels file is left as the annotator wrote it.

**Selection rules suggested by the `no` items.** Each comes from a site the
selection should not offer:
- a `?` target or subject (`EQU ?`, `Name ?`): "Where on earth were you
  from, where on earth were you?" (`a0002`) and "Carlos killed, Who
  killed …" (`a0043`) are restarts that change the clause type, not
  repairs within a clause. In `p90/d1638` the model's reasoning weighs
  `no_site` four times and sees that the sentence has no preposition, then
  picks `Source`, which can only surface as a stranded "from".
- a `Quantity` with no numeral in the sentence: "Tom and Mary three, both
  laughed" (`a0038`; "both" is `Quantity 2`).
- edge label roles with no preposition in the sentence, and prepositions
  built with a spatial operator (see round 2 results).

**Open questions from the notes, for the user.**
- *Edge label.* 7 notes call the preposition swaps awkward ("living from,
  in Tokyo"; "spat from the ground, on the ground"). With three roles,
  every swap is a large change of meaning, and the model takes `Source`
  almost every time. Natural slips (in/at/on) are all `Location`, so they
  are not edge-label repairs in SBN. Whether to keep the type as it is,
  widen it (spatial operators, more roles) or weight it down is open.
- *No interregnum.* "arrive in Osaka, Tokyo" (`a0095`) and "18 May, June"
  read as a list or an address. The planned interregnum insertion removes
  this; items that stay without one may need a filter for such pairs.
- *Dummy necessity* (`a0017`, Koko's IQ): the reparandum "Molly" relates
  only to `iq.n.01`, which comes after the repair box, and R3 forbids a
  forward edge, so ⑤ (or moving `iq.n.01`, ④) is needed.

# Full run on train: decisions and selection (2026-10-04)

**Decided (user, 2026-10-04).** All repair types are generated by the LLM,
content word included (the rule pipeline's core layer is no longer the plan
for the data). Tense is back in, with the quota it had before it was cut
(train 500): its reason for removal, the cost of a multi-concept rule
operator, does not apply to LLM generation, and it passed round 2. Train
first; dev and test follow PMB's split and reuse the same workflow later.
Six types go to the full run with the round-2 prompt unchanged; edge label
gets a small round 3 with its new site definition first. Place containment
is a post-hoc check only, so the prompt the six types were validated on
stays as it was.

**Selection patches in `select_docs.py`.**
- Edge label: gold `Source` or `Destination`, or `Location` of `be.v.03`;
  the target after the verb, not a `?` slot, not `location.n.01`; the
  sentence has a preposition for the role (Source: from/off/out;
  Destination: to/into/onto/toward(s)/for, directional only; Location: in/on/
  at/inside). Of the 50 round-2 documents, 18 remain, among them none of
  the eight `no_site` answers and none of the static-verb swaps ("living
  from, in Tokyo", "spat from, on the ground").
- Forwarding: no `?` subject ("Carlos killed, Who killed …").
- Number: the value is written as a numeral or number word in the sentence
  ("both" = `Quantity 2` is not).
- Temporal: no `?` value (`YearOfCentury ?`).

**Post-hoc: `place_inside`** in `analyse_run.py` (WordNet part holonyms;
"Osaka, Japan"); `auto_pass` fails it. WordNet knows large places only.

**Supply on train** (pool 9,100 after examples and earlier samples), against
the RP quota and what it needs at 70% headroom and 94% pass (quota / 0.7 /
0.94):

| type | quota | needed | eligible docs |
|---|---|---|---|
| content word | 2,200 | 3,343 | 9,086 |
| name | 500 | 760 | 2,978 (755 at one doc per value) |
| number | 400 | 608 | 650 (144 at one per value) |
| temporal | 220 | 334 | 228 (111 at one per value) |
| tense | 500 | 760 | 6,928 |
| edge label | 500 | 760 | 642 |
| forwarding | 200 | 304 | 4,188 |

Number, temporal and edge label cannot reach their quota on train; the
one-document-per-value rule, written for 10-item samples, cuts the constant
types further. Open: a cap per value instead of one.

**Post-hoc: `site_ok`.** The item gives the model the whole document and
the model picks the site, so selection only guarantees that a good site
exists. `analyse_run.py` now checks the chosen `site_index` against
`select_docs.SITES` for the type; `auto_pass` requires it. On round 2 it
applies the 2026-10-04 rules after the fact (29 rows outside, mostly edge
label), so round-2 pass rates under the old and new site rules differ; the
round-2 tables above are under the old ones.

## Review before the full run (2026-10-04)

Done in this session: `check_withdraw.py` (with node signatures and box
structure), `site_ok`, `place_inside`, the selection patches.

Before the full run:
1. ~~`select_docs.py`: per-value cap and quotas.~~ Done: `--quota`, `--cap`.
   All seven types were drawn together (seed 13, excluding rounds 1 and 2),
   so edge label keeps its documents: `sample_train_full.tsv` (5,993: content
   word 3,343, name 760, number 598, temporal 228, tense 760, forwarding
   304), `sample_r3_edge.tsv` (50) and `sample_train_edge_reserve.tsv` (575,
   for edge label's full run if round 3 passes).
2. ~~Content word quota by device.~~ Decided 2026-10-04 (user): no device
   axis. Content word is 2,200 in total and the device distribution is
   reported as it comes out. The device follows from the site, which follows
   from syntax (DESIGN.md step 4); forcing it would repeat the pilot's lesson
   that pushing for a phenomenon produces awkward repairs. The prompt stays
   as round 2 validated it.
3. ~~Edge label prompt.~~ Done in `prompt_docs_edge_r3/` (§4.4 rewritten,
   with two sentence-only negative examples; nothing else differs).
   `prompt_docs/` stays as round 2 ran it, because the documents are one
   system prompt for every type: editing them in place would have changed
   the six types' validated condition. `run_llm.py --docs` picks the set and
   `run_meta` records its sha256.
4. ~~Job files.~~ `hpc/run_train_full.sub` (4-day limit; about 30 h
   expected) and `hpc/run_r3_edge.sub`. Batch jobs may run up to one week
   on LST (`+MaxWallTime = 604800`; 24 h is the interactive limit).
5. Commit everything before submitting, so the run can name its commit.
6. Audit plan for the full run: blind sample per type, as in round 2.

After the full run, not to be lost:
- round 2's six-type rows ran with the same `prompt_docs/`; those that pass
  the current checks (including `site_ok`) can join the data, since their
  documents were excluded from the full-run sample;
- interregnum insertion, emitting paired base and marked rows (`pair_id`,
  DESIGN.md §1 step 5); `sorry` reserved for the test set;
- dev and test on PMB's split;
- a second site per document where a type runs short (RP allows two);
- the Deferred list below (name–concept table, Smatch operator
  normalisation, anaphora and cross-turn);
- evaluation: smatch++ needs `GenericStandardizer`; PMB's `INDEX_PATTERN`
  reads one digit (the `magnitude` check keeps such SBN out of the data, so
  documents with an index of 10 or more are not in the corpus, a limit to
  report);
- per-request seeds do not make vLLM runs reproducible under batching;
- `documentation/model_experiment_pipeline.md` §5.3 still describes the rule
  pipeline as the source of the core layer, and "Found here, for the
  rule-pipeline upgrade" below no longer feeds the data.

# Edge label round 3 (2026-10-05)

50 documents of `sample_r3_edge.tsv` with `prompt_docs_edge_r3/` (sha256
`1886b812…`, confirmed in `run_meta`). 49 answered, one `no_site`; 48 of 49
pass the automatic checks (one repair box writes `Destination` for gold
`Location`).

**The `no_site` was right, and showed a mismatch.** "This woman is from
China" is `be.v.03 Source`: selection offered it (any `Source`), §4.4 allows
only be's `Location`. Selection now follows the prompt.

**Reading all 49.** Direction swaps on motion and transfer verbs (27–30)
read naturally ("Tom walked from, into Mary's room", "He withdrew his hand
to, from the table"). The 19 `be.v.03` repairs split by subject: with a
person they work ("She's from, in Boston", "I'm from, in Modena"); with an
existential or a thing they do not ("There are eleven rooms from, in this
house", "There's a cat from, on the desk", "My room is from, on the fourth
floor"). `select_docs._be_person` now requires a person subject (not a
NEGATION quantifier) and no existential `EQU` for `be.v.03`; this changes
selection only, not the prompt. Under it, 35 of the 49 sites remain, 34
pass every check (97%), and the edge reserve keeps 405 of 575 documents.

All 34 go to a blind audit (`audit/r3_edge`): naturalness is the open
question for this type and the set is small enough to read whole.

# Full run on train: first analysis (2026-10-06)

`results/meta-models_Muse-Glimmer-30B.train_full.tsv`: 5,991 of 5,993
documents (`p69/d0350`, `p94/d2611` have no row and no raw reply, probably a
request error; a resubmit would retry them). `run_meta` confirms
`prompt_docs/` with sha256 `6b17847a…`, the round-2 prompt. 6 `no_site`
(0.1%; round 2 had up to 4% for these types), one `error_max_turns`.

| type | answered | pass all checks | % | % without synonyms | RP quota |
|---|---|---|---|---|---|
| content word | 3,341 | 3,114 | 93.2 | 96.5 | 2,200 |
| name | 759 | 725 | 95.5 | 95.5 | 500 |
| number | 597 | 565 | 94.6 | 94.6 | 400 |
| temporal | 228 | 221 | 96.9 | 96.9 | 220 |
| tense | 760 | 730 | 96.1 | 96.1 | 500 |
| forwarding | 299 | 288 | 96.3 | 96.3 | 200 |
| **all** | 5,984 | 5,643 | 94.3 | 96.1 | 4,020 |

Every type has more passing items than its quota. Synonyms (112 content
words) are the largest single failure; the next is the reparandum and the
repair written the wrong way round ("covered in pigeon, crow poop", "is
missing, empty!"), which `repair_part` catches through the onset and the
rest-of-sentence checks.

**Check fix.** At this size `repair_part` raised two false alarms that the
small rounds had not shown: a final full stop the gold sentence lacks
("slicing tofu"), and typographic apostrophes ("That’s"). `norm()` now
drops final punctuation and maps curly quotes to plain ones.

Earlier runs, re-analysed with the current checks and site rules, pass
fewer items than their own sections report (Opus round 1 61 of 68, Muse
round 1 58, round 2 285 of 339), because `site_ok` applies the narrowed
site rules after the fact.

Blind audit sample: `audit/train_full`, 15 per type (90).

# Audits of edge round 3 and the full run (2026-10-08)

Labelled by the user: `audit/r3_edge` (34, all), `audit/train_full` (90,
15 per type; `a0001` left blank on purpose, `a0073` and `a0076` without
answers). Claude labelled the edge set independently and read every item
of both sets with the indices resolved.

**Edge label: natural 48%** (14 yes, 15 no, 5 unsure; threshold 80%).
Claude's independent labels give 55%; on the 25 items both decided, 21
agree. Both reject verbs that carry a direction ("escaped to, from the
zoo", "imports clothes to, from Turkey"). They differ on direction-neutral
verbs: the user rejects "Tom has gone from, to Boston" because "have gone"
is a near-fixed construction for having left and not returned; tense and
aspect move the judgement, so no verb list can stand in for it.
`a0015` ("wiped him into, out of her memory") is a meaning error: "out" is
part of `wipe_out.v.04`, not a role preposition.

**Full run: meaning 100% (72 yes, 15 unsure), natural 97%.** Every unsure
meaning label was checked against the notation and is correct. Eleven are
forwarding repairs whose repeated verb points to its own subject rather
than to the anchor: the user's own notation (`repair_sbn_notation.qmd`,
"Josh drove, no, Marsha drove …") does exactly this, while the subject
repair ("Josh, no, Mary plays tennis well") points the verb to the anchor.
The rule behind the difference: a predicate said once is shared by both
versions and attaches to the anchor; a predicate said in each version
attaches to that version's subject. After withdrawal both give the same
graph.

## Decided (2026-10-08, user)

- **Edge label** stays in train as a small, human-selected sample:
  `sample_train_edge.tsv` (403 reserve documents still meeting the site
  rules) runs with `prompt_docs_edge_r3/` (`hpc/run_train_edge.sub`); the
  user then labels naturalness and keeps the yes items (about 100–150). No
  verb list: it would approximate a judgement that depends on tense and
  context. Not covered by the notation and noted for later: spatial
  operator swaps ("on, actually, under the desk": `Location` against
  `location.n.01 SZN`), which may be the most natural preposition repairs.
- **"Natural"** means natural for fluent speakers, in van Leusen's sense of
  incompatibility in context: grammatical, recognisable as a repair, with
  the withdrawn and the committed content recoverable. Repairs only a
  second-language learner would make are out of scope: disfluency can
  fall anywhere, and widening the speaker population would make the data
  unrepresentative.
- **Fluent readings without an editing expression.** Text has no focus or
  prosody, so some base forms are the same string as a fluent construction:
  "two, three brothers" (a range), "Jim, Tom broke it" (a vocative),
  apposition, lists. The experimental design is not changed: training is
  already unpaired with about 30% marked items (`model_experiment_pipeline.md`
  §457), so these items are assigned to the marked share; the test set
  keeps its marked/unmarked pairs and reports them as a separate
  `ambiguous` stratum. Left unrouted, about 570 bare items would teach the
  juxtaposition template that the hard-negative set tests for, and the
  hard-negative set excludes numbers, so ranges would go unmeasured.
  `analyse_run.py` writes `ambiguous` (`number_range`, `initial_name`; not a
  failure). Full run, passing items: 293 and 279. Apposition and lists are
  not detected yet. The earlier plan to drop n, n+1 numbers is replaced by
  this; randomising the reparandum number was rejected because it edits
  model output and pushes the distribution away from real number repairs.
- **Forwarding**: the predicate follows its own subject (above). Open, for
  later: repairs that change a subject bound by a pronoun in the predicate
  ("John warmed, I warmed myself", "my homework") read more like a restart.

Still to do after these: capitalisation of the repair onset ("She needed,
He needed money": 78 of 288 passing forwarding items) in the post-processing
script, with the interregnum and the marked/base assignment.

## Edge label on train: run and selection set (2026-10-08)

`results/meta-models_Muse-Glimmer-30B.train_edge.tsv`, 403 documents,
`prompt_docs_edge_r3/` (sha256 `1886b812…` in `run_meta`). 377 answered,
26 `no_site`: nearly all refuse verbs that are not motion or transfer
("graduated from", "expelled from", "resigned from", "drank from") or that
carry a direction ("escaped from"), as §4.4's "the first version must be
something a speaker could have meant" asks. 366 of 377 pass every check
(97%). The failures are swaps (the gold "from" taken as the reparandum:
"Tom stole something from, to my house"), a role renamed in the repair box,
and one ③ dummy whose `EQU` lands on `time.n.08`.

`check_repair_part.py`: when the repair part is not the end of the gold
sentence, the onset message printed a cut-off fragment ("'om the east'");
it now reads the whole left side. Verdicts are unchanged.

Selection set: `audit/train_edge`, all 366 passing items. `audit.py serve
--questions natural` shows only the naturalness question (the checks cover
meaning); the user keeps the yes items, about 100–150.

## Edge label: places (prompt_docs_edge_r4, 2026-10-10)

The train target is about 500 per type and from/to gives 100–150, so §4.4
gains a third kind: a different place relative to the same thing. PMB writes
"on", "in" and "at" as a plain `Location`/`Destination` to the thing, and
"under", "above", "behind", "near", "beside", "inside", "outside", "around"
as `location.n.01` with `SZN`, `SZP`, `SXP`, `APX`, `SY1`, `STI`, `STO`,
`SXY`. "Next to" (`next.a.02 Theme`) and "in front of" (`front.n.09
PartOf`) use other concepts and are left out; `SXN` never occurs forward in
gold. The committed version is the gold one, so the direction follows the
gold: a gold operator gives "on, under" and operator swaps (106 train
documents), a gold plain role gives "under, on" (907 sites).

One template for all three: two ③ dummies keep the gold role, and a
version with an operator carries its own `location.n.01` inside its box,
pointing back to the thing, which moves in front (④). Examples S–U (gold
`p61/d0863`, `p16/d2891`, `p91/d3375`, added to `EXAMPLE_DOCS`). Three
negatives in §4.4: on/in/at against each other (same SBN), one place
including the other (near/beside, in/inside), a preposition that belongs to
the verb or a fixed phrase. `task.md`: `reparandum_value` for a place is
the rejected operator or `plain`.

`check_repair_part.py`: "reparandum identical to repair" now compares
resolved targets. It used to compare role names only, so "on, under the
table" (both dummies `EQU` + `Location`) counted as identical. New check:
`APX`/`SY1` and plain/`STI` (when both words are "in"/"inside"/"into")
flag as overlap. No verdict changed on the 6,885 answered rows in
`results/`.

Hand estimate (one judge, before the naturalness labels): about 95
operator swaps and 55 "on, under" items on the 106 documents; about 39% of
the 907 plain sites have a usable place (20 of 51 sampled), about 530
items. That estimate still counted "in front of" and "next to", now left
out, so it is a little high. Furniture and buildings give 2–3 options,
cities "near" only, countries and fixed phrases none.

**Site rule** (`select_docs.site_place`). A verb's `Location` or
`Destination`, after the verb, that is either `location.n.01` with a
spatial operator whose word is in the sentence, or a plain role with "on",
"in", "at", "into" or "onto" right before the thing. The thing must be a
physical entity and not a body part ("on the shoulder", "under his arm"), a
country, state or continent ("born in Switzerland"), a relational noun ("in
the middle", "on the right"), a person or the question slot.
`--edge-sites place` draws only documents with no direction site, since the
model picks the site itself and would otherwise fall back on "from, to"; the
default `direction` draws exactly what it did before (checked: same output
for the same seed). `site_edge_label` is now the union, and
`analyse_run.site_ok` picks the rule by the answer's kind (`reparandum_value`
an operator or `plain` → place). Without that, ten round-2 answers like
"There's a cat from, on the desk" would turn from `outside` to `ok`; with it
all 6,885 stored verdicts are unchanged.

Supply (train gold, examples excluded): 778 documents with a place site,
666 with no direction site (705 plain, 79 operator sites). Only 254 of the
666 are in no earlier sample: 412 were drawn for other types, mostly
`sample_train_full` (content word 174, name 64, tense 44), and 38 for edge
label in rounds 1–3.

**Decided (2026-10-10, user): documents may be reused across types**, up
to two items per source document (`model_experiment_pipeline.md` §5.3).
`select_docs.py --exclude-type edge_label` leaves out only earlier
edge-label documents; a document already used twice is always left out.
The two items sit at different sites of the same sentence.

Sample: `sample_train_place.tsv`, 627 documents (374 used once for another
type, 253 fresh), drawn with

    python3 select_docs.py --edge-sites place --exclude-type edge_label \
      --quota content_word=0 constant_name=0 constant_number=0 \
              constant_temporal=0 tense=0 forwarding=0 edge_label=1000 \
      --exclude sample_train_full.tsv sample_train_edge.tsv \
                sample_r3_edge.tsv pinned_sample.tsv pinned_sample_r2.tsv \
      --out sample_train_place.tsv

(seed 13). Cities are the largest group of things (95), which mostly allow
only "near". Job: `hpc/run_train_place.sub`, `prompt_docs_edge_r4/`
(sha256 `24c14008…`).

# Temporal: a second item per document (2026-10-10)

**Target.** About 3,500 repair items on train (the middle of 1:2–1:4
against 9,447 fluent documents, §5.4), about 500 per type. Temporal has 269
passing items (full run 221, round 2 48) and cannot grow by new documents:
under the current site rule only 297 train documents (388 sites) hold a
temporal constant (`ClockTime` 147, `YearOfCentury` 120, `MonthOfYear` 59,
`DayOfMonth` 47, `DayOfWeek` 15), and the first pass used nearly all of them.

**Per-document cap (decided, user).** The "two items per source document"
rule (`DESIGN.md` §4.3) guards against a cell whose count overstates its
variety: 500 items from 148 documents is 148 sentence contexts. It is not
a correctness constraint. It becomes a reporting requirement: each type
gives items and distinct source documents. Use by other types is not
counted. A second item at the same site with another reparandum is a
minimal pair whose targets differ exactly at the reparandum value; at this
scale (about 200 pairs in about 13,000 training rows) it adds a copying
signal and little weight.

**Run.** `sample_temporal_extra.py` → `sample_train_temporal_extra.tsv`, one
row per eligible document (run_llm keys on `doc_id`): 80 `new_site` (a
site no passing item uses: a second site, or a document with no passing
item) and 217 `same_site` (the only site again, `avoid_value` = the first
item's reparandum). The prompt is `prompt_docs_temporal_r2/`: a copy of
`prompt_docs/` whose `item_template.md` adds `Repair site` and `Reparandum
already used`, and whose `task.md` says to take that site (step 2) and
avoid that value (step 3); sha256 `8d49b9ac…`. No new examples. Job
`hpc/run_train_temporal_extra.sub`. These items come from a different
prompt than the first 269 and are reported apart.

**Check.** `analyse_run.py` as usual, then `check_temporal_extra.py` on its
checked TSV adds `pin`: `site` (another site was repaired), `same_as_used`,
`same_as_gold`, or `ok`. Tested with the first-pass rows of the same
documents as input: all 217 `same_site` rows give `same_as_used`, the
`new_site` rows give `site` except the 4 whose failed first item was at the
pinned site.

# Forwarding: more documents (2026-10-10)

Forwarding has 333 passing items (full run 288, round 2 45), about 170
short of 500. Supply is not the limit: 899 eligible train documents were
never drawn for any sample. `sample_train_forwarding_extra.tsv` takes 185
of them (`select_docs.py --per-type 0 --quota forwarding=185 --seed 13`,
every earlier sample excluded); `hpc/run_train_forwarding_extra.sub` runs
them with `prompt_docs/` unchanged (sha256 `6b17847a…`), so the new items
pool with the first 333. Repairs whose predicate holds a pronoun bound to
the subject ("I warmed myself") are not filtered here; they are handled
for all forwarding items when the dataset is assembled.

**Position.** The site rule is structural (subject, verb and tense
`time.n.08` adjacent, one box), so a subject inside a sentence qualifies.
In the data, 320 of the 333 passing items repair the sentence-initial
subject. About 5% of eligible documents (227 of 4,399) have a site past
the first clause, and the 13 medial items (4%) are close to that share, so
the skew follows PMB, whose sentences are short and mostly open with the
subject. Three of the 13 are not forwarding but pass the checks ("He had
his molar pulled, his tooth pulled"; "Another aspect is, Another focal
point is"; "Give the book to child wants, whomever wants it").

# Train: status and open items (2026-10-10)

The design-level record is `documentation/model_experiment_pipeline.md`
§15; this section is the playground's view. Where earlier sections of this
README give quotas (the 10-04 supply table: content word 2,200, name 500,
number 400, temporal 220, tense 500, edge label 500, forwarding 200) or the
two-per-document cap as a selection rule, this section replaces them.

**Target.** About 3,500 repair items on train, about 500 per type; quotas
are caps. The per-document cap is a reporting requirement (items and
distinct source documents per type).

| type | passing (full + r2) | extra run | expected |
|---|---:|---|---:|
| content word | 3,114 + 43 | – | enough |
| name | 725 + 48 | – | enough |
| number | 565 + 46 | – | enough |
| tense | 730 + 47 | – | enough |
| temporal | 221 + 48 | `train_temporal_extra` (297, new prompt) | ~557 |
| forwarding | 288 + 45 | `train_forwarding_extra` (185, same prompt) | ~510 |
| edge label | 112 + 14 (r3 yes) | `train_place` (other thread) | < 500 |

**Thresholds.** The four reference thresholds (checks ≥ 95%, `no_site`
≤ 3%, meaning ≥ 90%, natural ≥ 80%) were set here before round 2 and have
no outside source. They decide whether a type goes to larger synthesis;
they are not a quality benchmark. Edge label is the only type below them
and was moved to human selection rather than given a lower bar.

**train_edge labels (user, 2026-10-09).** 363 of 366 labelled
(`a0035` blank; `a0062`, `a0352` have a note but no label): yes 112, no
199, unsure 52 (natural 36% of yes+no, 95% CI 0.31–0.41). The labels are
first-impression judgements: unsure means "needed a second thought". By
labelling time the yes rate moves 31%, 20%, 32%, 40% over four quarters
(Fisher Q2 against Q4, p = .004; items were shuffled), and near-identical
templates are spread over all three labels (bare "X go from, to Y": 5 no,
11 unsure, 8 yes). Systematic no's: `be.v.03` with a time adverbial ("I'm
from Hong Kong, in Hong Kong right now" reads as a fluent apposition),
imperatives and "Let's", and verbs that carry a direction (escape, steal,
put, wipe, head for). A reported rate needs the single-annotator,
intuition-only caveat.

**Open, for dataset assembly (separate thread).**
- Marked share: 30% overall (as §5.5 reads) or per type. Per type, the
  ambiguous items of number (317) and name (295) exceed it; overall, the
  marker would correlate with type.
- `ambiguous` (`analyse_run.py`): thresholds (≤ 20, multiples of 5) not
  validated against human judgement, so check about 30 items before a
  stratum is reported. `initial_name` fires on any sentence-initial name,
  including cities ("Geneva, Bern is the capital of Switzerland") and lists
  ("Guan Yu, Liu Bei and Zhuge Liang …"); still a fluent reading, but the
  docstring says vocative only. Not detected: medial apposition and lists
  ("arrive in Osaka, Tokyo"), temporal ("18 May, June", which the temporal
  extra run will also produce), content-word lists.
- Forwarding: onset capitalisation (78 items), pronouns bound to the
  subject in the predicate.
- Audit of the temporal extra run (new prompt, about 20 items, half per
  `kind`). The forwarding extra run uses the full run's prompt and pools
  with its audit.
- dev/test: not started.

# Deferred

Each of these was in an earlier draft. They are left out until a first-round
result shows a need for them.

- **Revise loop with validator feedback.** Costly in quota, and retries
  against the validator remove the detectable errors while leaving the
  undetectable ones, which hides the error distribution this playground is
  meant to expose. If added later, report first-attempt validity separately
  from post-retry validity, keep the retry bound small, and keep feedback at
  the level of which check failed, not which rule.
- **Felicity self-rating.** Models rate their own output too generously, and
  there is no human judgement set to check the ratings against.
  (`calib_items.tsv`/`calib_key.tsv` were built for the synonymy question.)
- **Checks the validator cannot see.** An index can point to the wrong but
  legal concept and still pass all structural checks. Both candidates below
  now exist, as `check_untouched.py` and the oracle column of
  `analyse_run.py`:
  - an *untouched-subgraph* check: gold triples with neither endpoint at the
    repair site must survive unchanged in the repaired graph, after
    de-inverting device ② roles. Full isomorphism with gold is not a usable
    target, because devices ③ and ⑤ and delayed repairs change the structure.
  - an *oracle diff* against `repair_transform.build_repair` for the same site
    and reparandum, where the model's choice falls inside what the rule-based
    pipeline supports.
- **Inferring strategy and devices from the SBN** rather than trusting the
  self-report.
- **Smatch with inverted operators.** Smatch normalises inverted roles
  (`ROf` → `R-of`) but not operators: `EQU` written on the other concept, or
  `SZP` for a gold `SZN`, scores as a different triple. Before scoring, map
  symmetric operators to one fixed direction and converse pairs to one member.
- **Near-synonym diagnostic.** Whether the model's reparandum falls inside the
  project's near-synonym exclusion set from `wn_candidates.py`.
- **Anaphora and cross-turn repairs** (`DESIGN.md` tier 4). They need a
  second sentence or speaker turn, which the model would write together
  with its SBN, with no gold to compare against. Anaphora covers a narrow
  case and cross-turn data is rare in PMB. Taken up once the single-sentence
  types run.
- **Name–concept consistency check** (offline). A table of the names in
  gold by host concept (326 male, 136 female names; places by concept),
  used to flag a reparandum name that does not fit its concept (`male.n.02
  Name "Mary"`). Names are not a tool: any quoted string is legal SBN, and
  choosing a fitting name is part of what is tested. The same table can
  serve as the rule pipeline's name pool (`DESIGN.md` §2).
- **Batching several items per session** to save quota. If used, log each
  item's position in the batch and check whether error rate changes with it.

NLI/surprisal scores are not used as a quality filter in any version. The
calibration on the rule-based pool showed they do not discriminate on this
kind of data.

# Found here, for the rule-pipeline upgrade

Not acted on now; recorded so the later upgrade of `repair_transform.py`
can pick them up.

- **R5 counts a constant as a connection.** `build_repair` treats a
  reparandum carrying only `Name "…"` as connected. For "Tom kicked the can"
  it writes `CORRECTION <1 female.n.02 Name "Tom" CONJUNCTION <2 …`, whose
  reparandum has no edge to the event. The notation's own subject case uses
  ⑤ here.
- **No operator inversion.** Where an operator points into the
  `CORRECTION` box, `build_repair` falls back to ⑤ (example J).
- **`retarget_to_repair` mixes two cases** (`repair_transform.py`
  lines 202–204): a source before the `CORRECTION` box pointing over it, and
  a source after the repair simply re-indexed. Device counts from the rule
  pipeline overstate the first.
- **Satellite adjectives are excluded from the candidate pool.**
  `wn_candidates.py` (line 46) drops WordNet `.s.` synsets because PMB gold
  contains no `.s.` names. But PMB writes satellites as `.a.`: 498 distinct
  gold adjectives are satellites (`boring.a.01`, `grey.a.01`), so a large
  part of the adjective inventory is missing from the pool.
- **`sorry` is drawn for all splits.** `generate_repairs.INTERREGNA`
  includes it, but it is meant to appear in the test set only.
- **Only content-word repairs.** Constant, tense, edge-label and forwarding
  repairs have no operator yet (`DESIGN.md` §3 tiers 2 and 4); examples L–Q
  of `prompt_docs/repair_notation.md` can serve as their regression tests.

# Environment, confirmed

`nltk.corpus.wordnet`, `nltk.corpus.verbnet` and `nltk.corpus.framenet`
(`framenet_v17`) all load and return results in this environment. No LLM API
client code exists elsewhere in the repository.
