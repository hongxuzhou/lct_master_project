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
- **Interregnum** is an input too: `none`, or one of I mean · no · no wait ·
  actually. The harness assigns it as `generate_repairs.py` does (30%
  marked, marker drawn at random), so unmarked repairs stay the majority.
  `sorry`, the fifth marker of `DESIGN.md` step 5, is reserved for the test
  set, to probe whether a trained model generalises to an unseen marker; it
  is never shown or generated here.
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
| `interregnum` | input |
| `status`, `reason` | model: `ok` or `no_site` |
| `site_index` | model: position of the repaired concept in the gold SBN (what `build_repair` takes as `i`) |
| `repair_synset` | model's `site_synset` |
| `reparandum_value` | model: the new constant, tense operator or role, where the synset does not change |
| `nl_repair`, `sbn_repair` | model |
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
`p76/d2001`, `p47/d1417`, `p14/d2976`.

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
| `constant_name` | a `Name` constant | 3254 |
| `constant_number` | a plain number (`Quantity 3`, `EQU 100`), not a year or time | 799 |
| `constant_temporal` | `ClockTime`, `DayOfWeek`, `DayOfMonth`, `MonthOfYear`, `YearOfCentury` | 311 |
| `tense` | a verb with `Time` to an adjacent `time.n.08 TPR/EQU/TSU now` in the same box | 7290 |
| `edge_label` | a verb with an index-valued `Destination`/`Source`/`Location` | 1871 |
| `forwarding` | noun subject, its verb, and the verb's tense `time.n.08`, adjacent, as in example Q | 4574 |

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
kept in order). Each of the two added checks was first shown to catch
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
