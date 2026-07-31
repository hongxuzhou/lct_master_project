# Feasibility boundary of LARD-style repair synthesis on PMB gold SBN

Investigation of the plan: take `data/pmb-5.1.0/split/en/train/gold.sbn`, mutate one
concept into a WordNet neighbour to obtain a reparandum, and splice
`CORRECTION` / `CONJUNCTION` around the pair to get a repair-aware SBN whose
diff against the clean SBN is known by construction.

**Verdict: the plan works, and works better than expected — but not for the
reasons or at the boundary that were anticipated.** The `entity.n.01` dummy is
*not* where the method fails outright — but it was, until the fix in §2a,
over-used relative to how rarely real annotators reach for it. The real
coverage boundary is elsewhere, and is described in §4.

Everything below is measured, not estimated. Code is in this directory;
`python3 test_against_doc.py`, `python3 analyse_feasibility.py`,
`python3 validate_with_pmb_parser.py`, `python3 generate_repairs.py`.

---

## 1. The transform is not a "diff two graphs" problem

The plan describes generating sentence A and sentence B, parsing both, and
diffing the two SBNs. That step is unnecessary and should be dropped: because
the reparandum is produced by *substitution* on a gold graph, the graph-level
diff is known a priori — it is exactly one concept node. No second parse, no
graph alignment, and therefore no parser error enters the data.

What is *not* trivial is everything the substitution drags with it. Inserting
`CORRECTION`/`CONJUNCTION` at concept position `i`:

* adds one (sometimes two) concepts, so **every** `±n` index that spans the
  splice point shifts — indices count all concepts linearly (C1);
* moves the repaired concept's argument edges across a box boundary, which C3
  forbids in one direction and permits in the other;
* adds **two boxes**, so every downstream `<n` box pointer *and* every
  box-valued role (`Proposition >1`) has to be renumbered too. This second
  index space is easy to miss and fails silently — see §5.

`repair_transform.py` implements all of this. Its correctness check is
`test_against_doc.py`: fed the *clean* SBN of each covered case in
`documentation/knowledge_base/repair_sbn_notation.qmd`, it must reproduce the
hand-written repair SBN character for character.

**8 of 9 covered cases reproduce exactly.** The ninth (forwarding repair) is
not a single-concept repair at all — see §4.3.

---

## 2. There is a fifth device, and it does most of the work

The notation doc's toolkit lists four devices. The subject self-repair case
uses a fifth one that is not in the table:

> **Device ⑤ — anchor dummy.** Put `entity.n.01` *before* the `CORRECTION`,
> redirect every inbound edge onto it, and hang both copies off it with `EQU`.

This is what makes `entity.n.01 / CORRECTION / male.n.02 Name "Josh" EQU -1 /
CONJUNCTION / female.n.02 Name "Mary" EQU -2 / play.v.01 Agent -3` legal. It
should be written into the toolkit table, because it is the *only* device that
fixes two situations the other four cannot:

| Situation | ①drop | ②invert | ④move | ⑤anchor |
|---|---|---|---|---|
| inbound edge whose role has no `…Of` inverse (`Role`, `Location`, `Patient`, `Stimulus`, `Destination`) | ✗ | ✗ | ✗ | ✓ |
| reparandum would be left with no edges at all | ✗ | ✗ | ✗ | ✓ |

Crucially ⑤ solves the non-invertible-inbound problem **without adding roles to
`SBNSpec.ROLES`**. Before ⑤ was implemented, 5,001 sites were blocked by
non-invertible inbound roles (`Role` ×966, `Location` ×569, `Patient` ×554,
`Stimulus` ×445, `Destination` ×385 …) and the obvious fix would have been to
mint `RoleOf`, `LocationOf`, `PatientOf`. That would have been schema growth
for an ad-hoc case. With ⑤, 18 sites remain blocked.

**Strategy order matters and should be POS-dependent.** `entity.n.01 EQU X`
asserts an entity identity, which only reads correctly when `X` is a referent.
For adjectives and verbs, device ④ (movement) is tried first instead —
otherwise a prenominal adjective repair produces `entity.n.01 EQU fresh.a.01`
("an entity that *is* freshness"), which parses but is nonsense. With the fix,
"Sixty new museums opened" correctly yields
`quantity.n.01 EQU 60 museum.n.01 Quantity -1 CORRECTION <1 current.a.01 AttributeOf -1 CONJUNCTION <2 new.a.01 AttributeOf -2 …`.

## 2a. ⑤ was over-used: "couldn't invert" is not the same failure as "would dangle"

⑤ is not free. It is a node-level operation (a new concept plus two `EQU`
edges) standing in for what is, in every other device, an edge-level rewrite —
a heavier intervention than SBN's own economy of primitives would suggest, and
Bos's own repeated use of `entity.n.01` (disjunction, partitives, coordination,
the anonymous-pronoun placeholder) is nowhere near as frequent as ⑤ fired here.
Measured: `LocationOf`/`PatientOf`/`StimulusOf`/`DestinationOf`/`RoleOf` occur
**zero** times anywhere in gold — unlike the five project-added invertible
roles (`ThemeOf`, `TimeOf`, `AgentOf`, `MannerOf`, `Co-ThemeOf`), which were
chosen because they mirror the *shape* of the six stock invertible roles
(`PartOf`, `AttributeOf`, …), inventing these would have zero precedent even by
that looser standard. And the `entity.n.01`-with-≥2-edges shape that gold *does*
have (7.7% of docs) is overwhelmingly coordination/partitive/quantifier
constructions, not "two competing readings of one referent" — the pattern ⑤
actually implements has exactly one native precedent, the doc's own
subject-repair example.

The original code conflated two questions that only look like one:

1. *Can this specific inbound role be flipped to a registered `…Of` form?*
2. *Would the reparandum end up with zero edges at all if it can't?*

`INVERSE_OF` only maps base-role → `…Of`-role (`Attribute`→`AttributeOf`), not
the reverse. So a role that is *already* `…Of`-shaped in gold's own writing —
`AttributeOf`, the standard way gold attaches a prenominal adjective to its
head noun — was never recognised as "already fine", and question (1) failing
was treated as automatically failing question (2) as well, escalating straight
to ⑤. It doesn't: a source concept sitting before both new boxes can *always*
legally retarget its edge onto the repair copy, because that copy lives in the
`CONJUNCTION` box and C3 calls that boundary permeable — whether the role has a
registered inverse or not. Losing one specific inbound edge only matters if
nothing else keeps the reparandum connected.

Worked example — "I met a little girl, [correction,] boy", repairing the noun
under a fixed prenominal adjective (a case not in the original toolkit table at
all): before the fix, the generator reached straight for ⑤. After separating
the two questions, it produces, unforced:

```
person.n.01 EQU speaker time.n.08 TPR now meet.v.01 Agent -2 Time -1
little.a.01 AttributeOf +2
    CORRECTION <1
girl.n.01 ThemeOf -2
    CONJUNCTION <2
boy.n.01 ThemeOf -3
```

`little.a.01` keeps its original `AttributeOf` edge (unflipped, just
renumbered) reaching forward into the permeable `CONJUNCTION` box; `girl`/`boy`
each separately invert `meet.v.01`'s `Theme` (which *does* have a registered
inverse) so neither copy dangles. Verified against `sbn_smatch.SBNGraph`: legal
DAG, `CORRECTION`/`CONJUNCTION` correct siblings. No `entity.n.01` anywhere.

Two more bugs surfaced chasing this down, both now guarded against directly
inside `build_repair` (not just in the external validator) so a bad candidate
is rejected and the strategy fallback silently tries the next one:

* **Device② can close a cycle through a path invisible to the edge being
  inverted.** "They sell apples, oranges, eggs, and so on." has
  `sell.v.01 Time +2` landing on `apple.n.01` (almost certainly a gold
  off-by-one — `Time` normally targets a `time.n.08` node one position over —
  but gold is gold, not something to second-guess) *and separately*
  `apple.n.01` is reachable from `sell.v.01` via `Theme → {apple, orange, egg}
  set-node → Sub → apple`. Repairing `apple` inverted the `Time` edge onto both
  copies, closing `sell → set-node → apple → sell`. This can't be ruled out in
  advance without a full reachability check, so `build_repair` now constructs
  the candidate's concept-to-concept graph and rejects it with a new
  `Blocker.CYCLE` if `networkx` finds one — the same check
  `validate_with_pmb_parser.py` runs externally, just moved inside the
  generator so the strategy fallback can react to it automatically instead of
  silently emitting a bad sample.
* **A pre-existing separator at the exact repair site collided with ours.**
  "Give the book to whomever wants it." already has its own `CONJUNCTION`
  opening right before `book.n.02` (part of the sentence's own
  universal-quantifier construction, unrelated to repair). The sort that
  decides separator order at a shared position was keying off *names*
  (`"CORRECTION"`/`"CONJUNCTION"`), so a pre-existing separator that happened
  to also be called `CONJUNCTION` tied with ours and (stable sort) lost,
  landing *after* ours instead of containing it — producing an empty
  `CORRECTION` box and a reparandum stranded in the wrong box. Fixed by tagging
  our two separators by object identity instead of by name, so a pre-existing
  opener at the same position always nests correctly around the splice
  regardless of what it happens to be called. `Blocker.SEPARATOR_COLLISION` is
  kept as a belt-and-braces check on top (verifies the reparandum/repair
  copies actually land in *our* boxes), in case some other configuration
  produces the same collision by a different route.

Net effect, measured: anchor usage in the generated corpus dropped from **35%
to 27%** (5,328/15,197 → 4,098/15,197), with a fuller re-validation (12,000
sampled repairs) showing zero cycles and zero sibling violations — the two
failure modes above were real and are now caught before they reach the output,
not just in this fix's own test cases.

---

## 3. Coverage, measured

Over all 9,552 English gold train documents:

```
structural / non-lexical concepts   24655       (time.n.08, entity.n.01, person.n.01, …)
lexical concepts (n/v/a)            23232   100.0%
  not alignable to an uttered word   5346    23.0%
  aligned                           17886    77.0%
    no WordNet reparandum            1405     6.0%
    usable candidate site           16481    70.9%

splice outcome on usable sites
  legal                             16425    99.7%
  blocked                              56    0.3%
```

Blocked, all strategies exhausted: `index_overflow` 45, `dangling_reparandum`
23, `hoist_unavailable` 50 (overlapping). `Cycle` and `Separator_collision`
(§2a) reject individual *strategy attempts*, not sites — every site that hit
one of them was still rescued by falling through to a different strategy, so
neither appears in this final tally.

By part of speech, with the strategy each site ended up needing:

| POS | lexical | aligned | usable | legal | inplace / anchor / hoist |
|---|---|---|---|---|---|
| n | 11781 | 10060 | 9560 | 9525 | 4576 / 4949 / 0 |
| v | 8609 | 5091 | 4229 | 4221 | 4170 / 7 / 44 |
| a | 2842 | 2735 | 2692 | 2679 | 2050 / 148 / 481 |

Anchor's share of legal splices: **31%** (5,104/16,425), down from ~44% before
§2a's fix — most of the difference is exactly the class §2a describes: nouns
whose only inbound edge is a non-registered role but which stay connected
through something else. Verbs barely used anchor to begin with (7 sites) —
unsurprising, since a verb's own arguments are usually forward-pointing and
get dropped by device① rather than needing to reach backward into it.

Per document: **8,732 / 9,552 (91.4%) yield at least one legal repair site.**
At two samples per document the generator produces **15,197 samples covering
95.0% of gold documents**.

Two supporting facts that the design depends on:

* **SBN concept order tracks surface word order in 98.2% of documents.** This
  is what licenses device ① ("the argument had not been uttered yet") — dropping
  a forward role is only honest if the target word really does come later.
* **Adjective satellites are a non-issue.** WordNet adjectives have no
  hypernyms, so LARD's co-hyponym recipe returns nothing for them and the naive
  implementation scores 0/2842. The fallback (`similar_to` / `also_see` /
  antonym) fixes it. Note that the similarity cluster is full of `.s.`
  satellite synsets, which `SBNSpec.SYNSET_PATTERN` cannot write (`n|v|a|r|x`
  only) and which occur **zero** times in gold — so they must be filtered, and
  after filtering every adjective still has at least one usable candidate.

Every generated sample was checked against the project's own parser
(`sbn_smatch.SBNGraph`). Of 12,000 sampled: 11,996 parse, all parsed samples
produce `CORRECTION` and `CONJUNCTION` as siblings off the same parent box,
11,985 lose no edges. All remaining failures are pre-existing gold defects
(§5), not transform bugs — see §2a for the two transform bugs that *were*
found this way (cycles, separator collisions) and are now caught internally
before a sample is ever emitted.

---

## 4. Where it actually breaks

### 4.1 Not the `entity.n.01` dummy, once ⑤ is scoped correctly

The anticipated problem — that the `entity.n.01` dummy pattern would need
complex handling — is inverted: the dummy is cheap (one node, two `EQU`
edges) and mechanically derivable. What §2a corrects is *how often* it's
reached for. Before that fix it fired whenever an inbound role merely lacked a
registered inverse (44% of legal splices); after separating "can't formally
invert this one edge" from "the reparandum would actually dangle", it now
fires only when the reparandum genuinely has no other connection — 31% of
legal splices, and structurally these are exactly the subject-position /
bare-argument referents (§2a's C3 sibling-crossing restriction has no other
fix). Its remaining cost is at *scoring* time: it adds an `entity.n.01`
instance triple plus two `EQU` triples to every graph that uses it, which a
downstream parser must reproduce to score well — a real but now much smaller
tax than before the fix.

### 4.2 The real boundary is the *mutation operator*, not the splice

The splice is nearly total (99.7%). What is narrow is **what a WordNet synset
substitution can express.** SBN carries four kinds of repairable material and
lexical substitution reaches only one:

| Repairable material | SBN encoding | Reachable by synset substitution? | Gold docs containing it |
|---|---|---|---|
| content word | concept node | **yes** | 91.4% (≥1 legal site) |
| constant | role *value* (`DayOfWeek monday`, `ClockTime 22:00`, `Quantity 3`, `Name "Josh"`) | no | 47.1% |
| tense / aspect | operator on `time.n.08` (`TPR`/`EQU`/`TSU`) | no | 97.3% |
| edge label | role name (`Destination` → `Source`) | no | 97.9% |

This is the headline. The notation doc's covered-case inventory is broader than
the generator's reach:

| Covered case | generable now |
|---|---|
| SV sentence, verb repair | ✓ |
| SVO subject repair | ✓ (device ⑤) |
| SVO object repair | ✓ (device ②) |
| Negation | ✓ (inherited; the repair correctly merges into the negated box) |
| Donkey sentence | ✓ |
| Retracting repair | ✓ |
| Adjunct (`monday` → `tuesday`) | ✗ — repairs a *constant* |
| Tense & aspect replacement | ✗ — repairs an *operator* |
| Preposition replacement | ✗ — repairs an *edge label* (device ③) |
| Forwarding repair | ✗ — repairs a *span* (subject **and** verb re-uttered) |
| Anaphora over reparandum+repair | ✗ — needs a second utterance |
| Intra-turn + cross-turn combined | ✗ — needs discourse composition |

The good news: three of the six gaps are cheap extensions of the same
machinery, and they are large. A **constant-mutation operator** (perturb
`DayOfWeek`, `ClockTime`, `Quantity`, `MonthOfYear`, `Name`) reaches 47.1% of
documents and directly produces the Adjunct case. A **tense/aspect operator**
(perturb `TPR`/`EQU`/`TSU` on `time.n.08`) reaches 97.3% and is structurally
identical to a verb repair, as the notation doc already argues. An
**edge-label operator** with device ③ reaches 97.9% and produces the
preposition case. None of them needs any new notation.

The remaining three (span repair, anaphora, cross-turn) require composing more
than one gold document or more than one concept, which is a different generator,
not an extension of this one.

### 4.3 Repetition and degree->1 replacement

LARD's degree-2/3 replacements repeat 1–3 words before the reparandum. Only
words with **no SBN concept** can be repeated for free (determiners,
prepositions, auxiliaries, particles) — repeating a content word would require
that word to appear inside the `CORRECTION` box too, which is a two-concept
repair. The generator therefore repeats only function words. Pure LARD
*repetitions* ("Let's meet [today + today]") are outside this notation
altogether: reparandum and repair are the same concept, so the graph has
nothing to quarantine.

### 4.4 Semantic plausibility of the reparandum — the weakest link

This is the largest *quality* problem, and it is inherited from LARD rather
than caused by SBN. WordNet co-hyponymy gives good nouns and poor verbs,
because the verb hierarchy is shallow: `bear.v.02` sits directly under
`produce.v.01` (depth 1), so its co-hyponyms are `sporulate`, `manufacture`,
`grind_out`. The generator ranks candidates by relation tightness (verb group
and adjective similarity cluster before co-hyponyms), single-word lemmas,
sense number and path similarity, which fixes the worst of it —
`see.v.01 → watch.v.03`, `doctor.n.01 → dentist.n.01`,
`dog.n.01 → hyena/jackal/fox`, `new.a.01 → current/fresh/modern` — but leaves
residue: *"The wind is blowing region, or rather, east."*

LARD 2025's answer is a sentence-encoder re-rank (their eq. 1–3), and that is
the right fix here too. It is not implemented, because it needs a model
download. It should be the next thing done, and it is orthogonal to everything
in §1–§3.

### 4.5 Alignment

23% of lexical concepts cannot be tied to a surface word by lemma+POS matching.
Some are genuinely non-surface (implicit concepts PMB introduces); some are
matcher failures. These sites are dropped, which costs recall but not
correctness. PMB 5.1.0's split files carry no token alignment, so this ceiling
is a property of the data source, not the method.

---

## 5. Toolchain defects found along the way

These affect the evaluation pipeline as much as the generator.

1. **`CauserOf` is attested in gold but missing from `SBNSpec.ROLES`.**
   `p36/d2375` and `p30/d2358` abort the stock parser with "Invalid token
   found". One line to fix in `sbn_spec.py`.

2. **`INDEX_PATTERN` matches a single digit**, so `-10` is silently read as
   `-1`. The splice inflates index magnitudes, so this is a live constraint,
   not a theoretical one: 45 sites are blocked by it, and 44 generated samples
   sit at magnitude 9 — one concept away from silent corruption. The generator
   records `max_abs_index` per row so these can be filtered.

3. **Box pointers are a second index space and the splice must renumber them.**
   16.2% of gold documents contain a separator; 1.9% contain a box-valued role
   (`Proposition >1` ×172, `Proposition <1` ×12). Before this was handled, the
   splice silently produced **cyclic graphs** (a `Proposition <1` from inside
   the `CONJUNCTION` box pointing back at the `CORRECTION` box) and mis-targeted
   downstream `CONJUNCTION` pointers. Both parse without error. This is the
   single most dangerous class of bug in this pipeline and the reason
   `validate_with_pmb_parser.py` exists.

4. **11 gold documents lose an edge at parse time** because two roles on one
   concept carry the same index — e.g. `p21/d2512` "The dog barks at all
   strangers." is `bark.v.04 Agent -2 Time +1 Recipient +1`, where `Time` and
   `Recipient` both resolve to `person.n.01`. The graph is a `DiGraph`, so one
   edge is dropped. Pre-existing, not introduced here, but it caps achievable
   Smatch on those documents.

---

## 6. Recommendation

Proceed with the plan, with three changes:

1. **Drop the "parse two sentences and diff" step.** Substitute on the gold
   graph directly; the diff is known.
2. **Add the three cheap mutation operators** (constant, tense/aspect,
   edge-label + device ③) before scaling up. They roughly triple the notation
   coverage of the generated set at no notational cost, and they are the only
   way the generated data will exercise the parts of the notation the thesis
   actually argues for.
3. **Add the contextual re-rank** from LARD 2025. Reparandum plausibility, not
   graph legality, is what currently limits sample quality.

Also: write device ⑤ into the notation doc's toolkit table, and make the
strategy order POS-dependent there as well — the doc currently presents ④ as a
last resort, but for prenominal modifiers it is the *first* choice, and ⑤ is
not listed at all. **Done as of §2a:** ⑤'s eligibility is now gated on the
reparandum actually dangling, not merely on a role lacking a registered
inverse — this was applied, not just recommended, and the corpus has been
regenerated under it.

---

## Files

| file | what |
|---|---|
| `sbn_lin.py` | linear-order SBN parser/serialiser (the PMB graph parser discards the linear order indices are computed over) |
| `repair_transform.py` | the splice, three strategies, five devices, both index spaces |
| `wn_candidates.py` | reparandum pool from WordNet + SBN-writability filter + plausibility ranking |
| `inflect_en.py` | lemma + PTB tag → surface form (WordNet exception lists + regular orthography) |
| `generate_repairs.py` | end-to-end generator → TSV |
| `test_against_doc.py` | regression against the notation doc's covered cases |
| `analyse_feasibility.py` | the funnel and per-POS numbers in §3 |
| `validate_with_pmb_parser.py` | round-trip every sample through `sbn_smatch.SBNGraph` |
| `repairs_train.tsv` | 15,197 generated samples |
