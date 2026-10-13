# Can repair-aware SBN be synthesised from PMB gold? — a feasibility study

This document reports what was **measured** when the idea was tested. What we
decided to *build* as a result lives in `DESIGN.md`; this file deliberately
stops at the evidence.

Everything below is reproducible from this directory:

```
python3 test_against_doc.py          # does the transform match the notation doc?
python3 analyse_feasibility.py       # the coverage funnel
python3 validate_with_pmb_parser.py  # round-trip through PMB's own parser
python3 generate_repairs.py          # end-to-end generation
```

---

## 0. The question, and the short answer

**The question.** The Parallel Meaning Bank (PMB) train gold gives us ~9,500 English
sentences, each paired with a hand-checked semantic graph in SBN notation.
None of them contain self-repair — they are clean written sentences. We want
training data for a parser that *can* handle self-repair. Can we manufacture
it by taking a gold sentence, corrupting one word, and re-annotating the result
as a repair?

Concretely, turn this:

```
The cat hunts the rat.
cat.n.01  time.n.08 EQU now  hunt.v.01 Agent -2 Theme +1 Time -1  rat.n.01
```

into this:

```
The cat chases, actually, hunts the rat.
cat.n.01
time.n.08 EQU now
    CORRECTION <1
chase.v.01 Agent -2 Time -1
    CONJUNCTION <2                    % actually
hunt.v.01 Agent -3 Theme +1 Time -2
rat.n.01
```

**The short answer.** Yes.
The graph surgery succeds on 99.7% of the sites where it is attempted, and
91.4% of gold documents yield at least one usable site.

**But the method's real limit is not where it was expected.** It was assumed
the hard part would be the graph surgery. It isn't. The hard part is the
*mutation operator*, the thing that decides what the speaker "said by
mistake". We only built one kind (swap a content word for a WordNet
neighbour), and that one kind can only reach one of the four kinds of material
SBN can express. Sections 5 and 6 are about that boundary, and about the
quality problems the one operator we built actually produces.

---

## 1. The method, in enough detail that the rest of this document makes sense

### 1.1 Vocabulary

A self-repair has three parts. In *"The cat chases, actually, hunts the rat"*:

- **reparandum** — `chases`, the part the speaker abandons;
- **interregnum** — `actually`, the optional editing signal;
- **repair** — `hunts`, what replaces it.

In the SBN notation being developed for this thesis (see
`documentation/knowledge_base/repair_sbn_notation.qmd`, hereafter *the notation
doc*), the reparandum is quarantined inside a box opened by a `CORRECTION`
separator, and the repair goes into a sibling box opened by `CONJUNCTION`,
which merges it back into the surrounding context. Inserting that pair of
boxes into an existing graph is what this document calls **the splice**.

### 1.2 Why no second parse is needed

The original plan was: generate sentence A, generate sentence B, parse both,
and diff the two graphs to find what changed. That step turns out to be
unnecessary and should be dropped.

Because the reparandum is produced by *substituting* on a graph we already
have, the difference is known before we start: it is exactly one concept node.
No second parse happens, so no parser error can enter the data.

### 1.3 What the splice drags with it

Substitution is trivial. What is not trivial is everything that has to be
renumbered around it. SBN refers to other concepts by **relative position**:
`Agent -2` means "the concept two places back". Inserting one or two new
concepts therefore shifts every index that spans the insertion point.

There are in fact **two** independent index spaces, and the second one is easy
to miss:

1. **Concept indices** (`Agent -2`, `Theme +1`) count *all* concepts linearly.
2. **Box pointers** (`CORRECTION <1`, and box-valued roles like
   `Proposition >1`) count *boxes*. The splice adds two boxes, so every
   downstream box pointer shifts too.

Getting the second one wrong does not raise an error. It, however, silently produces a
graph that parses fine and means something else — including, in the cases we
hit, graphs containing cycles. This is the single most dangerous failure mode
in this pipeline, and the reason `validate_with_pmb_parser.py` exists.

### 1.4 The devices

SBN forbids an index from pointing *forward into* a `CORRECTION` box from
outside it (clarification **C3** of the notation doc). So after the splice,
some edges are illegal and must be rewritten. The notation doc lists four ways
to do that; the code implements a fifth:

| | Device | What it does | When |
|---|---|---|---|
| ① | **drop the role** | simply don't write the edge | the argument had not been uttered yet when the speaker abandoned the phrase — e.g. `chases` never got its object |
| ② | **invert the role** | `Theme +1` on the verb becomes `ThemeOf -1` on the object | something outside the box needs to point *into* it, and the role has a registered `…Of` form |
| ③ | **dummy concept** | introduce `entity.n.01` and hang the role on it | what is being repaired is an *edge label*, not a concept — SBN cannot put a bare role inside a box |
| ④ | **movement** | reorder so the target comes before the `CORRECTION`, making the index negative | last resort; breaks surface-order alignment and inflates index magnitudes |
| ⑤ | **anchor dummy** | put `entity.n.01` *before* the `CORRECTION`, redirect inbound edges onto it, hang both copies off it with `EQU` | the reparandum would otherwise have no edges at all — typically a bare subject |

⑤ is **not in the notation doc's toolkit table**, but the doc's own subject
self-repair example uses it. It should be written into that table.

Two findings about ⑤ that were not obvious:

**Strategy order must depend on part of speech.** `entity.n.01 EQU X` asserts
an *identity between entities*, which only reads correctly when `X` denotes a
referent. For adjectives and verbs, ④ is tried first instead, otherwise
repairing a prenominal adjective yields `entity.n.01 EQU fresh.a.01` ("an
entity that *is* freshness"), which parses but is nonsense.

**To avoid over-using ⑤, ask yourself two questions:**

1. Can *this particular inbound edge* be flipped to a registered `…Of` form?
2. Would the reparandum end up with **no edges at all** if it can't?

The lookup table only maps base role → `XXOf` role (`Attribute` →
`AttributeOf`), never the reverse. So a role that was *already* `XXOf`-shaped in
gold, e.g, `AttributeOf`, which is how gold normally attaches a prenominal
adjective to its noun, was never recognised as "already fine", and failing
question (1) was treated as automatically failing question (2). It doesn't
follow: a concept sitting *before* both new boxes can always legally retarget
its edge onto the repair copy, because that copy lives in the `CONJUNCTION`
box and C3 treats that boundary as permeable. Losing one specific inbound edge
only matters if nothing else keeps the reparandum connected.

Why this matters beyond tidiness: ⑤ is a node-level operation (a new concept
plus two `EQU` edges) standing in for what every other device does at the edge
level, and it adds three triples that a downstream parser must reproduce to
score well. It is also barely precedented — `LocationOf`, `PatientOf`,
`StimulusOf`, `DestinationOf` and `RoleOf` occur **zero** times in gold, and
while gold does contain `entity.n.01` carrying ≥2 index-valued edges (6.4% of
documents), those are overwhelmingly coordination and partitive constructions,
not "two competing readings of one referent".

After separating the two questions, anchor usage across the trial corpus fell
from **35% to 27%** (5,328 → 4,098 of 15,197 samples).

Worked example — *"I met a little girl, [correction,] boy"*, repairing the noun
under a fixed prenominal adjective. Before the fix the generator reached
straight for ⑤. After, it produces, unforced:

```
person.n.01 EQU speaker  time.n.08 TPR now  meet.v.01 Agent -2 Time -1
little.a.01 AttributeOf +2
    CORRECTION <1
girl.n.01 ThemeOf -2
    CONJUNCTION <2
boy.n.01 ThemeOf -3
```

`little.a.01` keeps its original `AttributeOf` edge (unflipped, just
renumbered) reaching forward into the permeable `CONJUNCTION` box; `girl` and
`boy` each separately invert `meet.v.01`'s `Theme`, so neither copy dangles.
No `entity.n.01` anywhere.

### 1.5 Two transform bugs found and now guarded against internally

Both are checked inside `build_repair` itself, not only in the external
validator, so a bad candidate is rejected and the strategy fallback silently
tries the next option.

**A cycle can close through a path invisible to the edge being inverted.**
*"They sell apples, oranges, eggs, and so on."* has `sell.v.01 Time +2` landing
on `apple.n.01` (almost certainly an off-by-one in gold — `Time` normally
targets a `time.n.08` node — but gold is gold), *and separately* `apple.n.01`
is reachable from `sell.v.01` via `Theme → {apple, orange, egg} set node → Sub
→ apple`. Repairing `apple` inverted the `Time` edge onto both copies, closing
a cycle. This cannot be ruled out in advance without a full reachability check,
so `build_repair` now builds the candidate's concept graph and rejects it if
`networkx` finds a cycle.

**A pre-existing separator at the splice site collided with ours.** *"Give the
book to whomever wants it."* already opens its own `CONJUNCTION` right before
`book.n.02`, as part of the sentence's universal-quantifier construction. The
sort deciding separator order at a shared position keyed off *names*, so the
pre-existing separator tied with ours and (stable sort) landed *after* it
instead of containing it — producing an empty `CORRECTION` box and a reparandum
stranded in the wrong box. Fixed by tagging our two separators by object
identity rather than by name.

---

## 2. Does the transform reproduce the notation doc's own examples?

`test_against_doc.py` feeds the transform the *clean* SBN of each hand-written
case in the notation doc and checks whether it reproduces the doc's repair
annotation.

The notation doc contains **13 covered cases**. The test file contains **9** of
them. Of those 9:

| outcome | count | which |
|---|---|---|
| reproduces the doc's own string exactly | **6** | SV sentence, verb repair, object repair, negation, donkey sentence, retracting repair |
| reproduces the doc's *topology* but not its content | **2** | subject repair, adjunct |
| not attempted | **1** | forwarding repair |

**The two partial cases both fail on the same thing: constants.** The
transform copies a concept's non-index role values verbatim onto both copies,
because it has no way to mutate them.

- *Subject repair.* The doc writes `male.n.02 Name "Josh"` in the `CORRECTION`
  box and `female.n.02 Name "Mary"` in the `CONJUNCTION` box. The transform
  produces `male.n.02 Name "Mary"` — right concept, wrong name. The test's
  expected string was edited to match, so it passes.
- *Adjunct.* The doc writes `DayOfWeek monday` against `DayOfWeek tuesday`.
  The transform cannot change `monday` to `tuesday`, so the test substitutes
  `time.n.08` for `time.n.08` — a no-op that exercises only the splice
  geometry. Both boxes end up saying `tuesday`.

**Four covered cases are not tested at all**: tense & aspect replacement,
preposition replacement, anaphora over reparandum and repair, and intra-turn
combined with cross-turn repair. A missing test is not evidence of
impossibility — see §5.2 for what each of them would actually need.

*Forwarding repair* is skipped because it is not a single-concept repair: the
speaker re-utters the subject **and** the verb, which is a span-level
operation the transform's one-position-in interface cannot express.

---

## 3. Coverage, measured

Over all 9,552 English gold training documents:

```
structural / non-lexical concepts   24655       (time.n.08, entity.n.01, person.n.01, …)
lexical concepts (n/v/a)            23232   100.0%
  not alignable to an uttered word   5346    23.0%      <- see §5.3
  aligned                           17886    77.0%
    no WordNet reparandum            1405     6.0%
    usable candidate site           16481    70.9%

splice outcome on usable sites
  legal                             16425    99.7%
  blocked                              56     0.3%
```

Blocked with all strategies exhausted: `index_overflow` 45,
`dangling_reparandum` 23, `hoist_unavailable` 50 (these overlap). The cycle
and separator-collision checks of §1.5 reject individual *strategy attempts*,
not sites — every site that hit one was rescued by a different strategy, so
neither appears in this tally.

By part of speech, with the strategy each site ended up needing:

| POS | lexical | aligned | usable | legal | inplace / anchor / hoist |
|---|---|---|---|---|---|
| n | 11781 | 10060 | 9560 | 9525 | 4576 / 4949 / 0 |
| v | 8609 | 5091 | 4229 | 4221 | 4170 / 7 / 44 |
| a | 2842 | 2735 | 2692 | 2679 | 2050 / 148 / 481 |

Anchor's share of legal splices is **31%** (5,104/16,425). Verbs barely use it
(7 sites) — unsurprising, since a verb's own arguments point forward and get
dropped by device ① rather than needing to reach backward into the box.

**Per document: 8,732 / 9,552 (91.4%) yield at least one legal repair site.**
At two samples per document the generator produced a trial corpus of **15,197
samples covering 9,076 documents (95.0%)**.

Two supporting facts the design depends on:

- **SBN concept order tracks surface word order in 98.2% of documents.** This
  is what licenses device ①: declining to write a forward role is only honest
  if the target word really does come later in the sentence.
- **Adjective satellites are a non-issue.** WordNet adjectives have no
  hypernyms, so the co-hyponym recipe returns nothing for them and scores 0 of
  2,842. The fallback (`similar_to` / `also_see` / antonym) fixes it. The
  similarity cluster is full of `.s.` satellite synsets, which SBN's synset
  pattern cannot write and which occur **zero** times in gold, so they must be
  filtered. After filtering, every adjective still has at least one usable
  candidate.

---

## 4. Validation against PMB's own parser

Every generated sample is round-tripped through `sbn_smatch.SBNGraph`, the
parser shipped with PMB.

A 2,000-sample re-validation: **1,999 parse; all 1,999 place `CORRECTION` and
`CONJUNCTION` as siblings off the same parent box; 1,999 lose no edges.** The
single failure is a pre-existing gold defect (§7, item 1), not a transform bug.
An earlier 12,000-sample run gave the same picture (11,996 parse, 11,985 lose
no edges), with all residual failures likewise traceable to gold defects.

The two transform bugs that *were* found this way are described in §1.5 and are
now caught internally before a sample is ever emitted.

---

## 5. Where the method actually stops

### 5.1 Not the dummy concept — the mutation operator

The anticipated problem was that the `entity.n.01` dummy would need complex
handling. It doesn't: it is one node and two edges, and mechanically derivable.
Once its eligibility was scoped correctly (§1.4) it fires on 31% of legal
splices, exactly the subject-position and bare-argument referents for which C3
leaves no alternative.

The splice is nearly total (99.7%). What is narrow is **what a WordNet synset
substitution can express**. SBN carries four kinds of repairable material, and
lexical substitution reaches one of them:

| Repairable material | How SBN encodes it | Reachable by synset substitution? | Gold docs containing it |
|---|---|---|---|
| content word | a concept node | **yes** | 91.4% (≥1 legal site) |
| constant | a role *value* — `Name "Josh"`, `DayOfWeek monday`, `Quantity 3` | no | 43.8% |
| tense / aspect | an operator on `time.n.08` — `TPR` / `EQU` / `TSU` | no | 97.3% |
| edge label | a role name — `Destination` → `Source` | no | 97.9% |

This is the headline finding: **the notation doc's inventory of cases is
broader than the generator's reach**, and the gap is one operator wide in three
different directions.

**The constant figure needs its composition stated, because the headline number
is misleading.** 43.8% is the union over ten value-bearing roles, and it is
dominated by one of them:

| role | docs | % of gold | occurrences |
|---|---|---|---|
| `Name` | 3264 | 34.2% | 4034 |
| `Quantity` | 994 | 10.4% | 1039 |
| `ClockTime` | 148 | 1.5% | 152 |
| `YearOfCentury` | 126 | 1.3% | 129 |
| `Title` | 67 | 0.7% | 67 |
| `MonthOfYear` | 59 | 0.6% | 59 |
| `DayOfMonth` | 47 | 0.5% | 47 |
| `Unit` | 37 | 0.4% | 39 |
| **`DayOfWeek`** | **15** | **0.2%** | **16** |

A constant-mutation operator is therefore mostly a *name-substitution*
operator. And the adjunct case the notation doc uses to motivate constant
repair — `monday → tuesday` — has a base of **15 documents** in all of gold.
The case for building the operator is that it makes the subject-repair and
adjunct cases *faithful* (§2), not that it reaches 43.8% of documents with
adjunct repairs.

### 5.2 What each unreached case would actually need

| Covered case | Generable now | What is missing |
|---|---|---|
| SV sentence, verb repair | ✓ | — |
| SVO subject repair | ✓ (device ⑤) | — |
| SVO object repair | ✓ (device ②) | — |
| Negation | ✓ | — (the repair correctly merges into the negated box) |
| Donkey sentence | ✓ | — |
| Retracting repair | ✓ | — |
| Adjunct (`monday` → `tuesday`) | ✗ | a constant-mutation operator |
| Preposition replacement | ✗ | an edge-label operator, **plus an insertion interface** |
| Tense & aspect replacement | ✗ | a tense operator, **plus multi-concept span support** |
| Forwarding repair | ✗ | multi-concept span support |
| Anaphora over reparandum+repair | ✗ | a second sentence |
| Intra-turn + cross-turn combined | ✗ | a second speaker turn |

Three of these deserve elaboration, because their difficulty is not what it
looks like.

**Preposition replacement is an insertion, not a substitution.** The notation
doc annotates *"I ran to, I mean, from the school"* with a dummy in each box:

```
    CORRECTION <1
entity.n.01 EQU -2 Destination -1
    CONJUNCTION <2
entity.n.01 EQU -3 Source -2
```

Neither `entity.n.01` corresponds to any concept in the clean graph. The
transform's interface is "replace the concept at position *i*", so there is no
site to key off. The perturbation itself needs zero semantic judgement — it is
a fixed template — but it needs a different entry point.

**Tense & aspect replacement is a two-concept repair, not a one-concept
repair.** The notation doc annotates *"She will go to, well, went to the
church"* like this:

```
female.n.02
    CORRECTION <1
time.n.08 TSU now
go.v.01 Theme -2 Time -1
    CONJUNCTION <2
time.n.08 TPR now
go.v.01 Theme -4 Time -1 Destination +1
church.n.02
```

Each box holds **two** concepts: the time node *and* the verb, because the
speaker re-utters the verb. The transform handles one concept position, so
this needs the same span machinery forwarding repair needs.

There is an alterative method that uses only one concept. It quarantines only `time.n.08` and invert
`Time` to `TimeOf`, as the adjunct case does. But it would annotate
the verb as *not* re-uttered, which contradicts the surface string. **Decision
taken: For this project, we decide to follow the notation doc.** Preserving surface-form correspondence is
what distinguishes SBN from more abstract meaning representations; giving it up
here to save engineering would undercut the thesis's own premise.

**The remaining three** (forwarding, anaphora, cross-turn) require composing
more than one concept or more than one document. Forwarding is a span
operation on a single document, so it is mechanisable with the same span
support tense/aspect needs. Anaphora and cross-turn need material PMB's
single-sentence documents do not contain at all.

### 5.3 Alignment: the ceiling is mostly the matcher, not the data

To repair a word, the generator must know which surface token a given SBN
concept corresponds to. PMB's split `.sbn` files do not record this, so
`generate_repairs.align()` guesses, by matching the concept's lemma and coarse
POS against the sentence's tokens. Concepts it cannot match are skipped
entirely, so they can never be repaired. The funnel in §3 reports that loss as
**23.0%**.

**PMB does ship the missing information, for part of its data.**
`data/pmb-5.1.0/src/ccg/standard/` contains CCG derivation trees for 1,132
documents, and every leaf carries the annotator's own synset assignment plus
character offsets:

```
t(n, 'hair', [lemma:'hair', from:15, to:19, pos:'NN', sem:'CON', wordnet:'hair.n.01'])
```

That is ground-truth token↔concept alignment. Under PMB 5.1.0's split these
1,132 documents fall as: train/gold 283, dev 303, test 522, test/long 1.

Measured against that ground truth on the 283 documents that are in
train/gold (679 lexical concepts):

```
no CCG counterpart — genuinely implicit concepts   39    5.7%
has a CCG counterpart                             640   94.3%
  heuristic agreed                                574   89.7% of alignable
  heuristic picked the wrong token                  3    0.5%
  heuristic found nothing                          63    9.8%
```

So of the alignment loss on this subset, roughly **two thirds is matcher
failure and one third is the data**. The earlier claim that this ceiling "is a
property of the data source, not the method" does not hold.

The failures are also not scattered. They are almost all **named entities**:

```
Japan      → state.n.04      "Japan has been sending athletes to the Olympics since 1912."
Scotland   → country.n.02    "What continent is Scotland in?"
Hawaii     → island.n.01     "The Macdonough left Hawaii on 10 August."
Macdonough → vehicle.n.01
Indiana    → team.n.01       "Who is the coach of Indiana?"
```

The reason is visible in the SBN:

```
state.n.04 Name "Japan" time.n.08 TPR now send.v.01 Agent -2 Time -1 …
```

PMB gives a proper name the synset of its **hypernym** and puts the name itself
in a `Name` constant. The matcher looks for the token `state` and never finds
it.

Two consequences worth separating:

- **The fix does not need the CCG files.** The string in `Name "Japan"` *is*
  the surface form; matching on it is near-exact and works on all 9,552
  documents, not just 1,132.
- **The CCG files are valuable as a yardstick.** They let the matcher's error
  rate be measured before and after, turning an unaccountable 23% into a
  decomposed, checkable number.

One caveat on generalising the measurement: these 283 documents are PMB 5.0.0's
standard test set redistributed by the 5.1.0 split, so they may not be
representative of all 9,552. The transferable claim is the *ratio* — most of
the loss is fixable — not the absolute rate.

---

## 6. Quality defects measured in the trial corpus

§§3–5 concern what the method *can* reach. This section concerns how good the
15,197 samples it did produce actually are. These are trial-run defects, not
properties of the approach.

Each subsection states the defect as it was measured on the trial corpus, so
the record stays comparable. §6.6 gives the current status: four of the six
have since been fixed and re-measured. The two that remain — the candidate
pool (§6.5) and the corpus composition (§6.4) — are the ones that need a
design change rather than a code fix.

### 6.1 8.0% of samples are unlearnable: the reparandum and repair are the same word

1,217 samples (8.0%) pair two senses of the *same* lemma, so the two surface
forms are identical:

```
run.v.29  → run.v.01    "He ran, actually, ran five miles."
die.v.02  → die.v.01    "He died, died of a heart attack."
move.v.02 → move.v.01   "The train was moving, no wait, was moving at 500 miles per hour."
```

The graph faithfully quarantines a real sense distinction; the input string
contains no information from which that distinction could be recovered. Worse,
the sample teaches a parser that a repeated identical word should open a
`CORRECTION` box — which is precisely the "pure repetition" case the notation
excludes, since there reparandum and repair are the same concept and the graph
has nothing to quarantine.

**Root cause**, in `wn_candidates.py:80-82`: for verbs, the top-ranked
candidate relation is WordNet's *verb group*, which by definition links
near-synonymous senses **of the same verb**. It was introduced to fix a real
problem — WordNet's verb hierarchy is shallow, so co-hyponyms of a verb are
often absurd (`bear.v.02` sits directly under `produce.v.01`, giving
`sporulate` and `grind_out`) — and it does improve plausibility. It just
improves it past the point of usefulness: the tightest neighbour of a verb
sense is another sense of the same verb, whose surface form is identical.

Adjectives have the same exposure through `similar_tos()`, less often. Nouns
are safe, since co-hyponyms always differ in lemma.

The fix is one guard: reject candidates whose lemma equals the reference lemma.
It was applied, and the defect measures 0.0% afterwards (§6.6).

**But the guard treats a symptom, not the cause, and §6.5 shows the cause is
still active.** Identical surface forms are the extreme end of a scale, and the
ranking that produced them — "prefer the tightest relation" — is unchanged. All
the guard does is move the top-ranked candidate one notch along that scale,
from *the same verb* to *a near-synonym of it*.

### 6.2 27.6% of samples use an editing signal that contradicts the graph

The interregnum list used for the trial run contained ten strings. Three of
them — `that is`, `or rather`, `in fact` — are **reformulation** markers, not
correction markers: they signal "let me put that another way", where both
formulations stand and the second glosses the first. 4,198 samples (27.6%) use
one.

Combined with a semantically close pair, the result reads as apposition, and
the string contradicts the graph:

```
original.a.03 → new.a.02     "We added something original, that is, new."
paroxysm.n.01 → heart_attack.n.01   "He died of a paroxysm, that is, heart attack."
composed.a.01 → cool.a.02    "I was as composed, that is, cool as a cucumber."
```

`CORRECTION` asserts that the reparandum is *withdrawn*. "X, that is, Y" says
the opposite.

Note this is the same underlying problem as §6.1, at a different point on one
axis. As the semantic distance between reparandum and repair shrinks, the
sample degrades from "clear correction" through "reads as a gloss" to
"indistinguishable strings". §6.1 is the limit of that axis, not a separate
bug.

**This has a direct consequence for how candidates should be selected.** LARD
(Passali et al.) — the method this generator is modelled on — selects, from all
candidates, the one whose sentence has the **highest** cosine similarity to the
original (their Alg. 2, step 5, using `multi-qa-distilbert-cos-v1`). For their
task, disfluency *detection*, that is correct: they want the disfluent input to
stay coherent. For ours it is actively harmful, because maximising similarity
drives straight toward near-synonymy, which is where both §6.1 and §6.2 live.
What is needed is a **band**: far enough apart that the correction has content,
close enough that it is a plausible production slip.

### 6.3 3.8% of samples have a capitalisation artefact

570 samples read like this:

```
The convention, I mean, The peace treaty will be signed tomorrow.
This anteroom, sorry, This classroom can accommodate only thirty students.
```

`make_nl()` capitalises the repair when the repair site is sentence-initial,
but after the splice the repair is no longer sentence-initial — the reparandum
is.

A further 23 samples (0.2%) carry a broken inflection (`adversaryest`,
`strivingest`); these are already flagged by the `inflection_confident` column
and can be filtered.

### 6.4 The corpus composition was not chosen

All 15,197 samples are single-concept content-word substitutions — there was
only one mutation operator. What varies is which device the graph topology
happened to force:

```
device ① drop role         36.8%      no device (bare SV shape)   10.8%
device ⑤ anchor dummy      26.5%      device ④ movement            2.4%
device ② invert role       13.7%      repair inside a NEGATION box  8.9%
```

These proportions are not a design decision; they are PMB's own noun/verb/
adjective distribution propagating through a fixed rule (subject → ⑤, object →
②, verb → ①, adjective → ④). Within a single mutation operator, "which case
of the notation is this" is not an independent dimension that can be sampled.

Separately, **69.6% of samples carry an interregnum** and only 30.4% do not.
That inverts the priority the pilot study argues for: the pilot's ablation
found detection of interregnum-marked repair at 100% across all positions,
versus 30.7–64.2% without a marker. A corpus that is 69.6% marked is mostly
teaching the easy case.

### 6.5 Semantic plausibility of the reparandum

This is the deepest of the quality problems, and the one whose diagnosis
changed most on measurement. It is inherited from LARD rather than caused by
SBN.

**A good reparandum is a sibling in a contrast set**, not a synonym and not a
random distant word: the notation doc's own examples are `banana_bread` /
`cherry_pie`, `girl` / `boy`, `monday` / `tuesday`, `beat` / `feed`. In each
pair the two options are incompatible values of the same slot — which is what
makes correcting one into the other worth doing.

#### The candidate ranker drives away from that, by design

`wn_candidates._plausibility` sorts by relation tightness first. For verbs the
tightest relation is WordNet's **verb group**; for adjectives it is the
**`similar_to` cluster**. Both are *defined* by WordNet as near-synonymy. They
are ranked first, and `generate_repairs` takes `cands[0]`.

Measured over 3,125 samples generated after the §6.1 guard was in place:

```
reparandum came from a co-hyponym relation        76.4%
reparandum came from a WordNet near-synonym relation  14.8%   (verbs 23%, adjectives 45%, nouns 0%)
other (adjective antonyms, etc.)                   8.8%
```

Human judgement on the resulting verb and adjective samples is that they are
not believable repairs: `watch.v.03 → see.v.01` ("We watched, saw the bird"),
`say.v.09 → tell.v.01`, `make.v.38 → cook.v.02`, `wary.a.01 → shy.a.03`. Each
of those pairs is linked by exactly the near-synonym relation above. Noun
samples, which never use it, are structurally sound (`bay.n.01 → lake.n.01`).

#### Deleting the near-synonym relations is not enough on its own

The first measurement of the cost looked reassuring — over distinct gold
synsets that had any candidate at all, almost none lost every candidate:

```
nouns       2849 -> 2849   (100%)   never used the relation
adjectives   849 ->  849   (100%)   antonyms remain
verbs       1619 -> 1554    (96%)   65 lose every candidate
```

That measurement asked the wrong question. "Does at least one candidate
survive" is not "is the surviving pool usable". Measured properly — coverage
over *all* distinct gold synsets of that part of speech, and the size of the
pool that remains — deletion alone leaves two of the three parts of speech
worse off than before:

```
             coverage   median pool
nouns          95.6%        10        fine; co-hyponymy was always the pool
verbs          91.3%*       20*       * only after adding VerbNet, below
adjectives     40.1%         1        antonym only -- unusable
```

Each part of speech needed a different remedy, and the two sections below give
them. Verbs are the harder case, so they come first.

#### Verbs: WordNet fails in both directions

Coverage is not the problem for verbs. **Quality is.** What the top candidate
becomes once verb groups are removed:

```
see.v.01     watch.v.03    ->  apperceive.v.01
write.v.01   publish.v.03  ->  rhyme.v.01
die.v.01     fail.v.04     ->  relax.v.01
call.v.01    address.v.06  ->  stigmatize.v.01
```

This is the shallow-hierarchy problem that motivated promoting verb groups in
the first place (`bear.v.02` sits directly under `produce.v.01`, giving
`sporulate` and `grind_out`). So the two relations WordNet offers for verbs
fail in **opposite** directions, with nothing in between:

| relation | failure |
|---|---|
| verb group | too close — a near-synonym; correcting it would be pedantic |
| co-hyponym | too far — the hierarchy is shallow, so siblings are absurd |

**WordNet alone cannot supply believable verb reparanda.** This is a property
of the resource, not of the ranking, and no re-ranking fixes it: a re-ranker
can only choose among what the pool contains.

#### VerbNet supplies the missing middle

VerbNet groups verbs by syntactic alternation behaviour (Levin classes), so
class-mates share an argument frame while differing in meaning. Its members
carry WordNet sense keys, so the mapping from a gold synset to a class is exact
rather than lemma-guessed. (One format detail: VerbNet writes `give%2:40:03`
where NLTK expects `give%2:40:03::`.)

For the verbs WordNet handled worst:

```
see.v.01    detect, examine, feel, find, hear, ...    "I heard, I mean, saw him"
eat.v.01    drink, feed, toast                        "I ate, no, drank it"
call.v.01   appoint, baptize, crown, dub, name
```

`see → hear` is the shape that was missing: same perception frame, same
argument structure, plainly different meaning.

There is a further reason this fits *this* project specifically. **PMB's own
edge labels are VerbNet thematic roles** — the CCG derivations annotate each
verb with `verbnet:['Theme','Pivot']`, and `Agent` / `Theme` / `Pivot` /
`Co-Theme` in SBN are VerbNet roles. A class-mate substitution is therefore
aligned with the annotation scheme, and has a structural reason to keep the
gold role set valid; a WordNet co-hyponym has none. (This is a structural
argument, not yet a measurement — the role-set preservation rate is testable
and worth testing.)

Two limits, measured over the 1,927 distinct gold verb synsets:

- **Coverage is 61.2%** (1,179 have at least one class-mate), *below* WordNet's
  ~84%. So VerbNet should be unioned with WordNet co-hyponyms, not substituted
  for them. `play.v.01` has no class-mate at all.
- **The pool is wide and unordered**: median 42 candidates, mean 86, maximum
  585. It contains `see → hear` and `write → calculate` side by side.

#### Adjectives: WordNet offers no contrast relation at all

Antonymy leaves adjectives at 40.1% coverage with a pool of exactly 1. Nothing
else in WordNet fills the gap, and the reason is structural rather than
incidental.

SBN can only write WordNet **head** adjectives (`.a.`): `SYNSET_PATTERN`
rejects satellites (`.s.`), and gold contains zero of them. Head adjectives
relate to other heads by exactly two relations, and both are unusable:

```
similar_to   returns satellites only -- measured as 0 writable candidates for
             every adjective tested (new, little, fresh, wary, hot).  Widening
             through it, from the concept or from its antonym, yields nothing.
also_see     returns writable heads, but near-synonyms: new -> fresh / modern
             / current, hot -> warm.  That is the relation being rejected.
```

So the choice is between near-synonyms and the two poles of a scale. Neither
is a contrast set.

#### FrameNet supplies the dimension, not just its endpoints

An adjective's FrameNet frame **is an attribute dimension** — `Size`, `Age`,
`Temperature`, `Chemical-sense_description` — and the frame's adjective lexical
units are the values on that dimension. That is the contrast set directly:

```
spicy  -> frame Chemical-sense_description
          bitter, bland, delicious, flavourless, fragrant, hot, insipid,
          piquant, pungent, salty, savory, smelly, sour, sweet, tart
```

`spicy → sour`, `spicy → bland`, `spicy → salty`: alternatives on one
dimension, all of them predicable of the same noun. Antonymy would have
returned only the endpoint.

Measured over the 876 distinct gold adjective synsets:

```
antonym only                    40.1%   pool median 1
FrameNet frame-mates            59.0%   pool median 8
union, after writability and same-lemma filtering
                                65.6%   pool median 6
```

Two limits:

- **35% still have no pool.** The misses are adjectives that do not name a
  scalar dimension: `abandoned`, `absent`, `absolute`, `adjacent`, `aged`,
  `according`.
- **Frame-mates are one-dimensional.** A cross-dimension repair — *"the curry
  is too greasy, spicy"*, swapping texture for taste — is not reachable this
  way, since the two adjectives evoke different frames. `greasy` is not a
  FrameNet lexical unit at all. Reaching those would need contextual
  generation conditioned on the modified noun, which the graph supplies via
  `AttributeOf`.

#### The three routes are one criterion

Stated as a lexical recipe, the pool looks like three unrelated hacks. It is
not. A contrast set is **one slot, different values**; the three parts of
speech differ only in what defines the slot:

| POS | the slot is defined by | resource |
|---|---|---|
| noun | a position in a taxonomy | WordNet co-hyponymy |
| verb | an argument frame | VerbNet class |
| adjective | an attribute dimension | FrameNet frame |

None of the three is a *similarity* relation. That is why ranking candidates by
similarity was the wrong instrument from the start, and why the adjective case
exposed it: adjectives are the one part of speech where no similarity relation
even approximates the criterion, so the error could not stay hidden.

The criterion behind all three is **selectional compatibility with the shared
argument** — both values must be predicable of the same thing. Co-hyponymy
approximates it because taxonomic siblings inherit selectional restrictions;
VerbNet classes encode it directly as a shared argument frame; FrameNet frames
encode it as a shared dimension. This also predicts where each route fails: at
the point where the approximation and the criterion come apart, as in
`tall.a.01 → osseous.a.01`, two adjectives in one frame that are not on one
comparable scale.

#### What follows

The pool and the selector are separate problems and both were wrong:

1. **The pool** lacked a contrast relation for verbs and adjectives. VerbNet
   and FrameNet supply them. Coverage after the change: nouns 95.6% (median
   pool 10), verbs 91.3% (20), adjectives 65.6% (6).
2. **The selector** — ranking by tightness and taking the top — actively seeks
   the bad end. Its job should be to hand a *wide* pool to the filter, not to
   pre-select from it.

Fixing the pool moves the bottleneck rather than removing it. Every pool still
contains near-synonyms alongside good candidates — `good.a.01`'s pool opens
with `effective`, `stuck.a.01`'s with `affixed` — and with no filter in place
the generator takes the first entry. Measured on a 3,144-sample batch generated
after the change, the visible defects are now all *selection* defects:

```
effective.a.01 -> good.a.01    "My brother is effective, is good at playing tennis."
affixed.a.01   -> stuck.a.01   "This drawer's affixed, no wait, stuck."
osseous.a.01   -> tall.a.01    "Is he osseous, actually, taller than his brother?"
```

LARD 2025's sentence-encoder re-rank is still needed, subject to the direction
correction of §6.2 (a band, not a maximum). Its role is now clearer: for nouns
it is polish; for verbs and adjectives it is load-bearing, because with pools
of 6 to 20 nothing else decides which candidate is used.

#### Morphological negation pairs: kept, and monitored

8.5% of adjective samples pair a word with its own affixal negation:
`unqualified → qualified`, `impatient → patient`, `incomplete → complete`,
`untidy → tidy`. These arrive through antonymy, which is where WordNet records
them.

They are linguistically defensible — *"she's unqualified, no, qualified"* is a
real repair — but the surface difference is one prefix, which puts them near
the same-lemma failure of §6.1: a model could learn the affix as a shortcut
rather than the construction. **Decision: keep them, and track the share.** At
8.5% of adjective samples they are a minority of a minority; if a later batch
pushes that materially higher, or if error analysis shows the model keying on
the affix, the decision should be revisited.

### 6.6 Status after the first two rounds of fixes

Round one fixed four defects and the alignment matcher of §5.3. Round two
rebuilt the candidate pool of §6.5. Both were re-measured on fresh batches of
about 3,100 samples (2,000 source documents, 2 samples each).

| measure | trial corpus | after |
|---|---|---|
| same-lemma reparandum/repair (§6.1) | 8.01% | **0.00%** |
| mid-sentence capitalisation (§6.3) | 3.90% | **0.00%** |
| reformulation-marker interregna (§6.2) | 27.6% | **0** — the set is now five correction markers |
| samples carrying an interregnum (§6.4) | 69.6% | **31.2%** |
| alignment agreement vs CCG truth (§5.3) | 89.7% | **95.5%** |
| noun pool coverage / median size (§6.5) | 95.6% / 10 | unchanged |
| verb pool coverage / median size (§6.5) | ~84% / ≤10, near-synonym-first | **91.3% / 20** |
| adjective pool coverage / median size (§6.5) | ~100% / ≤10, near-synonym-first | **65.6% / 6** |
| `test_against_doc.py` | 8 pass / 1 skip | unchanged |
| PMB parser validation, 2,000 samples | 1,999 parse | 1,999 parse |

Two of those rows are not improvements on their face and should be read
together with §6.5. Adjective coverage *fell*, from a pool that was wide but
led with near-synonyms to one that is narrower and leads with genuine
contrasts. The trade was taken deliberately: a sample the model cannot learn
from is worse than no sample.

**New dependencies**, both small NLTK corpora, both now required rather than
optional (the code fails loudly if either is missing, because their absence
degrades the pool silently):

```
python3 -c "import nltk; nltk.download('verbnet')"        # 429 classes
python3 -c "import nltk; nltk.download('framenet_v17')"   # 1,221 frames, 2,392 adjective LUs
```

On alignment, the residue splits as: 27 concepts the matcher still fails to
locate (was 63) and 2 where it picks the wrong token (was 3). The `Name`
matcher of §5.3 accounts for the difference.

**One methodological point is worth recording, because it cost a wrong
intermediate result.** An earlier version of the `Name` matcher fed the newly
aligned concepts into WordNet substitution, producing

```
"Adversary, Hitler assumed power in 1933."       (male.n.02 -> adversary.n.01)
```

The graph is legal, the parser accepts it, and all five rule checks of the
quality pipeline pass. The defect is visible only by reading the sentence.
Those concepts write their surface form into a `Name` constant, so substituting
the *synset* puts the hypernym's lemma into the sentence; they belong to the
constant operator, not to lexical substitution. The fix is to keep the two
alignments in separate maps.

The general lesson: **graph-level validation cannot detect a whole class of
defect here**, which is the reason the quality design separates a rule layer
from a language layer — and the reason every iteration has to include reading
actual samples, not only checking metrics.

Two pre-existing inflection defects surfaced the same way and are not yet
fixed:

```
sew.v.01     -> "Tom sewn, I mean, stuck ..."      past tense taken as past participle ("sewed")
big_cat.n.01 -> "allergic to bigs cat"             multiword lemma pluralised on the first word
```

---

## 7. Defects in PMB's own toolchain, found along the way

These affect the evaluation pipeline as much as the generator.

1. **`CauserOf` is attested in gold but missing from `SBNSpec.ROLES`.**
   `p36/d2375` and `p30/d2358` abort the stock parser with "Invalid token
   found". One line to fix in `sbn_spec.py`.

2. **`INDEX_PATTERN` matches a single digit**, so `-10` is silently read as
   `-1`. The splice inflates index magnitudes, so this is a live constraint,
   not a theoretical one: 45 sites are blocked by it, and 39 generated samples
   sit at magnitude 9 — one concept away from silent corruption. The generator
   records `max_abs_index` per row so these can be filtered.

3. **Box pointers are a second index space** (§1.3). 16.2% of gold documents
   contain a separator; 1.9% contain a box-valued role (`Proposition >1` ×172,
   `Proposition <1` ×12, `Proposition <2` ×1). Before this was handled, the
   splice silently produced
   cyclic graphs and mis-targeted downstream pointers, both of which parse
   without error.

4. **11 gold documents lose an edge at parse time**, because two roles on one
   concept carry the same index. `p21/d2512` *"The dog barks at all
   strangers."* is `bark.v.04 Agent -2 Time +1 Recipient +1`, where `Time` and
   `Recipient` both resolve to `person.n.01`. The graph is a `DiGraph`, so one
   edge is dropped. Pre-existing, but it caps achievable Smatch on those
   documents.

---

## 8. What follows from all this

The findings that carry forward, in the order they constrain the design:

1. **Drop the "parse two sentences and diff" step.** Substitute on the gold
   graph directly; the difference is known (§1.2).
2. **The splice is not the bottleneck; the mutation operator is** (§5.1). Three
   more operators — constant, tense/aspect, edge label — are what would let the
   generated data exercise the parts of the notation the thesis argues for.
   None of them requires any notational change.
3. **Two of those three need new interfaces, not just new value pools** (§5.2):
   an insertion entry point for device ③, and multi-concept span support for
   tense/aspect.
4. **Fix the corpus defects of §6 before regenerating at scale.** Same-lemma
   candidates, reformulation markers, capitalisation, and the interregnum
   ratio together affected roughly a third of the trial samples. **Done and
   re-measured — see §6.6.**
5. **Fix alignment via `Name` constants** (§5.3), and use the CCG derivations
   to measure the improvement rather than to perform it. **Done: 89.7% →
   95.5% (§6.6).** Keep the two alignments in separate maps; a name-bearing
   concept is not a lexical substitution site.
6. **The candidate pool is a resource problem before it is a ranking problem**
   (§6.5). A reparandum must be a *contrast set sibling* — one slot, a
   different value — and no single resource supplies that relation for all
   three parts of speech. The slot is a taxonomic position for nouns
   (WordNet co-hyponymy), an argument frame for verbs (VerbNet classes), and
   an attribute dimension for adjectives (FrameNet frames). None of the three
   is a similarity relation, which is why similarity ranking was the wrong
   instrument. Union each new resource with the WordNet pool rather than
   replacing it.
7. **Candidate selection needs a similarity band, not a maximum** (§6.2), and
   the ranker must stop pre-selecting from the pool (§6.5). The band's
   thresholds have no ground truth and will need calibrating against a small
   hand-labelled sample.
8. **Graph validation cannot catch every defect** (§6.6). A sample can pass
   every rule check and still be nonsense. Reading samples is part of each
   iteration, not an optional final step.

Also: **write device ⑤ into the notation doc's toolkit table**, and record
there that strategy order is POS-dependent — the doc presents ④ as a last
resort, but for prenominal modifiers it is the first choice, and ⑤ is not
listed at all.

Decisions taken on the basis of these findings are recorded in `DESIGN.md`.

---

## Files

| file | what |
|---|---|
| `sbn_lin.py` | linear-order SBN parser/serialiser (PMB's own graph parser discards the linear order that indices are computed over) |
| `repair_transform.py` | the splice: three strategies, five devices, both index spaces |
| `wn_candidates.py` | reparandum pool from WordNet + SBN-writability filter + plausibility ranking |
| `inflect_en.py` | lemma + PTB tag → surface form (WordNet exception lists + regular orthography) |
| `generate_repairs.py` | end-to-end generator → TSV |
| `test_against_doc.py` | regression against the notation doc's covered cases (§2) |
| `analyse_feasibility.py` | the funnel and per-POS numbers of §3 |
| `validate_with_pmb_parser.py` | round-trips every sample through `sbn_smatch.SBNGraph` (§4) |
| `nonfinite_probe.py` | whether non-finite verb constructions survive a splice |
| `repairs_train.tsv` | the 15,197-sample trial corpus — **experimental output, not a release**; see §6 |

---

## Appendix: corrections to the first version of this document

The first version of this file made four claims that later measurement
contradicted. They are corrected in place above; recorded here because the
first version was committed and may have been read.

| claim in v1 | status |
|---|---|
| "8 of 9 covered cases reproduce the doc character for character" | **Overstated.** 6 reproduce exactly; 2 (subject, adjunct) reproduce only the topology, because their expected strings were edited to keep constants unchanged. §2 |
| "23% alignment loss … is a property of the data source, not the method" | **Wrong.** Measured against PMB's CCG derivations, about two thirds of the loss is matcher failure, dominated by named entities. §5.3 |
| Tense/aspect repair is "structurally identical to a verb repair" and a "cheap extension" | **Wrong.** The notation doc's own annotation puts two concepts in each box. It needs multi-concept span support — the largest single piece of new machinery in the plan. §5.2 |
| "constant … reaches 47.1% of documents" | **Imprecise, and misleading without composition.** The reproducible figure over ten value-bearing roles is 43.8%, of which `Name` alone is 34.2%. `DayOfWeek`, the role the motivating example uses, occurs in 15 documents. §5.1 |

Two defect classes measured after the first version, now in §6: same-lemma
reparandum/repair pairs (8.0% of the trial corpus) and reformulation-marker
interregna (27.6%).
