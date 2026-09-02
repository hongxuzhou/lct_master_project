# Design of the repair-aware SBN dataset

This document records **what we decided to build and why**. The evidence behind
the decisions is in `FINDINGS.md`; where a decision rests on a specific
measurement, the section is cited rather than the number repeated.

Implementation follows §9, which marks what is done. Section 8 lists what is
still undecided.

---

## 0. What we are building, and what for

We need training and evaluation data for a semantic parser that can handle
**self-repair** — the everyday phenomenon of a speaker abandoning what they
were saying and restarting:

> The cat chases, actually, hunts the rat.

The Parallel Meaning Bank (PMB) gives us roughly 9,500 English sentences paired
with hand-checked semantic graphs in SBN notation, but every one of them is a
clean written sentence. No self-repair appears anywhere in it. So the data has
to be manufactured: take a gold sentence, corrupt one part of it to play the
role of the abandoned attempt, and re-annotate the result using the
`CORRECTION` / `CONJUNCTION` construction developed in
`documentation/knowledge_base/repair_sbn_notation.qmd` (hereafter *the notation
doc*).

The feasibility of the graph surgery is settled: it succeeds on 99.7% of the
sites where it is attempted (`FINDINGS.md` §3). This document is about
everything that surrounds it — which cases to generate, in what proportion,
how to tell a good sample from a bad one, and what the finished file looks
like.

### The organising principle: coverage, not frequency

A synthetic corpus has no natural distribution to imitate. PMB is written text;
the true frequency of self-repair in it is zero. Any corpus we build is skewed
by construction, so the only real question is **whether we chose the skew**.

We choose it by **coverage of the notation's case inventory**: every
construction the notation licenses should appear in the data in learnable
quantity. This is deliberately not frequency-matching to spontaneous speech,
for two reasons. First, there is no frequency to match against in this source.
Second, the pilot study's findings (`colloquium_prep/pilot_eval/`) show that a
parser's failure modes cluster by *structural configuration*, not by how often
a configuration occurs — so the data has to contain each configuration, not a
realistic mixture of them.

One consequence of adopting coverage as the goal: a "coverage ceiling" — how
many gold documents *could* host a given kind of repair — only constrains the
design when it falls below a single cell's target. A ceiling of 34% of
documents is not a skew problem if the target for that cell is 800 samples.
A ceiling of 15 documents is (see §3, tier 3).

---

## 1. The pipeline

Six steps, in this order. Each depends on the output of the one before it.

### Step 1 — Select source documents

Take PMB gold data, and use **PMB's own train/dev/test split** rather than
re-splitting:

| our split | source file | documents |
|---|---|---|
| train | `data/pmb-5.1.0/split/en/train/gold.sbn` | 9,552 |
| dev | `data/pmb-5.1.0/split/en/dev/standard.sbn` | 1,194 |
| test | `data/pmb-5.1.0/split/en/test/standard.sbn` | 1,194 |
| test (long) | `data/pmb-5.1.0/split/en/test/long.sbn` | 30 |

**Why not re-split.** The parser we will be evaluating was fine-tuned on PMB
gold train. If the repair test set were also derived from train, the model
would have seen those sentences and their clean graphs already, and the score
would partly measure memorisation. Deriving dev and test from PMB's own dev and
test keeps that out.

`test/long.sbn` (30 documents, mean 33.2 words and up to 75, against 5.3 for
`test/standard.sbn`) is kept as a separate long-sentence probe rather than
merged into the main test set. Long sentences inflate index magnitudes, so
these are also where the single-digit index limit of §5.1 bites hardest.

### Step 2 — Locate the repair site

To repair a word, the generator must know which surface token a given SBN
concept corresponds to. PMB's split files do not record this, so the current
code guesses by matching lemma and coarse part of speech, and skips whatever it
cannot match — meaning those positions can never be repaired at all.

**Decision: add a `Name`-constant matcher before the lemma matcher.** The
dominant failure class is proper names, where PMB assigns the concept its
hypernym synset and puts the name in a constant: `state.n.04 Name "Japan"`. The
matcher looks for the token `state` and never finds it. But the string inside
`Name "…"` *is* the surface form, so matching on it directly is near-exact, and
it works on all 9,552 documents (`FINDINGS.md` §5.3).

**Decision: use PMB's CCG derivations as a yardstick, not as the fix.**
`data/pmb-5.1.0/src/ccg/standard/` holds ground-truth token↔concept alignment
for 1,132 documents. That is enough to measure the matcher's error rate before
and after the fix, which turns an unaccountable 23% into a decomposed figure we
can report. It is not enough to *be* the fix, since it covers only about 9.5%
of gold.

### Step 3 — Generate base repairs

A **base repair** is a repair with no interregnum: *"The cat chases, hunts the
rat."* This layer is generated first and in full.

**Why base first.** The pilot's ablation found that when an editing signal like
"I mean" is present, a model detects the repair 100% of the time regardless of
where it sits; without one, detection falls to 30.7–64.2%. The marked case is
the easy case. The trial corpus was 69.6% marked (`FINDINGS.md` §6.4), which
inverts this priority. Generating the unmarked layer first and completely makes
the marked layer an addition rather than a majority.

Generation is **over-produced**: for each cell of the sampling grid we generate
more candidates than the target, so that filtering has something to discard.
This follows LARD's practice, but not LARD's selection rule — see §5.2. How
much surplus, and in what units, is §4.7.

### Step 4 — Sample to quota

Draw from the over-produced pool according to a quota grid. The grid itself,
and the reasoning that fixes its numbers, is §4; this step only records where
it sits in the pipeline and one property of the grid that constrains what can
be sampled at all.

One honest limitation to record here. Within concept substitution, "which case
of the notation is this" is **not** an independently controllable dimension:
the device the graph ends up needing is forced by the syntax (subject → anchor
dummy, object → role inversion, verb → dropped argument, adjective →
movement). The trial corpus's device distribution was therefore PMB's own
noun/verb/adjective mix propagating through a fixed rule, not a design choice
(`FINDINGS.md` §6.4).

What *can* be controlled: the generator knows, before emitting a sample, which
device a given site would require. So rare devices can be **deliberately
oversampled** by preferring sites that force them — device ④ (movement)
occurred in only 2.4% of the trial corpus and can be raised this way. This is
site selection, not case invention.

**What is explicitly not a sampling axis: the position of the reparandum in the
sentence.** The pilot injected repairs at head, middle and tail, and found
large differences between them. That effect is an artefact of the injection
procedure rather than a property of self-repair: a head insertion happens to
read like a vocative, a middle one like asyndetic coordination, a tail one like
apposition. Stratifying generation by position would (a) quota on an artefact
and (b) hand the model a surface cue — "comma near the start means repair" —
that lets it bypass the structural relation it is supposed to learn.

### Step 5 — Insert interregna

Take a proportion of the sampled base repairs and add an editing signal.

**The interregnum set is five markers**, deliberately small and free of
ambiguity:

```
I mean · no · no wait · sorry · actually
```

Removed from the trial run's ten-item list, with reasons:

| removed | why |
|---|---|
| `that is`, `or rather`, `in fact` | **Reformulation** markers, not correction markers. They mean "let me put that another way", where both formulations stand and the second glosses the first — the opposite of what `CORRECTION` asserts. With a semantically close pair they produce apposition: *"We added something original, that is, new."* (`FINDINGS.md` §6.2) |
| `well` | Primarily a hesitation or buffer. *"X, well, Y"* can be a correction or just a pause before continuing. Ambiguous. |
| `I mean to say` | Not ambiguous, just unnatural — nobody says this in speech, and it is a verbose variant of `I mean`. |

`actually` sits closest to the boundary of the ones we kept: it can be a plain
intensifier elsewhere, though in the *"X, actually, Y"* slot it reads reliably
as a correction. **Kept by explicit decision.**

Five markers rather than one also answers a limitation the pilot flagged about
itself: that study used `I mean` as its only marker, so "the model detects
interregna" may have meant no more than "the model recognises those two words".

**Note that this step changes the graph, not only the sentence.** The
interregnum is written into the comment field on the `CONJUNCTION` line. A base
repair and its marked counterpart are therefore **two separate samples**, not
two renderings of one. This is useful rather than awkward: it gives matched
pairs for the same within-item comparison the pilot ran.

### Step 6 — Merge hand-written cases, then write out

Three notation cases cannot be produced from single PMB documents at all (§3,
tier 4). They are written by hand and merged in.

**Split by source document, never by row.** One gold document yields several
samples; splitting by row would put the same clean graph in both train and
test.

---

## 2. How each kind of repair is generated

The design turns on one question asked before any code runs: **what kind of
material is being repaired?** SBN can express four kinds, and each needs its
own perturbation rule.

| Material | How SBN writes it | Perturbation | Gold docs containing it |
|---|---|---|---|
| content word | a concept node, `movie.n.01` | swap for a lexical neighbour — see §2.1 for which resource, per part of speech | 91.4% |
| constant | a role *value*, `Name "Josh"`, `DayOfWeek monday`, `Quantity 3` | swap for another value from a closed pool (name list, weekday list, numeral arithmetic) | 43.8% |
| tense / aspect | an operator on `time.n.08`, `TPR` / `EQU` / `TSU` | swap the operator, and re-inflect the verb to match | 97.3% |
| edge label | a role name, `Destination` vs `Source` | swap the role name, and carry it on an inserted `entity.n.01` dummy (device ③) | 97.9% |

Only the first exists today, which is why all 15,197 trial samples are content
word substitutions wearing eight different syntactic costumes.

**The splice is shared.** Whichever material is perturbed, inserting the
`CORRECTION` / `CONJUNCTION` box pair and renumbering both index spaces is the
same code, and it already works. Adding an operator means writing a rule for
"which pool does the replacement value come from", not rebuilding the core.

### 2.1 Where content-word candidates come from, by part of speech

A believable reparandum is a **sibling in a contrast set**: the same slot, an
incompatible value. The notation doc's own pairs are of this kind —
`banana_bread` / `cherry_pie`, `girl` / `boy`, `beat` / `feed`. No single
lexical resource supplies that relation for all three parts of speech, so the
pool is assembled differently for each (`FINDINGS.md` §6.5).

**What defines the "slot" differs by part of speech**, and that is the whole
reason three resources are involved rather than one:

| POS | the slot is | resource | coverage / median pool |
|---|---|---|---|
| noun | a position in a taxonomy | WordNet co-hyponyms | 95.6% / 10 |
| verb | an argument frame | **VerbNet classes** ∪ WordNet co-hyponyms | 91.3% / 20 |
| adjective | an attribute dimension | **FrameNet frames** ∪ WordNet antonyms | 65.6% / 6 |

**None of the three is a similarity relation.** The underlying criterion is
selectional compatibility with the shared argument — both values must be
predicable of the same thing. Co-hyponymy approximates it because taxonomic
siblings inherit selectional restrictions; a VerbNet class states it directly
as a shared argument frame; a FrameNet frame states it as a shared dimension.

Four decisions follow.

**Reject the relations WordNet itself calls near-synonymy** — verb groups, and
the `similar_to` cluster of the concept being repaired. Both were previously
ranked *first*. They produced 23% of verb samples and 45% of adjective samples,
and those samples are not believable repairs: correcting `watch` to `see` would
be pedantic, not a slip.

**Add VerbNet for verbs, because WordNet has no usable middle for them.**
Dropping verb groups leaves co-hyponyms, and WordNet's verb hierarchy is too
shallow for those to be sensible on their own (`write` → `rhyme`, `die` →
`relax`). VerbNet class-mates share an argument frame while differing in
meaning: `see` → `hear`, `eat` → `drink`. Members carry WordNet sense keys, so
the synset-to-class mapping is exact rather than lemma-guessed.

This also fits the annotation scheme specifically: **PMB's edge labels are
VerbNet thematic roles**, so a class-mate has a structural reason to accept the
gold role set, which a WordNet co-hyponym does not.

**Add FrameNet for adjectives, because WordNet offers them no contrast relation
at all.** SBN can only write WordNet head adjectives (`.a.`), and heads relate
to heads only by `similar_to` (which returns unwritable satellites — measured
at 0 usable candidates) and `also_see` (which returns near-synonyms). That
leaves antonymy: 40.1% coverage, pool of 1. A FrameNet frame is an attribute
dimension and its adjective lexical units are the values on it, so it supplies
the whole scale rather than its endpoints:

```
spicy -> bitter, bland, delicious, salty, savory, sour, sweet, tart, ...
happy -> angry, ashamed, dejected, elated, excited, contented, ...
```

**Union, never replace.** Each new resource has lower standalone coverage than
the WordNet relation it supplements — VerbNet 61.2% against ~84%, FrameNet
59.0% against ~100% — so both are added to the WordNet pool rather than swapped
in.

Three consequences for the rest of the design:

- **The pool is now wide and deliberately unordered.** Selecting from it is the
  filter's job (§5.2), not the pool builder's. Ranking by relation tightness
  and taking the top candidate is exactly what drove generation into the
  near-synonym end.
- **The language-model filter moves earlier in the order of work** (§9). For
  nouns it is polish. For verbs and adjectives it is load-bearing: every pool
  still contains near-synonyms next to good candidates (`good.a.01`'s opens
  with `effective`), and nothing else chooses between them.
- **Adjective coverage fell, and that was the intended trade.** From a wide
  pool led by near-synonyms to a narrower one led by real contrasts. A sample
  the model cannot learn from is worse than no sample.

**Morphological negation pairs are kept.** 8.5% of adjective samples pair a
word with its own affixal negation (`unqualified` → `qualified`, `impatient` →
`patient`), which antonymy supplies. They are real repairs, but the surface
difference is one prefix, close enough to the same-lemma failure that a model
could learn the affix as a shortcut. The share is tracked; if it rises
materially, or if error analysis shows the model keying on the affix, revisit.

**Dependencies**, both small NLTK corpora, both required rather than optional —
the code fails loudly when either is missing, because their absence degrades
the pool silently:

```
python3 -c "import nltk; nltk.download('verbnet')"        # 429 classes
python3 -c "import nltk; nltk.download('framenet_v17')"   # 1,221 frames
```

Note that VerbNet writes sense keys as `give%2:40:03` where NLTK expects
`give%2:40:03::`. FrameNet gives lemmas rather than synsets, so the pool takes
each lemma's first `.a.` sense — an approximation whose error rate has not
been measured.

### 2.2 Two operators need more than a value pool

The existing interface does not fit them:

**Edge-label repair is an insertion, not a substitution.** The notation doc
annotates *"I ran to, I mean, from the school"* with a dummy concept in each
box, and neither dummy corresponds to anything in the clean graph. The current
entry point is "replace the concept at position *i*"; there is no position to
name. The perturbation itself is a fixed template requiring no semantic
judgement — only the entry point is missing.

**Tense/aspect repair puts two concepts in each box.** The notation doc's
annotation of *"She will go to, well, went to the church"* quarantines both the
time node and the verb, because the speaker re-utters the verb:

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

A cheaper single-concept version exists — quarantine only `time.n.08` and
invert `Time` to `TimeOf`, exactly as the adjunct case does — but it would
annotate the verb as *not* re-uttered, contradicting the surface string.

**Decision: follow the notation doc.** Correspondence to what was actually
said is what distinguishes SBN from more abstract meaning representations;
trading it away here to save engineering would undercut the premise the thesis
argues from. This makes multi-concept span support the largest single piece of
new machinery in the plan — and it is the same machinery forwarding repair
needs, so it buys two cases.

---

## 3. Which cases are machine-made, which need a human

The notation doc defines **13 covered cases**. They fall into four tiers by how
much human involvement they need — a target with fully-automatic generation at
the centre and fully-manual construction at the rim.

### Tier 1 — Fully automatic, already working (7 cases)

SV sentence verb repair · subject repair · verb repair · object repair ·
repair inside negation · donkey sentence (universal quantification) ·
retracting repair.

All seven share one perturbation (content-word substitution). They look like
seven different cases because the graph topology forces a different device in
each syntactic position, but there is nothing case-specific in the code. No
human involvement.

### Tier 2 — Fully automatic, but a new operator must be written first (3 cases)

| case | what is missing |
|---|---|
| adjunct (`monday` → `tuesday`) | constant operator |
| preposition replacement (`ran to` → `ran from`) | edge-label operator **+ insertion entry point** |
| tense & aspect (`will go` → `went`) | tense operator **+ multi-concept span support** |

These require **zero** semantic judgement — arguably less than the operator we
already have, whose quality bottleneck is precisely that choosing a plausible
WordNet neighbour needs judgement. They are absent for interface reasons, not
feasibility reasons.

Building them also repairs a soundness problem in the existing test suite: the
subject-repair and adjunct regression tests currently pass only because their
expected strings were edited to leave constants unchanged (`FINDINGS.md` §2).
With a constant operator they can be restored to the notation doc's own
strings.

### Tier 3 — Automatic, but the source data does not supply enough (1 case)

The weekday sub-case of adjunct repair. All of gold contains **15 documents**
with a `DayOfWeek` role. Related thin cells: `ClockTime` 148 documents,
`YearOfCentury` 126, `MonthOfYear` 59.

This is not a technical limit; the data simply does not contain more. Options
are to hand-write additional instances, or to transplant weekday phrases into
other host sentences by recombining CCG derivation trees (§8, open decision).

### Tier 4 — Human involvement required (3 cases)

| case | why | how much human |
|---|---|---|
| forwarding repair (*Josh drove → Marsha drove*) | span-level: subject **and** verb are re-uttered. The span machinery from tier 2 can generate the skeleton, but whether the result reads naturally needs checking | semi-automatic: machine draft, human review |
| anaphora over reparandum and repair (*"They are both good desserts"*) | needs a second sentence, which PMB's single-sentence documents do not contain | fully manual |
| intra-turn repair combined with cross-turn | needs a second speaker turn | fully manual |

**The two fully manual cases go into the test set only.** A few dozen
hand-written samples cannot teach anything if scattered through training data.
Placing them wholly in test turns them into a deliberate probe: *does a model
trained without ever seeing this kind of repair generalise to it?* That is a
worthwhile experiment, but it must be a chosen design, not an accident of
merging.

### Out of scope

Quantifier replacement · rhetorical relation replacement · modal verb
replacement · delayed (discontinuous) repair.

These are ruled out by the notation doc's own Limitations section: the
`CORRECTION` construction's topology cannot hold them. Not attempted, not
generated, not counted as gaps.

---

## 4. How much data, and of what

This section fixes the size of the dataset. The reasoning runs in four steps:
first ruling out the constraint people usually reach for, then identifying the
one that actually binds, then asking how much each individual cell needs, and
finally deciding what a "cell" is.

### 4.1 The model's capacity is not the constraint

The obvious way to size a fine-tuning set is to ask how much the adapter can
absorb. For this setup the answer is "far more than we will produce", so the
question gives no useful bound.

The pilot's LoRA configuration (`colloquium_prep/lora_scripts/gemma_2_lora.py`)
follows *LoRA Without Regret*: rank 32 applied to **all** linear layers, not
attention only. On Gemma-2-9B — 42 layers, hidden size 3584, MLP width 14336 —
that works out as follows. A LoRA adapter on a matrix with `in` inputs and
`out` outputs holds `r × (in + out)` parameters, so per layer:

```
q_proj   32 × (3584 + 4096)   =   245,760
k_proj   32 × (3584 + 2048)   =   180,224
v_proj   32 × (3584 + 2048)   =   180,224
o_proj   32 × (4096 + 3584)   =   245,760
gate     32 × (3584 + 14336)  =   573,440
up       32 × (3584 + 14336)  =   573,440
down     32 × (14336 + 3584)  =   573,440
                       total    2,572,288  per layer
                             × 42 layers  ≈ 108 million
```

About 108M trainable parameters, roughly 1.2% of the base model.

Now the data side. A repair sample's target is an SBN graph of roughly 80
tokens, and `assistant_only_loss=True` means the prompt contributes no loss.
Five thousand samples is therefore about 400,000 loss-bearing tokens; at three
epochs, about 1.2 million token updates. That is two orders of magnitude below
the parameter count.

The script's own comment — "r=32 is ample for ~9.5k examples" — is correct, and
stays correct at twenty thousand. **Capacity sets no ceiling here, so the
ceiling has to come from somewhere else.**

### 4.2 What actually binds: the ratio to clean data

The pilot's central finding is that the parser **already over-produces
structure**. Faced with a repair it has never been trained on, it does not
discard the reparandum; it glues reparandum and repair together into one larger
graph. Given an interregnum it represents "I mean" as a speech act and wraps it
around both sides, inflating the graph until 205–230 of 836 outputs contain
cycles.

The prior we are trying to shift is therefore "when in doubt, emit more
structure". Training heavily on repair data does not correct that prior — it
inverts it. A model whose training set is mostly repairs learns that
`CORRECTION` is normal, and begins emitting it on clean input.

Three consequences:

1. **Repair should not dominate.** Target roughly **1 part repair to 2 parts
   clean**. With PMB gold train at 9,552 documents, that puts the repair total
   at about **4,500**.
2. **Clean sentences must stay in the stage-2 training mix as negatives** —
   examples whose target contains no `CORRECTION` at all. Without them there is
   no signal for *not* producing the construction.
3. **This gives a concrete acceptance test.** Run the stage-2 model on clean
   PMB dev and check that F1 has not dropped and that no spurious `CORRECTION`
   appears. The pilot already measured that baseline (F1 0.947, 97.1% parse
   success), so the comparison is direct.

### 4.3 Each cell needs less than intuition suggests

Within one cell, every sample is **the same graph transformation applied to
different words**. The model has to learn the transformation; it does not have
to learn the vocabulary, which it already knows. For a pattern this regular,
the learning curve flattens after a few hundred examples, and going to two
thousand mostly buys redundancy.

400–500 per cell sits on the flat part of that curve with margin to spare.

One constraint goes with this: **cap samples at 2 per source document** (3 in
the supply-limited cells of §4.6). Five hundred samples drawn from 148
documents would mean each sentence appearing three or four times with different
substitutions — near-duplicates that inflate the count without adding
information.

### 4.4 What counts as a cell: the device, not the notation case

Step 4 of §1 records that within concept substitution, which notation case a
sample belongs to is forced by the syntax and cannot be sampled independently.
That makes "case" the wrong unit for a quota.

The right unit is the **device**, because that is what the model must actually
produce differently: an anchor dummy, a dropped role and an inverted role
produce visibly different graphs. So tier-1 cells are indexed by device, and
tier-2 cells by the kind of material being repaired.

### 4.5 The grid

| cell | train | dev | test |
|---|---|---|---|
| concept × ① drop role | 500 | 60 | 100 |
| concept × ② invert role | 500 | 60 | 100 |
| concept × ⑤ anchor dummy | 500 | 60 | 100 |
| concept × ④ movement | 300 | 35 | 45 |
| concept × no device | 400 | 50 | 80 |
| constant × person name | 500 | 60 | 100 |
| constant × numeric | 400 | 50 | 80 |
| constant × temporal | 220 | 30 | 50 |
| tense / aspect | 500 | 60 | 100 |
| edge label (device ③) | 500 | 60 | 100 |
| **machine subtotal** | **4,320** | **525** | **855** |
| forwarding (machine draft, human review) | 200 | 30 | 60 |
| anaphora (hand-written) | — | — | 40 |
| cross-turn (hand-written) | — | — | 40 |
| **total** | **4,520** | **555** | **995** |

Against 9,552 clean documents, the training ratio is 1 : 2.1.

The test set at roughly 1,000 rows is the same order as the pilot's 836
sentences, so that study's analysis code and statistical design transfer
directly. The ~100 rows per cell is what makes a per-cell paired comparison
meaningful at pilot-comparable power; below about 50 the per-cell numbers
become anecdotes.

### 4.6 Where supply runs out, and the rule that follows

Two cells cannot be filled freely, and it is worth being precise about *what*
is scarce in each. Candidate **rows** are never scarce — each repair site
offers up to ten WordNet candidates, and the training split has 16,425 legal
sites, so the raw pool runs to six figures. What is scarce is **distinct source
sentences**, and that is what governs diversity.

**Device ④ (movement).** Only 525 sites in the whole training split genuinely
require movement (481 adjectives, 44 verbs, 0 nouns). Movement could be
*forced* on sites where a cheaper device also applies, which would raise supply
arbitrarily — but the notation doc calls ④ a last resort, so forcing it would
manufacture annotations no human annotator would write. We accept the limit
instead: 300 of 525 available sites.

**Temporal constants.** Counting documents that carry `ClockTime`, `DayOfWeek`,
`MonthOfYear`, `DayOfMonth` or `YearOfCentury`: 312 train / 52 dev / 73 test.

That second cell is also the answer to a question left open earlier: *does
`DayOfWeek` scarcity need special handling?* At this scale, **no — provided the
cell is defined as "temporal constant" rather than "weekday"**. Weekday alone
occurs in 15 / 2 / 3 documents across the splits, which cannot support a
reportable cell. Pooled, the cell is comfortably fillable. The scarcity was
largely an artefact of choosing too fine a granularity, and it returns only if
weekday repair must be reported as its own result — which is the case for CCG
recombination, and is deferred (§8).

From these two cases comes a general rule:

> **Set each cell's quota at roughly 70% of what it can supply.**

The reason is not caution for its own sake. The similarity band of §5.2 has no
ground truth yet and will be re-tuned after we see real output. Leaving 30%
headroom means a *stricter* band can still fill the quota by re-filtering the
existing pool, instead of forcing a full regeneration.

### 4.7 Over-generation and the surplus pool

Some proportion of generated samples will be legal, fluent, and still not sound
like something a speaker would actually say — *"The wind is blowing region, or
rather, east."* This is the residue the language-model filter exists to catch
(§5.2).

**We do not know that proportion yet**, and this document will not guess one.
For planning purposes, assume a strict band could reject 30–40% of scored rows;
LARD 2025 reached for a sentence-encoder re-rank to address the same problem,
which suggests it is not negligible.

Fortunately, buffer is nearly free — but only in the right units:

- **In rows, buffer is abundant.** The filter rejects *candidates*, not sites;
  a site survives if any one of its ten candidates lands in the band. The raw
  pool is orders of magnitude above quota, and scoring tens of thousands of
  short sentences with an encoder costs minutes.
- **In distinct sentences, buffer is what §4.6 measured**, and it is why the
  70% rule exists.
- **In the temporal cell, buffer comes from the value pool rather than the site
  pool.** There is roughly one usable site per document there, but each site
  admits many replacement values (`monday` has six alternatives, a date or a
  clock time has dozens). The filter still has something to choose among.

Three defects that might otherwise be assumed to consume buffer do **not**,
because they are fixed upstream rather than filtered downstream: same-lemma
candidates, reformulation markers, and the capitalisation bug (§5.4). They
shrink the candidate pool at generation time; they never reach the filter.

**Practical arrangement.** Generate every in-band candidate and score it, then
sample to quota. Keep the unsampled remainder as a **surplus pool file**,
separate from the released splits. Because `similarity_score` is stored per row
(§6.4), re-tuning the band later is a query against that pool, not a new
generation run.

---

## 5. Quality control

Two automated layers plus a human pass. They check genuinely different things
and the order matters.

### 5.1 Layer one — rules, over the graph

Every check here is a yes/no with no grey zone, and none needs a model.

1. **The graph parses.** Round-trip every sample through `sbn_smatch.SBNGraph`,
   PMB's own parser. Anything it rejects is discarded.
2. **No cycles.** Inserting boxes shifts every downstream index; an arithmetic
   error can create a cycle in a graph that still **parses without complaint**
   and simply means something else. This is the most dangerous failure mode in
   the pipeline, so it is checked explicitly rather than left to the parser.
3. **Box membership is correct.** `CORRECTION` and `CONJUNCTION` must be
   sibling boxes under one parent; the reparandum must land in the first, the
   repair in the second.
4. **No edges lost.** Edge count before and after the splice must match.
5. **Index magnitude ≤ 9.** PMB's index pattern matches a single digit, so
   `-10` is silently truncated to `-1` with no error raised. The splice
   inflates magnitudes, making this a live constraint.

### 5.2 Layer two — a language model, over the sentence

This layer judges only the natural-language side. Graph legality is already
guaranteed by layer one, and a sentence encoder has nothing useful to say
about it.

**We do not copy LARD's selection rule, because it points the wrong way.**
LARD — the method this generator is modelled on — picks, from all candidates,
the one whose sentence has the **highest** cosine similarity to the original.
For their task (disfluency detection) that is right: they want the disfluent
input to stay coherent. For us it is harmful, because maximising similarity
drives straight toward near-synonymy, and near-synonymy is where our two worst
defect classes live:

- at moderate closeness, the sample reads as a gloss rather than a correction —
  *"We added something original, that is, new"* (27.6% of the trial corpus);
- at the limit, the two surface forms are **identical** — *"He ran, actually,
  ran five miles"* (8.0%), where the graph quarantines a sense distinction the
  string cannot express.

These are one axis, not two bugs.

**So the criterion is a band, not a maximum:**

- **lower bound** — far enough apart that the correction carries content. This
  excludes synonyms, and excludes same-lemma pairs automatically.
- **upper bound** — close enough to be a plausible production slip: same
  semantic field, same syntactic slot. Without it we get *"The wind is blowing
  region, or rather, east."*

#### The two bounds are different phenomena and need different instruments

It is tempting to measure both ends with one number — a sentence-level cosine,
as LARD does. That works poorly, for a reason worth stating:

| bound | what is actually wrong | instrument |
|---|---|---|
| **lower** (too similar) | a relation between **two words** — they mean nearly the same thing | semantic distance between the reparandum and repair *words* |
| **upper** (too distant, awkward) | a relation between a word and its **context** — the word does not fit where it was put | a language model's surprisal for the reparandum in its left context |

A sentence-level cosine is a blunt instrument for the upper bound in
particular. The two sentences differ in exactly one word and share everything
else, so most of the similarity being measured comes from the shared material,
diluting the signal. LARD was unaffected because it wanted the maximum; we want
to discriminate, so the dilution is a real loss.

**Language-model perplexity cannot replace the semantic measure.** It handles
the upper bound well — *"the wind is blowing region"* is a low-probability
continuation and an LM sees that immediately. It is blind to the lower bound,
and blind in the damaging direction: *"We added something original, that is,
new"* is **perfectly fluent** and scores *well*. That failure is not a fluency
failure at all — the string is a well-formed appositive gloss; what is wrong is
that the graph asserts withdrawal. No fluency metric can see it.

One caveat on using perplexity for the upper bound: self-repair is *inherently*
disfluent, so every good sample scores worse than its clean source. Raw
perplexity therefore conflates "this is a repair" with "this is a bad repair".
It has to be used comparatively — among candidates at the same site, where the
context is held constant.

#### Calibration

**The band has no ground truth and must be calibrated.** The method: sample
candidates across several score deciles, have a human label each as *reads as a
slip* / *reads as an explanation* / *reads as a change of subject*, and set the
thresholds from that. Both bounds are calibrated on the same labelled slice.
The procedure is itself reportable in the thesis's methods section.

#### What the filter is not responsible for

Two guards belong in the candidate generator, not here, because their cause is
identifiable in advance and a filter would only be cleaning up after it
(`FINDINGS.md` §6.1, §6.5):

- reject candidates sharing the reference's lemma;
- reject the WordNet relations that *are* near-synonymy — verb groups and
  `similar_to` clusters (§2.1).

Filtering cannot compensate for a pool that lacks good candidates. That is why
§2.1 changes the pool first: for verbs, before VerbNet was added, no threshold
setting could have produced a good sample, because none was on offer.

### 5.3 Layer three — human spot-check

A cosine similarity measures semantic distance. It cannot tell whether
something is a *believable slip*. A human pass is therefore not optional — but
it has to be sized honestly, because a small sample supports a much weaker
claim than it looks like it does.

**Two different jobs, two different sample sizes:**

- **Per cell, ~30 rows — a sentinel, not a rate.** At a true defect rate of
  20%, a sample of 30 gives a 95% confidence interval of roughly ±14 points.
  That is enough to answer "does this cell have a systematic problem?" and not
  enough to report a number. Use it that way.
- **Pooled across cells, ~360 rows (12 cells × 30) — a reportable figure.** At
  that size the corpus-level defect rate carries an interval of about ±4
  points, which is precise enough for a claim in the thesis.

**What the spot-check produces is a number, not a smaller corpus.** Deleting 30
flagged rows changes nothing. The output is a measured residual defect rate,
plus a decision: if that rate is too high, tighten the similarity band and
re-filter. Because `similarity_score` is stored on every row (§6.4) and the
unsampled surplus is retained (§4.7), re-filtering is a query, not a
regeneration.

### 5.4 Defects that must be fixed before regenerating

Measured in the trial corpus, all fixable, together affecting roughly a third
of its rows (`FINDINGS.md` §6). Status is as of the second fix round; the
first round's entries are marked where its findings still stand
(`FINDINGS.md` §6.6). The second round was prompted by 150 hand-annotated
calibration items, whose oddities turned out to share one cause: `align`
returned a single token index, so a multiword concept was only ever matched,
inflected and replaced on its first word.

| defect | share | fix | status |
|---|---|---|---|
| same-lemma reparandum/repair | 8.0% | reject candidates whose lemma matches the reference | **done** — 0.0% |
| reformulation-marker interregna | 27.6% | the five-marker set of §1 step 5 | **done** — 0 |
| mid-sentence capitalisation (*"The convention, I mean, The peace treaty…"*) | 3.8% | capitalise by final position, not by original position | **done** — 0.0% |
| interregnum ratio inverted | 69.6% marked | generate the base layer first (§1 step 3) | **done** — 31.2% |
| alignment misses named entities | 23% unaligned | `Name`-constant matcher (§1 step 2) | **done** — agreement 89.7% → 95.5% |
| near-synonym candidates | verbs 23%, adjectives 45% | change the pool, not the filter (§2.1) | open |
| broken inflections (`adversaryest`) | 0.2% | already flagged by a column; filter | open |
| past tense taken as past participle (*"Tom sewn"*) | not yet counted | irregular-verb table picks the wrong form | open |
| multiword lemma pluralised on the first word (*"bigs cat"*) | 1.5% of pool | right-headed for nouns, left-headed for verbs, head-initial for postmodified phrases (`inflect_en._head_index`) | **done** — 0 |
| multiword repair replaced on its first word only (*"his hole card **card**"*, *"harmonizing **up** its presence"*) | 5.4% of pool | `align` returns the token span spelling the concept, and gives the site up when it cannot resolve one | **done** — 0 |
| reparandum inflected from the modifier's tag (*"I love hamburger, that is, hot dogs"*) | 11.5% of multiword-noun samples | take the tag from the compound's head token, not the aligned one | **done** — 0 |
| candidate already present in the sentence (*"Lungs, heart, **veins**, **veins** and capillaries"*) | 0.12% | surface-and-lemma guard in the pool builder | **done** — 0 |
| argument-frame mismatch (*"I conferred your brother on the street"*) | 2.2% of pool | **not a pool guard — see below** | open |

**One defect class does not appear in this table, and that is the point of
§5.1 and §5.2 being separate layers.** During the first fix round, an
intermediate version produced *"Adversary, Hitler assumed power in 1933."* —
a legal graph that passed every rule check, and whose only symptom was that the
sentence is nonsense. Reading samples is part of each iteration, not a final
audit step.

#### Argument-frame mismatch belongs to the upper bound, not to the pool

The last row of the table is the one open defect this design already believed
it had solved, so it needs its reasoning recorded rather than a one-line fix.

**What it looks like.** A multiword verb repair replaced by a single-word
candidate loses its particle, and the candidate's own frame may not accept the
arguments left behind:

```
run_into.v.04  -> confer      I conferred your brother on the street.     no
put_in.v.05    -> distribute  She distributed for a raise.                no
wipe_away.v.01 -> shed        Tom shed his tears.                         fine
rip_off.v.01   -> overcharge  Tom overcharged you.                        fine
```

5,159 rows, 2.2% of the pool. It is not new — before `align` returned spans the
same rows read *"I conferred **into** your brother"*, an obviously broken
string. Repairing the residue did not repair this; it only changed the
signature from visibly broken to plausibly wrong, which makes it harder to
spot in annotation and is worth saying out loud.

**Why §2.1's answer does not cover it.** VerbNet was added precisely to
guarantee a shared argument frame, and these candidates come from VerbNet, not
from the co-hyponym half of the union — the co-hyponym pool for these synsets
is empty. Class membership states that two verbs share an alternation frame *at
class level*; it does not state that they are interchangeable in one surface
realisation. When the repair is phrasal, the frame that made them class-mates
was stated for the phrasal form, and deleting the particle deletes it.

**Why layer one cannot be recruited.** `d1_drop_role` fires on 4,629 of the
5,159 rows, on the good and the bad alike — it describes splice mechanics (can
the reparandum copy carry that edge) and not English valency. PMB's edge labels
being VerbNet roles does not make the graph layer sensitive to this.

**Why it is nevertheless not a pool guard**, despite §5.2's rule that a cause
identifiable in advance belongs in the generator. The good and the bad rows are
structurally identical — same relation source, same device, same shape
(multiword repair × single-word candidate). Any rule expressible at pool level
kills both, and roughly four in ten of this class read fine. What separates
them is whether the word fits where it was put, which is the upper bound's
definition verbatim: *a relation between a word and its context, measured by a
language model's surprisal for the reparandum in its left context*. **This is
the strongest case yet for building the upper bound**, which as of the second
fix round is still unimplemented — `nli_filter` carries only
`SYNONYMY_REJECT_ABOVE`.

**What the pool should do is instrument, not filter.** Emit a `frame_change`
column marking multiword-repair × single-word-candidate rows, so calibration
can stratify on the class and measure whether the upper bound actually
separates it. Without the column there is no way to verify that the bound
solved the problem it was built for. Same principle as §6.4: store the score,
not only the verdict.

---

## 6. What the finished dataset looks like

One row per sample. Fields are grouped by what they are for.

### 6.1 Content

| field | null? | description |
|---|---|---|
| `repair_id` | no | unique row identifier |
| `pair_id` | no | links a base repair to its interregnum-bearing counterpart, so matched-pair comparisons are possible without re-deriving the link |
| `nl_repair` | no | the sentence containing the self-repair |
| `sbn_repair` | no | its repair-aware SBN graph |
| `nl_clean` | **yes** | the original sentence with no repair |
| `sbn_clean` | **yes** | the original gold graph |

`nl_clean` matters as much as `sbn_clean`: it is the no-repair control
condition, and without it the matched-pair comparison the pilot ran cannot be
reproduced.

### 6.2 The repair itself

| field | null? | description |
|---|---|---|
| `reparandum_surface` | no | the abandoned text, as a string |
| `repair_surface` | no | the replacing text, as a string |
| `reparandum_node` | no | its identity in the graph — a synset (`run.v.29`), a constant value (`"Josh"`), or an operator (`TSU`), depending on what was repaired |
| `repair_node` | no | likewise |
| `reparandum_span` | no | character start/end of the reparandum within `nl_repair` |
| `repair_span` | no | character start/end of the repair |
| `interregnum` | yes | the editing signal, empty if none |

**Surface strings and graph identities are both needed** because they serve
different evaluations: span-extraction scoring (as in the pilot's ablation)
needs the strings; graph scoring needs the nodes. Collapsing them into one
"reparandum" field forces a guess about which was meant.

**Storing the spans removes a step and its errors.** The pilot had to recover
ground-truth spans by diffing the repaired sentence against the original. We
generate the sample, so we already know the offsets; writing them down is free.

### 6.3 Classification — three fields, not one

| field | values |
|---|---|
| `case` | which of the notation doc's covered cases: `subject` / `object` / `verb` / `negation` / `donkey` / `retracting` / `adjunct` / `tense_aspect` / `preposition` / `forwarding` / `anaphora` / `cross_turn` |
| `repaired_material` | `concept` / `constant` / `tense_operator` / `edge_label` — the axis that drives the generator (§2) |
| `device` | which device the topology forced: `drop_role` / `invert_role` / `dummy_concept` / `movement` / `anchor_dummy` |

These are kept separate because they are genuinely independent, and because
error analysis needs the third one: the pilot found that a parser's failure
modes cluster by structural configuration. Merged into a single "repair type"
field, that grouping becomes impossible to recover.

### 6.4 Provenance and quality

| field | null? | description |
|---|---|---|
| `provenance` | no | `machine` / `machine_reviewed` / `hand_written` |
| `similarity_score` | yes | the encoder's score for this pair |
| `passed_lm_filter` | no | whether it fell inside the band |
| `human_checked` | no | whether a human reviewed this row |
| `max_abs_index` | no | largest index magnitude in `sbn_repair`; needed because 10 truncates silently |

`provenance` is not optional. Three different production routes are merged into
one file (§3), and without this column the results cannot be reported
separately, nor can a reader be told how much of the data is hand-made.

**Store scores, not just verdicts.** Keeping `similarity_score` alongside
`passed_lm_filter` means the band can be re-tuned by re-filtering rather than
re-generating.

### 6.5 Provenance of the source, and room for other corpora

| field | null? | description |
|---|---|---|
| `source_corpus` | no | `pmb-5.1.0` today; `switchboard` or others later |
| `source_id` | no | document identifier within that corpus (`p50/d0779` for PMB) |
| `split` | no | `train` / `dev` / `test` / `test_long` |
| `is_synthetic` | no | true for our constructed repairs; false for naturally occurring ones |
| `annotation_status` | no | `gold` (the source corpus's own annotation) / `derived` (computed by us from gold) / `hand` (annotated by a human) / `none` (text only, graph pending) |

The point of these fields is not to record a corpus name. It is that **real
spontaneous speech has no clean counterpart and no gold graph.** A genuine
self-repair from Switchboard does not come with "what the speaker would have
said if they hadn't stumbled", and certainly not with an annotated semantic
graph.

So the schema commitment is this: `nl_clean` and `sbn_clean` are **nullable**,
and `annotation_status` records what is actually known. With those two
provisions, a future Switchboard import can enter as text-only rows with status
`none` and be annotated incrementally, without a schema migration. Making the
clean columns mandatory would close that door permanently.

---

## 7. Changes needed in the notation doc

Three, all consequences of decisions recorded above:

1. **Add device ⑤ (anchor dummy) to the repair toolkit table.** The doc's own
   subject-repair example uses it, but the table lists only four devices.
2. **Record that device order depends on part of speech.** The table presents
   ④ (movement) as a last resort, but for prenominal modifiers it is the first
   choice, and applying ⑤ to an adjective produces nonsense
   (`entity.n.01 EQU fresh.a.01`).
3. **Remove `in fact` from the interregnum examples** (currently at line 70).
   It is a reformulation marker and has been dropped from our set (§1 step 5);
   leaving it in the doc would recommend a marker the dataset excludes.

---

## 8. Still undecided

1. **Whether to do CCG-tree recombination.** Zhang et al. (2024) build a
   compositional challenge set by recombining PMB's CCG derivation trees, which
   would let a weekday phrase be transplanted into many host sentences and lift
   the `DayOfWeek` ceiling from 15 documents to roughly 190.

   *Current recommendation: defer.* Two constraints found on inspection: PMB
   ships CCG derivations for only 1,132 documents, and the repository contains
   **no Boxer** — the component that turns a recombined tree back into a graph.
   We would have to compose the recombined SBN ourselves, which is a second
   splice engine to write and validate. Its stronger justification is as an
   independent contribution (a compositional challenge set for repair parsing),
   which argues for scheduling it separately from the main dataset rather than
   as a dependency of it.

2. **The similarity band's thresholds**, pending the calibration described in
   §5.2. This one cannot be settled in advance on principle: it depends on what
   the first real batch of output looks like. The design accommodates that by
   storing scores and retaining surplus (§4.7), so the thresholds can be set
   after the fact without regenerating.

---

## 9. Order of work

Derived from the dependencies above, not a schedule. Development proceeds in
small increments, each built on a verified previous state; every increment ends
by reading samples, not only by checking metrics (§5.4).

**Done:**

1. ~~Fix the trial-corpus defects (§5.4)~~ — four of them; see the status
   column there.
2. ~~Fix alignment via the `Name` matcher, and measure the improvement against
   the CCG ground truth (§1 step 2)~~ — 89.7% → 95.5%.
3. ~~**Rebuild the content-word candidate pool** (§2.1)~~ — done: near-synonym
   relations dropped, VerbNet added for verbs, FrameNet for adjectives, and
   ranking by relation tightness removed so the pool reaches the filter wide
   rather than pre-selected.

**Next, in this order:**

4. **Add the two filter signals and calibrate them** (§5.2). This moved ahead
   of the remaining operators, because it is not polish: pools now run to 6–20
   candidates with no ranking, so nothing decides which one is used, and every
   pool still contains near-synonyms. Verb and adjective output is unusable
   until the filter exists; noun output is already sound and serves as the
   stable baseline meanwhile.
5. Build the constant operator — smallest new operator, and it makes two
   existing regression tests honest (§3 tier 2).
6. Build the edge-label operator with its insertion entry point.
7. Build multi-concept span support, then the tense/aspect operator and
   forwarding repair on top of it — the largest piece, and the one that
   delivers two cases.
8. Generate the full candidate pool across all operators and score it.
9. Filter, sample to the grid of §4.5, insert interregna, and keep the
   remainder as the surplus pool.
10. Hand-write the two discourse cases; merge into test.
11. Human spot-check (§5.3); write out with the schema of §6.

**Then measure, and only then consider scaling.** The grid of §4.5 is a
starting size, not a final one, and re-running the generator is cheap. Train,
then look at per-cell F1 on dev: if a cell sits well below the clean baseline
and stops improving, raise that cell's quota and regenerate it alone. Deciding
by measurement beats trying to guess the right size now.
