# SBN Annotation & Evaluation Manual (Core Notation)

**Scope.** This manual documents the *original* SBN formalism as proposed and refined by Johan Bos, independent of any extension for self-repair, ellipsis, or other discourse phenomena. It is written for operational use: it tells you what a legal SBN token looks like and how tokens combine into a well-formed meaning representation, without requiring you to know the underlying model theory (first-order logic, DRT translation functions, or de Bruijn-register semantics). A reader — human or model — who follows this manual should be able to (a) write a syntactically legal SBN string for a sentence, and (b) judge whether an existing SBN string is well-formed and structurally sound.

**A terminological note, stated up front because it matters for citation and for talking to Bos's group.** The acronym "SBN" is not used consistently across the primary literature:

| Source | Name used |
|---|---|
| Bos, *Variable-free Discourse Representation Structures* (2021) | **Simplified Box Notation** |
| Bos, *Quantification Annotation in DRT* (ISA-17/IWCS, 2021) | **Simplified Box Notation** |
| Bos, *The Sequence Notation: Catching Complex Meanings in Simple Graphs* (IWCS 2023) | "the sequence notation" — avoids the SBN acronym entirely in running text |
| Zhang, Bouma & Bos, *Neural Semantic Parsing with Extremely Rich Symbolic Meaning Representations* (2025) | "sequence notation," referencing Bos (2023) |
| Zhang & Bos, *Is Neural Semantic Parsing Good at Ellipsis Resolution or Isn't It?* (2025) | **Sequence Box Notation** |

These are not three different formalisms — they are the same notation at different points in its development, referred to inconsistently even by its own author across a five-year span. This manual treats them as one continuous object and uses **SBN** throughout, defaulting to Bos's own original expansion, *Simplified Box Notation*, while flagging below the few points where the notation itself (not just the name) visibly changed between the 2021 and 2023 papers.

**Sources used.** Bos (2021) *Variable-free Discourse Representation Structures*; Bos (2021) *Quantification Annotation in Discourse Representation Theory*; Bos (2023) *The Sequence Notation: Catching Complex Meanings in Simple Graphs* (IWCS); Zhang, Bouma & Bos (2025) *Neural Semantic Parsing with Extremely Rich Symbolic Meaning Representations*, including its Appendices B–C; Zhang & Bos (2025), *Is Neural Semantic Parsing Good at Ellipsis Resolution or Isn't It?*; Bonial, Corvey, Palmer, Petukhova & Bunt (2011), *A Hierarchical Unification of LIRICS and VerbNet Semantic Roles*, cited by Bos as the source of the SBN role inventory. Also cross-checked directly against the reference implementation, `sbn_spec.py` and `graph_base.py` (PMB toolchain, an SBN-parsing library whose docstring identifies it as based on PMB 4.0.0), where the two diverge from or extend what the papers state — flagged explicitly wherever this occurs, since the uploaded copy of `sbn_spec.py` is a hand-modified working file (it already contains the repair-extension separator `CORRECTION` with an inline comment marking it as newly added) rather than a clean baseline. Content specific to the self-repair extension (the `CORRECTION` separator and related material) has been deliberately excluded, as instructed.

---

## 1. What SBN is, in one paragraph

SBN is a meaning-representation format for Discourse Representation Theory (DRT) that removes two things classic DRT relies on: explicit variables and an explicit box-within-box tree structure. Instead, a meaning is written as a flat, linearly ordered **sequence of tokens** — the same sequence a human annotator would type into a plain text editor, and the same sequence a sequence-to-sequence neural model would generate. Variables are replaced by relative, de-Bruijn-style **indices** (count backward or forward through the sequence rather than naming a variable), and box embedding is replaced by inline **separator** tokens that mark where one "context" ends and the next begins. Every SBN sequence corresponds to exactly one directed acyclic graph (DAG): concepts become oval nodes, contexts become box nodes, and role/operator relations become labelled edges between them. The sequence notation and the graph are two views of the same object; neither is more primitive than the other, though the sequence is what gets typed, stored, and parsed.

---

## 2. Vocabulary: the legal components

Bos's own inventory of SBN's "ingredients" is: **concepts, constants, roles, operators, indices, contexts, separators, connectors**. Each is defined below, in the order you will actually use them when annotating a sentence left to right.

### 2.1 Concepts

A concept is a one-place predicate that (a) introduces a new discourse referent, and (b) classifies that referent under a WordNet sense.

**Format:** `lemma.pos.sense`, e.g. `cat.n.01`, `see.v.03`, `keen.a.01`, `now.r.01`.

- `lemma` is always **lower-case**. Multi-word lemmas are joined with an underscore: `banana_bread.n.01`, `publishing_group.n.01`, `nonexecutive.a.01` (single word here, shown for contrast).
- `pos` is one of four single-letter tags: `n` (noun), `v` (verb), `a` (adjective), `r` (adverb).
- `sense` is the WordNet 3.0 sense number for that lemma/POS pair, zero-padded to two digits (`.01`, `.02`, … `.11`, …).
- Concepts are treated as **language-neutral**: the WordNet synset is the interlingual unit, and the English lemma string is just its conventional label in an English-language WordNet.
- One reserved concept is worth knowing: `entity.n.01`, the most general noun synset (the top of the noun hierarchy). It is used as a semantically empty placeholder whenever a discourse referent needs to be introduced without committing to a more specific type — for instance when normalising negation/disjunction so that every discourse referent's classifying predicate lives in the same context it's declared in, or when building a **plural/collective referent** out of two named entities (see §4.6).

A **verb concept is itself a one-place predicate**, following neo-Davidsonian event semantics: `buy.v.01` introduces an event referent, and its participants are attached afterward as separate role-headed conditions, not as arguments of the verb itself.

### 2.2 Constants

Constants are the atomic values a concept can be grounded to. The picture below combines Bos's prose description with the reference parser's actual `CONSTANTS`/pattern definitions (`sbn_spec.py`), which diverge from the prose in one place and add several constant classes Bos's papers don't mention at all:

- **Literal strings** — proper names of people, places, organisations, artefacts, or any other literal mention. Double-quoted: `Name "Mary"`, `Name "Elsevier N.V."`.
- **Numbers** — integers or reals, unquoted: `EQU 61`, `Quantity 2`.
- **Year constants** — a distinct, **single-quoted** four-character form, confirmed in the reference parser but not documented in any paper: `'2008'`, or with a wildcard digit for an imprecise decade, `'196X'`.
- **A bare single capital letter**, confirmed in the reference parser (`VALUE_CONSTANT`) with its own worked example in the source comment — *"Tom got an A on his exam"* → `Value "A"`.
- **`?`** — confirmed as a legitimate constant for an unknown/unspecified quantity or value (matches the corpus finding discussed under §2.8's empirical addendum).
- **Weekday names** (`monday` … `sunday`) — confirmed as bare constants in the reference parser's `CONSTANTS` set, presumably the values taken by the `DayOfWeek` role.
- **Deictic constants** — Bos's prose (2023) states four: `speaker`, `hearer`, `now`, `here`. The reference parser's `CONSTANTS` set confirms `speaker`, `hearer`, and `now`, plus `unknown_ref`, but **does not include `here`**. This manual cannot resolve the discrepancy — either location deixis is handled some other way in practice (e.g. via a concept rather than a bare constant) or `here` was never implemented — so treat `here` as documented-but-unconfirmed.
- **A known, acknowledged ambiguity**: the reference parser's own header comment states that a constant which happens to look like a signed number (e.g. the literal value `-1` in "the temperature was -1 degrees") cannot currently be distinguished from an actual backward index `-1`. This is a real parsing limitation, not a documentation gap — worth keeping in mind for any data containing genuinely negative numeric values.

### 2.3 Roles

A role is a two-place predicate connecting a concept to something else (a constant, or — much more commonly — another concept via an index).

**Format:** starts with an **upper-case** letter, then usually lower-case, with internal capitalisation for multi-word roles (`AttributeOf`, `MonthOfYear`). This is what visually distinguishes a role from a concept (always lower-case) and from a separator (always ALL-UPPERCASE, §2.7).

**Role inversion.** Any role `R` has a dual, written by suffixing `Of` onto it (`Theme` → `ThemeOf`, `Agent` → `AgentOf`), such that `R(x,y) ↔ ROf(y,x)`. This lets you swap which of the two connected concepts is written first, purely for alignment with surface word order, without changing the truth-conditional meaning. **This generality is Bos's stated theory, not what the parser actually enforces.** The reference implementation (`sbn_spec.py`, PMB toolchain) recognises only a fixed, closed set of ten invertible forms: `AttributeOf`, `ColourOf`, `ContentOf`, `InstanceOf`, `PartOf`, `SubOf`, `AgentOf`, `MannerOf`, `ThemeOf`, `TimeOf`. In principle no other role's `Of`-form is guaranteed to be recognised, even though Bos's own papers use inverted forms outside the base six-role subset of this list (`ThemeOf`, `AgentOf`, `TimeOf`, and `MannerOf` all occur in his own worked examples) — the implementation's whitelist appears to have simply lagged behind the published notation at some point and been patched to catch up, rather than the general rule ever having been fully implemented as stated. `boy.n.01 buy.v.01 Agent -1 Theme +1 book.n.02` and `boy.n.01 buy.v.01 Agent -1 book.n.02 ThemeOf -1` are the same meaning; only the second keeps the object concept adjacent to where "book" appears in the English sentence. Inverted roles are normalised back to their base form before any structural comparison (e.g. Smatch scoring), so inversion is purely a writing-convenience device, not a semantic distinction.

**Two functionally different subclasses of role co-exist under one surface format**, and it is worth keeping them apart mentally even though nothing in the string itself marks the difference:

1. **Thematic (event-participant) roles** — connect an event/state concept to its participants. This is the VerbNet/LIRICS-derived inventory (Bonial et al. 2011), adopted by Bos "by and large."
2. **Relational (static/attributive) roles** — connect two entity concepts to each other outside of any event (naming, part-whole, ownership, subset membership). These are PMB-specific additions on top of the VerbNet/LIRICS set.

The table below reproduces the role hierarchy given in Zhang, Bouma & Bos (2025, Appendix B) — the most recent and most complete published inventory, organised there as a taxonomy tree; short glosses are drawn from Bonial et al. (2011) where that paper defines the role explicitly, and otherwise inferred from how the role is actually used in Bos's worked examples (flagged *[inferred]*).

**Attribute-branch roles** (attach a property to an entity):

| Role | Gloss |
|---|---|
| `Attribute` | undergoer that is a property of an entity, rather than the entity itself |
| `Quantity` | numeral/amount attached to a countable entity *[inferred]* |
| `Name` | literal proper-name string attached to an entity |
| `Order` | ordinal/sequence position *[inferred]* |
| `Colour` | colour property *[inferred]* |
| `Unit` | measurement unit attached to a quantity/measure concept *[inferred]* |
| `Title` | honorific/title string attached to a person *[inferred]* |

**Participant → Time-branch roles:**

| Role | Gloss |
|---|---|
| `Time` | instant or interval during which a state holds or event occurs |
| `Duration` | length or extent of time |
| `SpecificTime` | calendar/clock anchoring of a time referent *[inferred]*, with children: |
| &nbsp;&nbsp;`YearOfCentury`, `ClockTime`, `DayOfWeek`, `MonthOfYear`, `DayOfMonth`, `Decade` | individual calendar/clock fields (`Decade` confirmed present in `sbn_spec.py` but absent from the 2025 paper's taxonomy figure) |
| `Frequency` | number of occurrences of an event within a span |
| `Start` / `Finish` | onset / endpoint of an event or state *[inferred, cf. LIRICS Initial/Final Time]* |

**Participant → Actor-branch roles** (instigators). **`Actor` itself is a taxonomy-diagram branch label, not an implemented, annotatable role** — confirmed absent from `sbn_spec.py`'s role set, unlike `Participant` one level up, which *is* directly usable. Only the entries below the branch header are things you can actually write:

| Role | Gloss |
|---|---|
| `Agent` | actor who initiates and carries out the event intentionally, and exists independently of the event |
| `Co-Agent` | agent acting reciprocally alongside another agent in a symmetrical event (confirmed hyphenated in `sbn_spec.py`) |
| `Causer` | actor that initiates an event without intentionality (LIRICS calls this role *Cause*) |
| `Cause` | also present as its own, separate entry in `sbn_spec.py`, distinct from `Causer` — the relationship between the two is not documented in any source consulted; possibly a legacy/duplicate form |
| `Affector` | present in `sbn_spec.py`, no gloss in any source consulted *[undocumented]* |
| `Stimulus` | cause that elicits an emotional/psychological response in an event of perception |

**Participant → Undergoer-branch roles.** As with `Actor`, **`Undergoer` itself is a taxonomy-diagram label, not an implemented role** (confirmed absent from `sbn_spec.py`).

| Role | Gloss |
|---|---|
| `Theme` | undergoer central to the event/state, not structurally changed by it |
| `Co-Theme` | a second, equally-participating theme in a symmetrical event (confirmed hyphenated) |
| `Topic` | theme characterised by transferred information content (events of communication) |
| `Pivot` | theme that is more central than a paired, unequal co-participant |
| `Proposition` | argument that is an entire embedded clause/context rather than a single entity — see the empirical addendum under §2.8 for its distinct, box-level indexing behaviour |
| `Patient` | undergoer that experiences a change of state/location/condition |
| `Co-Patient` | a second, equally-participating patient (confirmed hyphenated) |
| `Experiencer` | patient that is aware of the event it undergoes (perception events) |
| `Material` | patient present at the start of the event, transformed into a new entity by it |
| `Beneficiary` | undergoer (dis)advantaged by the event |
| `Instrument` | undergoer manipulated by an agent to carry out the event |

**Participant → Abstract-branch roles:**

| Role | Gloss |
|---|---|
| `Degree` | scalar amount of a gradable property *[inferred]* |
| `Goal` | end point of the action, existing independently of the event |
| `Result` | goal that comes into existence through the event |
| `Product` | a concrete-object result |
| `Manner` | way in which the event is carried out *[inferred]* |
| `Context` | background circumstance *[inferred]* |
| `Measure` | present in `sbn_spec.py`, not in the 2025 paper's taxonomy; no gloss in any source consulted *[undocumented]* |

**Participant → Place-branch roles.** As with `Actor`/`Undergoer`, **`Place` itself is a taxonomy-diagram label, not an implemented role** (confirmed absent from `sbn_spec.py`).

| Role | Gloss |
|---|---|
| `Location` | a concrete place |
| `Source` | starting point of an action |
| `Destination` | concrete end point of an action (a "Goal" that is a place) |
| `Recipient` | an animate destination |
| `Value` | position along a formal scale |
| `Asset` | a concrete-object value |
| `Extent` | amount of measurable change over the event |
| `Path` | route or trajectory *[inferred]* |

**Static / relational roles** (top-level branches `Role` and `Relation` — not tied to an event; `Role` itself, unlike `Actor`/`Undergoer`/`Place` above, *is* directly implemented as a usable role token):

| Role | Gloss |
|---|---|
| `Of` | the generic inverse-role suffix itself; also used as a bare relational role in some constructions |
| `Affectee` | entity affected by a relation *[inferred]* |
| `User`, `Owner`, `Creator`, `Consumer`, `Bearer` | possession/agency-of-a-relation roles *[inferred from name]* |
| `Instance` | instance-of / type relation *[inferred]* |
| `Sub` | subset-member relation; used to build plural/collective referents (see §4.6) |
| `Part` | part-of relation, e.g. `member.n.01 PartOf -1` |
| `MadeOf` | material-composition relation. Confirmed present in `sbn_spec.py` **only** in its `Of`-suffixed form — there is no corresponding bare `Made` role in the implemented set, unlike most other `Of`-paired roles (`Attribute`/`AttributeOf`, `Theme`/`ThemeOf`) which have both forms |
| `FeatureOf` | has-feature relation. Same asymmetry as `MadeOf`: no bare `Feature` form is implemented |
| `Content` | contains-content relation |
| `Player` | present in `sbn_spec.py`, not in the 2025 paper's taxonomy; no gloss in any source consulted *[undocumented]* |
| `Operand` | present in `sbn_spec.py`, not in the 2025 paper's taxonomy; no gloss in any source consulted *[undocumented]* |
| `Equal` | present in `sbn_spec.py` as a role, distinct from the `EQU` comparison operator (§2.4); no gloss found *[undocumented]* |
| `Precondition` | present in `sbn_spec.py` as a role — note this is a second instance of a name shared across categories, since `PRECONDITION` is also a separator (§2.7); case-formatting is what distinguishes them, not the string itself |

**A caution on this table.** Roles marked *[inferred]* or *[undocumented]* have no prose definition in any of Bos's papers; where a role is confirmed to exist in the reference parser (`sbn_spec.py`, PMB toolchain) but has no gloss anywhere, that is stated explicitly rather than guessed at. Treat *[inferred]* glosses as a starting hypothesis to check against the corpus, and *[undocumented]* entries as open questions, not settled definitions.

### 2.4 Operators (comparison constraints)

Where a role expresses a thematic or relational connection, an **operator** expresses a *comparison* between two entities, times, or values — equality, inequality, ordering, or temporal/spatial relation. Bos states that operators can be written either as mathematical symbols (`=`, `≠`, `≈`, `<`, `≤`, `≺`) or as fixed three-uppercase-letter codes; in every actual SBN string in the corpus and in every worked example across these papers, the **letter-code form is what is actually used** — the symbolic form appears only in prose or in DRS-box renderings, never in the linear text format itself. Use the letter codes.

The table below starts from the inventory in Zhang, Bouma & Bos (2025, Appendix C), cross-checked against the reference parser's `DRS_OPERATORS` set (`sbn_spec.py`), which confirms all 17 of the paper's operators and adds eight more that don't appear in any paper consulted for this manual:

| Code | Meaning | Source |
|---|---|---|
| `EQU` | equal | paper + code |
| `NEQ` | not equal | paper + code |
| `APX` | approximately equal | paper + code |
| `LES` | less than | paper + code |
| `LEQ` | less than or equal | paper + code |
| `MOR` | greater than | paper + code |
| `TOP` | not more than | paper + code |
| `TPR` | temporally precedes (before) | paper + code |
| `TSU` | temporally succeeds (after) — flagged with a literal `# What does this mean?` comment in the reference parser's own source, i.e. even its maintainer registered this as under-documented | paper + code |
| `TIN` | temporal inclusion | paper + code |
| `TCT` | temporal contains | paper + code |
| `TAB` | temporal abut | paper + code |
| `ANA` | anaphoric link | paper + code |
| `SXP` | spatially behind | paper + code |
| `SXN` | spatially before | paper + code |
| `SZN` | spatially under | paper + code |
| `SZP` | spatially above | paper + code |
| `STI` | inside | code only |
| `STO` | outside | code only |
| `SY1` | beside | code only |
| `SY2` | between | code only |
| `SXY` | around | code only |
| `BOT` | *[undocumented — no gloss in any source consulted]* | code only |
| `ESU` | *[undocumented — no gloss in any source consulted]* | code only |
| `EPR` | *[undocumented — no gloss in any source consulted]* | code only |

**Two naming traps to watch for**, worth stating explicitly for a critical/operational reading:

- **`TOP` the operator** ("not more than") is unrelated to the "Top" label sometimes seen marking the root of the role-hierarchy diagram, and unrelated to the special TOP triple that Smatch scoring discards when comparing graphs. Three different things share the string "TOP."
- **`GRE`**, used for "greater than" in Bos's 2021 ISA-17 examples, does not appear in the 2025 operator table, which instead uses `MOR`. Treat `MOR` as the current form and `GRE` as an earlier synonym you may still meet in older PMB-derived material.

An operator attaches directly after a concept the same way a role does, taking either a constant or an index as its target: `quantity.n.01 EQU 61`, `time.n.08 TPR now`, `person.n.01 EQU speaker`.

### 2.5 Indices, anchors, and hooks

An **index** is a signed integer replacing a variable. Indices are counted relative to the current concept's position in the sequence, de-Bruijn style:

- `0` refers to the current (most recently introduced) concept.
- Negative indices count backward: `-1` is the concept introduced immediately before the current one, `-2` the one before that, and so on.
- Positive indices count forward: `+1` is the next concept to be introduced, `+2` the one after that.

**0-drop.** In a role or operator condition, one of the two arguments is always the current concept itself, i.e. index `0` — and that argument is *never written*. Only the other (target) index appears. So `Agent -1` is short for "the current event's Agent is the concept at relative position -1"; you never write `Agent 0 -1`.

Combining a role/operator with its target gives you one of two structures:

- **Anchor** = role/operator + **constant**: grounds a concept to an external literal, e.g. `Name "Mary"`, `TPR now`.
- **Hook** = role/operator + **index**: connects a concept to another concept in the sequence, e.g. `Agent -1`, `EQU -2`.

A concept may be followed by zero or more anchors/hooks, chained one after another: `old.a.01 AttributeOf -3 Value -2` attaches two separate hooks to the same concept. An index that never resolves to an antecedent concept (nothing at that relative position) is the SBN equivalent of a free variable — legal in an open/unfinished fragment, but not in a well-formed, closed representation (§3.5).

### 2.6 Contexts

A context is SBN's analogue of a DRS box: a set of concepts (with their anchors/hooks) that are "in scope together." Contexts are never written explicitly as a token — there is no bracket or box symbol. A context simply *is* the run of concepts between the start of the sequence (or the previous separator) and the next separator. A sequence with **n** separators has exactly **n + 1** contexts.

### 2.7 Separators

A separator marks a boundary between two contexts and states what logical or rhetorical relationship holds between them (negation, discourse relation, and so on).

**Format:** all-uppercase, no lower-case letters, distinguishing it visually from both concepts (lower-case) and roles (mixed case). A separator is always immediately followed by a **connector** (§2.8).

The table below is the inventory from Zhang, Bouma & Bos (2025, Appendix C) — the most recent and most complete formal listing, and the one this manual treats as authoritative:

| Separator | Role |
|---|---|
| `NEGATION` | local negation; also used to build disjunction and universal quantification (§4.1–4.3) |
| `CONJUNCTION` | merges the following material back into an existing context rather than opening a new one (§4.3) |
| `ALTERNATION` | rhetorical alternation |
| `ATTRIBUTION` | rhetorical attribution (reported content) |
| `CONDITION` | rhetorical conditional relation |
| `CONSEQUENCE` | rhetorical consequence relation |
| `CONTINUATION` | rhetorical continuation |
| `CONTRAST` | rhetorical contrast |
| `EXPLANATION` | rhetorical explanation |
| `NECESSITY` | rhetorical necessity/modal relation |
| `POSSIBILITY` | rhetorical possibility/modal relation |
| `PRECONDITION` | rhetorical precondition |
| `RESULT` | rhetorical result |
| `SOURCE` | rhetorical source-attribution |
| `ELABORATION` | rhetorical elaboration |
| `COMMENTARY` | rhetorical commentary |

**A discrepancy worth flagging rather than silently resolving.** `NARRATION` — used explicitly as a discourse relation in Bos's own worked examples in both the 2023 IWCS paper (the Max/evening example) and the 2021 Variable-free DRS paper (the same example, developed further) — is **absent** from the 2025 Appendix C table above. This manual cannot tell you with confidence whether `NARRATION` has been formally retired in favour of `CONTINUATION`, was simply omitted from the appendix because it didn't occur often enough in the taxonomical-encoding training data behind that table, or is still valid and the appendix is incomplete. Treat the Appendix C list as the operational default (it is tied to the most recent formalisation and to the actual PMB toolchain), but do not be surprised to encounter `NARRATION` in existing PMB-derived data, and flag it for clarification with Bos's group rather than silently normalising it to something else.

A related historical note: **`PRESUPPOSITION`** appears as a separator in the 2021 Variable-free DRS paper, used to project a proper name's declaration to an outer context while leaving a co-indexed trace behind (§4.5), but it likewise does not appear in the 2025 Appendix C table. The same caveat applies.

**Cross-checked against the reference implementation** (`sbn_spec.py`'s `NEW_BOX_INDICATORS` set): this confirms, independently of the 2025 paper, that `NARRATION` and `PRESUPPOSITION` are absent from the current separator inventory — two independent sources now agree on this, which is stronger evidence than either alone. But the same check surfaces a *new* discrepancy in the other direction: **`COMMENTARY`**, present in the 2025 paper's Appendix C table, is **absent from the reference parser's separator set**. This manual cannot resolve why; it may postdate the parser's baseline, or be paper-only and never implemented. If your own toolchain's `sbn_spec.py` differs from the one checked here, treat this as a prompt to re-verify against your actual running version rather than against this manual.

Two separators deserve special operational attention because their effect is not "introduce a nested/subordinate context" the way the others are:

- **`NEGATION`** does not introduce a new discourse referent or role. It packages everything after it as the negative complement of what came before. Disjunction and universal quantification are both built out of chained `NEGATION`s (§4.1–4.3) rather than having dedicated separators of their own — a deliberate design choice to avoid adding binary discourse relations (which would need well-formedness pairing, e.g. an `ANTECEDENT` always needing a matching `CONSEQUENCE`) to the inventory.
- **`CONJUNCTION`** is the odd one out structurally: it does *not* open a new context at all. It threads the material that follows it back into whatever context its connector points to. This is what makes it usable to give a quantified object wide scope over its containing clause (§4.3) — a technique the notation borrows from presuppositional accommodation and DRS-merging.

### 2.8 Connectors

A connector is attached directly to a separator and specifies which earlier- or later-introduced **context** (not concept — this is a context-level index, a different counting space from the entity-level indices of §2.5) the new material attaches to.

**Format:** `<N` for backward (the N-th previously introduced context, counting the separator's own position as the anchor) or `>N` for forward.

- `NEGATION <1` — the negated content attaches to the immediately preceding context.
- `NEGATION <2` — the negated content attaches to the context *before* the immediately preceding one (used when two negations are chained inside the same sequence and need to attach to different levels, e.g. "She is neither rich nor famous").
- Every newly introduced context connects to **exactly one** previously (or, less commonly, subsequently) introduced context — a context can never be linked to more than one other context. Usually the connector points to the adjacent context, but non-adjacent connectors occur with wide-scope readings, presupposition accommodation, non-local discourse relations, and the double-negation encoding of disjunction.
- The forward form (`>N`) is listed in Bos's own connector inventory alongside `<1`, `<2`, but no worked example of it as a *separator's* connector appears in any of the source papers read for this manual, so that specific use remains undocumented.

**Empirical addendum (gold PMB data, not from Bos's papers).** A targeted search of the gold training set (172 total occurrences of `>` in the corpus, all of them of the exact form `Proposition >1`) shows that forward-pointing indices in practice are reserved almost entirely for one specific construction: a clause-embedding predicate (*doubt, know, admit, claim, be convinced*, …) whose `Proposition` role takes an entire embedded clause as its argument, paired with a `CONTINUATION <0` separator that opens that clause as a new context. Position-counting confirms `Proposition >1` is **not** ordinary concept-level forward counting — across several examples the target concept sits two or more positions forward in the raw sequence, yet the index used is always `+1` — so `Proposition >N` is better understood as a **box-level** pointer ("my argument is the next context, as a whole") reusing entity-index syntax, rather than as a concept-level hook. Correspondingly, `CONTINUATION <0` does not fit the "`<N` = N contexts back" rule confirmed above for `NEGATION` and `EXPLANATION` (where `<1` reliably means "the immediately preceding context"): `<0` is used uniformly regardless of how many prior contexts exist. The most defensible reading is that `Proposition >1 … CONTINUATION <0` is a fixed, matched-pair idiom for finite clausal complementation, not a general instance of the connector-counting rule — and it is not discussed in any of the four Bos papers surveyed for this manual. This is worth verifying directly against `sbn_spec.py`/`graph_base.py` (or with Bos's group) rather than treated as settled by this manual, since it appears to be an implementation convention that predates or sits outside the published notation papers.

**A notational-evolution note.** Bos's earlier papers (2021) write the context-connector as a bare signed integer directly after the separator name with no chevron — e.g. `NEGATION -1`, or occasionally the integer written *before* the separator, `-1 NEGATION` (both orderings occur in that paper's figures, most likely because those examples are transcribed from box diagrams rather than typed as running SBN text). The chevron form (`SEPARATOR <N`) is what the 2023 paper introduces, is what all later papers (2025) consistently use, and is what actual PMB-derived data (confirmed against a real annotated example) uses. This manual treats the chevron form as the current standard and the bare-integer form as superseded, but the two are semantically identical: both are a context-level backward index.

### 2.9 Comments (not part of the formal notation)

Working SBN files conventionally carry a `%` comment at the end of each line, giving the word(s) of the source sentence that line corresponds to:

```
male.n.02 Name "Tom"      % Tom
time.n.08 TPR now         % was
cry.v.02 Agent -2 Time -1 % crying.
```

This is explicitly **not part of the meaning representation** — Bos states this directly — but is standard practice for readability, alignment verification, and (in the PMB corpus specifically) can extend to character-offset spans, e.g. `% Life never [0-10]`. Do not treat text after `%` as contributing to well-formedness or to evaluation.

---

## 3. Grammar: how the vocabulary combines

### 3.1 Formation rules, stated compactly

```
sequence      ::= context (separator context)*
context       ::= member+
member        ::= concept attachment*
attachment    ::= role target | operator target
role          ::= UpperInitial (mixed case, optional "Of" suffix)
operator      ::= three-letter uppercase code, from the fixed inventory (§2.4)
target        ::= constant | index
concept       ::= lowercase-lemma "." pos "." sense
constant      ::= '"' literal '"' | number | "speaker" | "hearer" | "now" | "here"
index         ::= "-" positive-integer | "+" positive-integer      (0 never written)
separator     ::= ALL-UPPERCASE word, from the fixed inventory (§2.7)  connector
connector     ::= "<" positive-integer | ">" positive-integer
```

This is a simplification for orientation, not a machine-checkable grammar (in particular it doesn't encode the referential constraints in §3.5, which are closer to a type system than to context-free syntax) — but it captures the shape every legal SBN string has.

### 3.2 Building a context: concepts, anchors, and hooks

Within a single context, you introduce a member (usually one concept per lexical content word — noun, verb, adjective, or adverb), optionally followed by one or more attachments chaining role/operator + target pairs. A bare concept with no attachments at all is legal (`book.n.02` on its own is a complete, if minimal, context member). A concept can carry several attachments in a row: `male.n.02 Title "Mister" Name "Vinken" EQU -10` chains two anchors and one hook onto a single concept.

**Default ordering.** There is no formal requirement that concepts appear in a particular order — DRT's discourse referents and conditions are, strictly speaking, unordered sets — but by convention the order of members within and across contexts mirrors the surface word order of the sentence being represented, and the choice of positive vs. negative index is what lets that surface order be preserved even when a role's two arguments occur out of "natural" order in the representation. Order is not merely a readability nicety here, though: because indices are positional, changing the order of members *does* change what a given index refers to, so reordering is not meaning-preserving without also updating every index. Because the convention is surface-order alignment, SBN incidentally also has a way to represent information structure: the same event described actively vs. passively, or the same content translated into a language with different word order, will legitimately produce different index patterns for a truth-conditionally identical meaning.

### 3.3 Building a complex sequence: separators and connectors

A single context by itself (a "simple sequence" in Bos's terms) represents one box. To represent anything requiring more than one box — negation, discourse relations, most quantification — you insert a separator plus connector at the point where one context ends and the next begins. A sequence with `n` separators therefore always has exactly `n + 1` contexts, and each context after the first must have a connector (attached to the separator that introduces it) pointing to exactly one earlier or later context.

### 3.4 Role inversion and 0-drop working together

Because 0-drop means only one argument of a role/operator is ever written, and role inversion means either argument can be made the implicit "current concept," the two combine to give considerable freedom in how a sentence's role structure is aligned to its surface order. As a limiting case, positive indices can be eliminated from a representation entirely by choosing, for every role, whichever direction (base or `Of`-inverted) keeps its target concept already-introduced (negative index) rather than not-yet-introduced (positive index) — at the cost of the resulting representation tracking surface word order less closely.

### 3.5 Well-formedness constraints

These are scattered across the source papers rather than presented as a single checklist there; this section collects them into one. A representation satisfying all of the following is what Bos calls a *closed* sequential meaning — the only kind that can be translated into an ordinary DRS at all:

1. **Every index resolves.** Every hook's index must have an antecedent concept at that relative position within a context the current context can see (see constraint 4). An index with no antecedent is a free variable — permitted only in a deliberately open/underspecified fragment, never in a finished annotation.
2. **Every connector resolves.** Every separator's connector must point to a context that actually exists in the sequence (not past either end).
3. **One incoming connection per context.** A newly introduced context connects to exactly one other context — never zero (except the very first context, which needs none) and never more than one.
4. **Accessibility mirrors DRT subordination.** A negated (subordinate) context cannot be the target of an index or connector originating from an unrelated context, and material inside a negated context can only refer backward (negative index) to material in the contexts that subordinate it — not forward into sibling or outer material via a positive index. This is the same accessibility restriction familiar from classical DRT, just re-expressed over indices and connectors instead of box nesting.
5. **Concepts are lower-case WordNet triplets; roles are mixed-case; separators are all-uppercase; operators are one of the fixed three-letter codes; constants that are literal strings are double-quoted.** A string that violates the surface-formatting conventions in §2 is not parseable regardless of whether points 1–4 hold.
6. **A separator is always immediately followed by a connector** — there is no such thing as a bare, unattached separator.

### 3.6 Graph correspondence

Every well-formed SBN sequence corresponds to a rooted directed acyclic graph, and this correspondence is itself a practical verification tool: if you can draw the graph without needing to guess where an edge should point, the sequence is almost certainly well-formed.

- **Concept nodes** are drawn as ovals.
- **Context nodes** are drawn as boxes.
- Every concept has a **membership edge** connecting it to the context (box) it belongs to.
- **Role and operator attachments** become labelled directed edges between two concept nodes (or a concept node and a literal, for anchors).
- **Separators** become labelled edges between two context (box) nodes.
- The sequence notation is a **topological ordering** of this graph. Because a DAG can admit more than one valid topological ordering, the "correct" one for annotation purposes is whichever ordering best mirrors the linguistic surface realisation — meaning that translating the same meaning into a different source language, or into a passive rather than active construction, can validly produce a different linear SBN string for graph-isomorphic content.

**Confirmed against the reference implementation** (`graph_base.py`/`sbn_spec.py`, PMB toolchain): there are exactly three node types (`SYNSET`, `CONSTANT`, `BOX` — confirming that constants get their own distinct visual/structural node category, separate from concepts, rather than being mere labels on an edge) and five edge types (`ROLE`, `DRS_OPERATOR`, `BOX_CONNECT`, `BOX_BOX_CONNECT`, `SYN_BOX_CONNECT`). The first two edge types map straightforwardly onto "role and operator attachments" above, and `BOX_BOX_CONNECT` onto "separators become labelled edges between boxes." `BOX_CONNECT` and `SYN_BOX_CONNECT` both sound like variants of the "membership edge" described above, but the two files available don't show which token maps to which — the concrete graph-building logic (`from_string`) is abstract in `graph_base.py` and implemented in a subclass not covered here. Treat the five-way split as confirmed and the exact mapping of the two membership-edge variants as an open question.

---

## 4. Representing standard semantic phenomena

### 4.1 Negation

`NEGATION` packages everything after it (up to the next separator or end of sequence) as the negated complement of the preceding context:

```
person.n.01           % Somebody
NEGATION <1            %
buy.v.01 Agent -1 Theme +1  % bought
book.n.02              % no book.
```

### 4.2 Disjunction

There is no dedicated disjunction separator. `(p ∨ q) ≡ ¬(¬p ∧ ¬q)` is used directly: disjunction is written as a negation whose content is itself two chained negations, one per disjunct, each attaching back with `<1` (so that both are negated relative to the *same* outer context, not relative to each other). This scales to any number of disjuncts without adding new machinery.

### 4.3 Universal quantification

There is no dedicated universal-quantifier separator either (and, per Bos's own critical reflection on this design choice, generalised quantifiers beyond universal — *most*, *few*, and so on — are **not** supported by original SBN at all; see §6). Universal quantification over a variable is written using `(p → q) ≡ ¬(p ∧ ¬q)`:

```
NEGATION <1
person.n.01             % Everyone
NEGATION <1
smoke.v.01 Agent -1      % smoked.
```

**Wide-scope quantified objects** are the one case that requires an extra device. A universally quantified noun phrase in *object* position needs to scope over the whole clause, but the object concept is naturally introduced late (after the verb, in English surface order) — after both `NEGATION`s have already been opened. `CONJUNCTION` solves this: it is inserted at the point where the object needs to be *semantically* accommodated, and its connector threads the following material back into the already-open negated context rather than opening a further nested box, giving the object effectively the same wide scope a raised quantifier would get in classical DRT.

### 4.4 Rhetorical / discourse relations

Any of the fourteen non-negation, non-conjunction separators in §2.7's table connects two adjacent (or, for non-local relations, non-adjacent) contexts and asserts that a specific rhetorical relation — in the SDRT sense — holds between them. Unlike full SDRT, where a discourse unit can itself recursively be a compound of other units and relations between units, SBN's discourse relations always connect two *single* contexts. This is a deliberate simplification: Bos notes that SDRT-style indirect relation inheritance (e.g. a `NARRATION` on top of an `ELABORATION` automatically implying a further `ELABORATION` relation) is not automatically captured and would require separate background inference rules layered on top of SBN if needed. Anaphoric reference to a *compound* discourse unit (as opposed to a single context) is likewise not supported by the base notation without additional machinery.

### 4.5 Presupposition and proper-name projection

Classic DRT observation: proper names introduced inside a subordinate context (e.g. inside the scope of a negation) still need to be available for anaphora outside that scope, so they conventionally "float" to the outermost level. SBN's operational solution is to place the name's full declaration at the outer level and leave a co-indexed "trace" reference at the position where it was semantically triggered, rather than trying to encode a cross-level index directly (which Bos judged too difficult to interpret reliably). Concretely:

```
person.n.01 Name "Tim"
PRESUPPOSITION <1
time.n.08 EQU now
NEGATION <1
person.n.01              % Everyone
NEGATION <1
see.v.01 Experiencer -1 Stimulus +1 Time -2   % saw
person.n.01 Name "Tim" EQU -4    % Tim (trace, co-indexed with the outer declaration)
bar.n.01 LocationOf -2           % in a bar.
```

(`PRESUPPOSITION` is one of the separators flagged in §2.7 as present in the 2021 paper but absent from the 2025 inventory — apply the same caution noted there.)

### 4.6 Plural and collective referents

To conjoin two named entities into a single plural discourse referent, introduce the entities, then introduce an `entity.n.01` concept whose two `Sub` hooks point backward and forward to the two conjuncts:

```
male.n.02 Name "Bert"
entity.n.01 Sub -1 Sub +1    % and
female.n.02 Name "Alice"
```

The resulting `entity.n.01` referent can then itself be the target of further roles (e.g. as the joint Agent of a following event), giving plural/collective reference without a dedicated plural-marking mechanism.

### 4.7 Tense and time

Tense is not a grammatical feature of the concept token itself; it is expressed compositionally, the same way any other participant is attached to an event. Introduce a `time.n.08` concept, relate it to `now` with the appropriate operator (`TPR` for past/precedes, `TSU` for future/succeeds, `EQU` for present), and attach it to the event concept via the `Time` role:

```
time.n.08 TPR now
smile.v.01 Time -1
```

### 4.8 Alignment and role inversion in practice

Because index direction and role inversion are both free choices, the same meaning can be written to track surface word order more or less tightly. Giving up "0-drop" (i.e. allowing both arguments of a role to be written as explicit non-zero indices) buys still-better alignment for long-distance phenomena like PP-fronting or topicalisation, at a cost to readability; the convention in practice is to keep 0-drop and accept imperfect alignment in those specific cases rather than complicate every ordinary sentence.

---

## 5. Evaluation (operational summary)

Bos does not propose new evaluation machinery for SBN; the recommended path (Poelman et al., cited by Bos 2023) is to convert an SBN graph to PENMAN format and then score it with **Smatch**: maximise the number of matching triples between predicted and gold graphs under a variable/index-renaming, and take the harmonic mean of precision and recall over that maximum. Two operational points to keep in mind:

- **Normalise role inversion before scoring.** Because `R(x,y)` and `RangeOf(y,x)` are the same fact written two ways, a naive triple comparison would wrongly penalise a correct-but-differently-inverted prediction; inverted roles are normalised to their base form first.
- **Well-formedness is a gating check, separate from Smatch itself.** A string that fails to parse into a graph at all, or parses but fails structural validity (e.g. a dangling index, or a truncated sequence cut off before the sequence closes), cannot be meaningfully Smatch-scored and should be reported as a distinct failure category rather than folded into a low F1 score.
- Smatch, in any of its published variants, flattens the graph to a triple set and is therefore **blind to quantifier scope** — two representations that differ only in the scope assigned to a quantifier can receive an identical Smatch score. This is a known limitation of the metric, not of SBN, but it matters for interpreting evaluation results.

---

## 6. Known limitations of original SBN (Bos's own assessment)

Stated directly by Bos, and worth keeping in front of you when deciding whether a given sentence is even in scope for the notation as defined:

- **No generalised quantifiers.** Only universal quantification (via double negation) is supported. *Most*, *few*, and other non-universal quantifier meanings are not accounted for in the base formalism; Bos sketches, but does not adopt, a possible extension using a new coordinated pair of separators.
- **Universal quantification and presupposition accommodation both require "movement"** of semantic material to an earlier position to get scope correct, which Bos explicitly flags as the hardest part of manual annotation in this notation.
- **Factives, focus particles, and generics** have no attractive annotation solution in SBN as of Bos's own annotation exercise (ISA-17 shared task write-up) — these were left unresolved, not merely difficult.
- **Anaphoric reference to compound discourse units** (as opposed to single contexts), and the `CONTRAST`/`PARALLEL`-type SDRT relations that depend on it, are not directly supported; some kind of summation operation analogous to split-antecedent plural pronouns would be needed.
- **No automatic inheritance of indirect discourse relations.** SDRT's automatic derivation of a further relation from a chain of two others (e.g. a `NARRATION` nested under an `ELABORATION` implying an outer `ELABORATION`) does not happen automatically in SBN; it would require separate inference rules.
- **Singular/plural is not grammatically distinguished** — the underlying model theory treats all entities as potentially set-valued, so singular and plural noun phrases are not given different representational treatment beyond what §4.6 shows for explicit conjunction.

**Separately: limitations found in the reference implementation, not stated by Bos.** These come from `sbn_spec.py` (PMB toolchain) rather than from any paper, and are implementation facts rather than design choices Bos has written about:

- **Role inversion is implemented as a small closed whitelist**, not the fully general mechanism the theory describes (§2.3) — only ten `Of`-forms are recognised.
- **A signed numeric constant is indistinguishable from an index** at the tokenizer level (e.g. a literal value of `-1` versus a backward index `-1`), acknowledged directly in the parser's own source comments as an open problem.
- **`here`**, listed by Bos as a deictic constant, does not appear in the reference parser's constant set — see §2.2.
- Several roles and operators exist in the reference parser with **no gloss in any source consulted for this manual** (`Player`, `Operand`, `Equal`, `Cause`, `Affector`, `Measure` among roles; `BOT`, `ESU`, `EPR` among operators) — flagged throughout §2.3–2.4 as *[undocumented]* rather than guessed at.

---

## 7. Quick-reference index

- Concept: `lemma.pos.sense`, lower-case (§2.1)
- Constant: `"string"` / number / `speaker`/`hearer`/`now`/`here` (§2.2)
- Role: `UpperCamelCase`, optional `Of` suffix for inversion (§2.3)
- Operator: fixed 3-letter uppercase code (§2.4) — table above
- Index: `-N` / `+N`, `0` never written (§2.5)
- Anchor = role/operator + constant; Hook = role/operator + index (§2.5)
- Context: implicit, delimited by separators (§2.6)
- Separator: ALL-UPPERCASE, fixed inventory (§2.7) — table above
- Connector: `<N` / `>N`, attached to a separator, indexes *contexts* not concepts (§2.8)
- `%` comment: alignment aid only, not part of the formal representation (§2.9)

