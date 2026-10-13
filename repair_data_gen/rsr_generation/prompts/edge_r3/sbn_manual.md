# SBN Annotation Manual (Core Notation)

This manual describes the Simplified Box Notation (SBN) as it is used in the
Parallel Meaning Bank (PMB) 5.1.0 gold data. It covers the base notation only.
The extension for self-repair (`CORRECTION` and the extra inverted roles) is
described in a separate document, which builds on this one.

Your output is read by the **SBN reader**, the program that converts an SBN
string into a graph. Output it cannot read is unusable.

Every inventory below (roles, operators, separators, constants) lists only
what occurs in PMB 5.1.0 English training gold **and** is accepted by the
SBN reader. Anything not listed should not be written. The number after
an item is its frequency in that gold set, as a guide to what is ordinary and
what is rare.

---

## 1. What SBN is

SBN writes a Discourse Representation Structure as a flat sequence of tokens
on one line. It has no variables and no explicit box brackets:

- Variables are replaced by relative **indices** (`-1`, `+2`), which point to
  other concepts by counting positions in the sequence.
- Box nesting is replaced by **separators** (`NEGATION <1`), which end one
  context and open the next.

Every well-formed SBN sequence corresponds to exactly one directed acyclic
graph (DAG): concepts and constants are nodes, contexts are box nodes, roles
and operators are labelled edges between nodes, and separators are labelled
edges between boxes.

A gold example ("Tom bought a new pick-up."):

```
male.n.02 Name "Tom" buy.v.01 Agent -1 Time +1 Theme +3 time.n.08 TPR now new.a.05 AttributeOf +1 pickup.n.01
```

---

## 2. Token types

### 2.1 Concepts

A concept introduces a discourse referent and classifies it with a WordNet 3.0
synset.

- Format `lemma.pos.sense`: `cat.n.01`, `see.v.03`, `new.a.05`, `fast.r.01`.
- `lemma` is lower-case; multi-word lemmas use an underscore
  (`suck_up.v.02`, `air_conditioning.n.01`).
- `pos` is one of `n`, `v`, `a`, `r`. WordNet's satellite adjectives (`s`)
  are written with `a` and keep their sense number: WordNet `boring.s.01`
  is `boring.a.01` in SBN.
- `sense` is the two-digit WordNet sense number.
- Usually one concept per content word, in surface order.
- Verbs are one-place predicates (neo-Davidsonian events). Their participants
  are attached by roles on the verb concept, not as arguments.
- `entity.n.01` is the general-purpose placeholder for a referent that needs
  no more specific type (pronouns like "it", plural referents, see §4.5).
- Persons and names: `male.n.02` (he/him), `female.n.02` (she/her),
  `person.n.01` (person, people, "I"/"you" with `EQU speaker`/`EQU hearer`).

### 2.2 Constants

A constant is a literal value at the end of a role or operator.

| Kind | Form | Example |
|---|---|---|
| deictic | `now` (10187), `speaker` (2934), `hearer` (518) | `time.n.08 TPR now`, `person.n.01 EQU speaker` |
| name string | double-quoted | `Name "Tom"`, `Name "New York"` |
| number | unquoted digits | `Quantity 2`, `YearOfCentury 1993` |
| plus/minus | bare `+` or `-` | `Value +` (comparative/high degree), `Quantity +` (plural, "several") |
| clock time | `HH:MM` | `ClockTime 22:00` |
| weekday | bare lower-case | `DayOfWeek monday` |
| single capital letter | bare | `Value A` |
| decade | single-quoted, last digit `X` | `YearOfCentury '198X'` ("the eighties") |
| question | bare `?` (437) | `country.n.02 Name ?` ("what country"), `person.n.01 EQU ?` ("who"); the slot a wh-question asks about, see §4.6 |
| score | two numbers joined by `-` (13) | `score.n.03 Value 3-0` ("3 to 0", "3:0") |
| fraction | two numbers joined by `/` (5) | `entity.n.01 Quantity 3/4` ("three-fourths") |

### 2.3 Roles

A role is a two-place relation from the concept it follows to a target (an
index or a constant). Roles start with an upper-case letter and contain
lower-case letters.

**Thematic roles** (event participants):

| Role | Gloss |
|---|---|
| `Agent` (4227) | intentional instigator of the event |
| `Co-Agent` (171) | second agent in a reciprocal event |
| `Causer` (118) | non-intentional instigator |
| `Stimulus` (870) | what triggers a perception or emotion |
| `Theme` (5651) | participant central to the event, not changed by it |
| `Co-Theme` (1160) | second theme; also the predicate nominal of a copula |
| `Patient` (1322) | participant that undergoes a change |
| `Co-Patient` (27) | second patient |
| `Experiencer` (1195) | aware participant in a perception or emotion; also the holder of an adjectival state ("he was tired") |
| `Pivot` (352) | participant more central than its partner (possession, "own") |
| `Topic` (130) | content of communication |
| `Proposition` (185) | a whole embedded clause as argument; always `Proposition >1`, see §2.6 |
| `Beneficiary` (177) | participant advantaged or disadvantaged |
| `Instrument` (150) | thing used by the agent |
| `Material` (34) | starting material transformed by the event |
| `Recipient` (309) | animate destination |
| `Goal` (63) | end point that exists independently of the event |
| `Result` (268) | end state brought about by the event; also the holder of some adjectival states |
| `Product` (84) | concrete object brought about by the event |
| `Location` (1107) | place |
| `Source` (514) | starting point |
| `Destination` (594) | concrete end point |
| `Path` (56) | route |
| `Manner` (177) | way the event is carried out |
| `Degree` (489) | degree of a gradable property |
| `Extent` (17) | amount of change |
| `Value` (570) | position on a scale |
| `Asset` (54) | value that is a concrete object (price, payment) |
| `Context` (46) | background circumstance |
| `Participant` (15) | participant of unspecified type |

**Time roles:**

| Role | Gloss |
|---|---|
| `Time` (10468) | links an event or state to its `time.n.08` referent |
| `Duration` (103) | length of time |
| `Frequency` (40) | how often |
| `Start` (13) / `Finish` (14) | onset / end point |
| `ClockTime` (152), `DayOfWeek` (16), `DayOfMonth` (47), `MonthOfYear` (59), `YearOfCentury` (129) | calendar and clock values, on a `time.n.08` concept |

**Attribute and relational roles:**

| Role | Gloss |
|---|---|
| `Name` (4034) | proper-name string |
| `Quantity` (1039) | number or amount |
| `Attribute` (274) | a property of an entity |
| `Colour` (74) | colour property |
| `Order` (30) | ordinal position |
| `Unit` (39) | unit of measurement |
| `Title` (67) | title or honorific string |
| `Role` (1195) | a person's role or function ("a teacher", "a grouch") |
| `User` (667), `Owner` (60), `Creator` (242), `Consumer` (22), `Bearer` (85) | relations of use, ownership, authorship (also possessives: "his behaviour" = `behavior.n.01 Creator -1`), consumption, bearing |
| `Sub` (796) | member of a plural referent, see §4.5 |
| `Part` (22) | has-part |
| `Instance` (6) | has-instance |
| `Content` (9) | contains |
| `Affectee` (6) | entity affected by a relation |
| `Of` (470) | generic relation ("the cause of", "gift as a") |

**Inverted roles.** Every role `R` has an inverse `ROf`: `R(x, y)` is the
same fact as `ROf(y, x)`. Inversion moves the role onto the other concept,
usually to keep surface order or to avoid a positive index. These two say the
same thing:

```
boy.n.01 buy.v.01 Agent -1 Theme +1 book.n.02
boy.n.01 buy.v.01 Agent -1 book.n.02 ThemeOf -1
```

Any role in the tables above can be inverted, with three exceptions:

- roles whose target is always a constant: `Name`, `ClockTime`,
  `DayOfWeek`, `DayOfMonth`, `MonthOfYear`, `YearOfCentury` (a constant
  cannot carry a role);
- `Proposition`, whose target is a box;
- `Of`, `MadeOf` and `FeatureOf`, which already exist only in an `Of` form.

An inverted role can also be turned back: `AttributeOf` on an adjective
becomes `Attribute` on the noun.

Gold uses only a few inverses, and mostly for attributes:

| Inverted role in gold | Use |
|---|---|
| `AttributeOf` (1802) | adjective → the entity it describes. The normal way to write an attributive or predicative adjective: `new.a.05 AttributeOf +1 pickup.n.01` |
| `PartOf` (572) | part → whole |
| `SubOf` (120) | member → group |
| `ColourOf` (80) | colour → entity |
| `InstanceOf` (37) | instance → type |
| `MadeOf` (38) | entity → its material |
| `ContentOf` (4) | content → container |
| `CauserOf` (2) | cause → what it causes |
| `AgentOf` (1) | agent → event |
| `FeatureOf` (1) | feature → entity |

For thematic roles, gold normally keeps the base form and uses a positive
index (`Theme +1`). Inverting them is legal and is used where a positive
index is not allowed (see the self-repair document).

### 2.4 Operators

An operator compares two referents or a referent and a constant. Operators are
three upper-case letters and attach exactly like roles.

| Operator | Meaning |
|---|---|
| `EQU` (9461) | equal; also present tense `time.n.08 EQU now`, and identity (`person.n.01 EQU speaker`) |
| `TPR` (4941) | temporally precedes; past tense `time.n.08 TPR now` |
| `TSU` (263) | temporally succeeds; future tense `time.n.08 TSU now` |
| `ANA` (893) | anaphoric link (a pronoun to its antecedent) |
| `TAB` (119) | temporal abut: x ends where y starts (`yesterday TAB today`) |
| `TIN` (60) | temporally included in |
| `TCT` (40) | temporally contains; "today" is `day.n.03 TCT now` |
| `APX` (83) | approximately equal ("near") |
| `NEQ` (32) | not equal ("others") |
| `MOR` (33) | more than ("over a thousand") |
| `LES` (9) | less than ("nearly ten") |
| `TOP` (10) | at most |
| `BOT` (6) | at least |
| `SZN` (35) | spatially under |
| `SZP` (7) | spatially above |
| `SXP` (12) | spatially behind |
| `STI` (18) | inside |
| `STO` (1) | outside |
| `SY1` (4) | beside |
| `SY2` (5) | between |
| `SXY` (7) | around |

**Inverting operators.** Operators can be inverted too (Bos 2021). They have
no `Of` form. A symmetric operator is simply moved to the other concept; an
ordering operator is replaced by its converse:

| Operator | Inverse |
|---|---|
| `EQU`, `NEQ`, `APX`, `SY1` | the same operator (symmetric) |
| `TPR` ↔ `TSU` | precedes ↔ succeeds |
| `TIN` ↔ `TCT` | included in ↔ contains |
| `LES` ↔ `MOR` | less than ↔ more than |
| `TOP` ↔ `BOT` | at most ↔ at least |
| `SZN` ↔ `SZP` | under ↔ above |
| `SXP` ↔ `SXN` | behind ↔ in front of |

`location.n.01 SZN +1 table.n.02` ("under the table") and
`location.n.01 table.n.02 SZP -1` say the same thing. `SXN` occurs only as
the inverse of `SXP`.

`ANA`, `TAB`, `STI`, `STO`, `SY2` and `SXY` have no inverse and cannot be
inverted. Operators whose target is a constant (`TPR now`) cannot be
inverted either.

### 2.5 Indices

An index is a signed integer that points to another concept. The current
concept is the one the role or operator is attached to, and it is never
written (so `Agent -1`, never `Agent 0 -1`).

- `-n` points to the concept `n` positions **before** the current one.
- `+n` points to the concept `n` positions **after** it.

**Counting rule.** Count every concept in the raw sequence, one by one.
Separators, roles, operators and constants do not count. Separators do not
reset or restrict the count: concepts in other boxes, including closed boxes,
still occupy positions. Example ("Even though he was tired, he worked."):

```
male.n.02 time.n.08 TPR now tired.a.01 Experiencer -2 Time -1 CONTRAST <1 male.n.02 ANA -3 work.v.01 Agent -1 Time +1 time.n.08 TPR now
```

`ANA -3` on the second `male.n.02` counts back over `tired.a.01` and
`time.n.08` (both in the first box) to reach the first `male.n.02`.

**Magnitude.** The SBN reader reads only one digit, so an index must be between
`-9` and `+9`. `-10` or larger is silently misread.

**Anchors and hooks.** A role or operator with a constant target is an
*anchor* (`Name "Tom"`, `TPR now`). With an index target it is a *hook*
(`Agent -1`). A concept can carry any number of them in a row:
`tall.a.01 Time -1 Value + AttributeOf -2`.

### 2.6 Separators and connectors

A separator ends the current context (box) and opens a new one. It is written
in capitals and is always followed immediately by a connector `<n`, which
says which earlier box the new box attaches to. Connectors count **boxes**,
not concepts: `<1` is the box immediately before the new one, `<2` the one
before that.

| Separator | Use | Usual connector |
|---|---|---|
| `NEGATION` (1482) | negation; also builds universal quantification (§4.2) | `<1` (1462), `<2` (18) |
| `CONTINUATION` (346) | next clause or sentence; also opens an embedded clause after `Proposition >1` | `<1` (156); `<0` (185) only after `Proposition >1` |
| `POSSIBILITY` (152) | possibility modal (can, may, might) | `<1` |
| `NECESSITY` (50) | necessity modal (must, should, have to) | `<1` |
| `CONJUNCTION` (85) | does not open a new scope: merges the following material into the box its connector points to | `<2` (73) |
| `CONTRAST` (67) | contrast, concession (but, although) | `<1` |
| `CONSEQUENCE` (42) | consequence (so, then) | `<1` |
| `EXPLANATION` (29) | explanation (because) | `<1` |
| `CONDITION` (5) | condition | `<1` |
| `PRECONDITION` (4) | precondition | `<1` |
| `RESULT` (4) | result | `<1` |
| `ALTERNATION` (4) | alternative (or) | `<1`, `<2` |

**`Proposition >1 … CONTINUATION <0`** is a fixed pair for clause-taking
predicates (say, know, think, according to). `Proposition >1` points to the
next box as a whole, so its target may be several concepts away;
`CONTINUATION <0` opens that box.

```
according.a.02 Proposition >1 Source +1 police.n.01 CONTINUATION <0 NEGATION <1 person.n.01 time.n.08 TPR now injure.v.01 Patient -2 Time -1
```
("According to the police, no one was injured.")

---

## 3. Well-formedness

A finished SBN must satisfy all of the following.

1. **Every index resolves**: there is a concept at the position it points to.
2. **Every connector resolves**: the box it points to exists.
3. **Every separator has exactly one connector**, written right after it.
4. **Accessibility.** Material inside a negated box can only be reached from
   that box or from boxes nested inside it. A concept in a negated box may
   point with a negative index to concepts in its *ancestor* boxes. Concepts
   in *sibling* boxes (two boxes that hang off the same parent) are not
   accessible to each other, even when one comes earlier in the sequence.
5. **Formatting**: concepts lower-case `lemma.pos.sense`; roles mixed case;
   operators three capitals; separators all capitals; name strings
   double-quoted. Every token is separated by a single space.
6. **The output is one line**, with no `%` comments.
7. **Only listed items**: roles, operators, separators and constants come from
   the inventories in §2.

---

## 4. Common constructions (gold examples)

### 4.1 Tense

A `time.n.08` concept anchored to `now`, linked to the event by `Time`:

```
female.n.02 Name "Naoko" POSSIBILITY <1 run.v.01 Theme -1 Manner +1 fast.r.01
male.n.02 time.n.08 TPR now bear.v.02 Patient -2 Time -1 Location +1 country.n.02 Name "America"
```
("Naoko can run fast." / "He was born in America.")

Past `TPR now`, present `EQU now`, future `TSU now`. The `time.n.08` concept
may come before or after the event; the index direction follows.

### 4.2 Negation and universal quantification

```
male.n.02 behavior.n.01 Creator -1 NEGATION <1 time.n.08 TPR now polite.a.01 AttributeOf -2 Time -1
NEGATION <1 person.n.01 NEGATION <1 like.v.03 Experiencer -1 Time +1 Stimulus +2 time.n.08 EQU now male.n.02
```
("His behavior was anything but polite." / "Everybody likes him.")

"Every X Ys" is "no X fails to Y": two nested `NEGATION <1`, the restrictor
in the first, the scope in the second.

A universally quantified noun that has to scope wide but appears late in the
sentence is brought into the right box with `CONJUNCTION`:

```
NEGATION <1 NEGATION <1 female.n.02 shower.v.03 Agent -1 Time +1 CONJUNCTION <2 morning.n.01
```
("She showers every morning.")

### 4.3 Clauses and discourse relations

```
person.n.01 meet.v.01 Theme -1 Time +1 time.n.08 EQU now CONTINUATION <1 person.n.01 quarrel.v.01 Agent -1 Time +1 time.n.08 EQU now
```
("Whenever they meet, they quarrel.")

### 4.4 Adjectives, copulas, possessives

```
tree.n.01 time.n.08 EQU now tall.a.01 Time -1 Value + AttributeOf -2
male.n.02 Name "Tom" be.v.01 Theme -1 Time +1 Co-Theme +2 time.n.08 EQU now person.n.01 Role +1 grouch.n.01
```
("This tree is tall." / "Tom's a grouch.")

An adjective attaches to its noun with `AttributeOf`. A noun predicate uses
`be.v.01` with `Theme` and `Co-Theme`.

### 4.5 Plurals and anaphora

```
glass.n.02 entity.n.01 Sub -1 Sub +1 dish.n.01 time.n.08 TPR now broken.a.01 Patient -3 Time -1
male.n.02 give.v.03 Agent -1 Time +1 Recipient +2 Theme +3 time.n.08 TPR now male.n.02 ANA -3 book.n.02
```
("Glasses and dishes were broken." / "He gave him a book.")

A coordinated plural is an `entity.n.01` with one `Sub` hook per member. A
pronoun that refers back is its own concept with `ANA` to its antecedent.

### 4.6 Questions

A wh-question marks the referent it asks about with the constant `?`, on the
role or operator that the answer would fill:

```
country.n.02 Name ? be.v.03 Source -1 Time +1 Theme +2 time.n.08 TPR now female.n.02 Name "Catherine the Great"
person.n.01 EQU ? be.v.08 Co-Theme -1 Time +1 Theme +2 time.n.08 EQU now male.n.02 Name "Niels Bohr"
person.n.01 Quantity ? Role +1 inhabitant.n.01 time.n.08 EQU now country.n.02 Name "Chechnya" have.v.01 Theme -4 Time -2 Pivot -1
old.a.01 Value ? Time +1 AttributeOf +2 time.n.08 EQU now male.n.02 Name "Brian Eno"
```
("What country was Catherine the Great from?" / "Who is Niels Bohr?" /
"How many inhabitants does Chechnya have?" / "How old is Brian Eno?")

| Question word | Written as (gold count) |
|---|---|
| which / what + noun | the noun with `Name ?` (`person.n.01` 105, `location.n.01` 32, `country.n.02` 6, `city.n.01` 5, …) |
| what | `entity.n.01 EQU ?` (69) |
| who | `person.n.01 EQU ?` (8) |
| where / why / how (manner) | `location.n.01 EQU ?` (21) / `reason.n.01 EQU ?` / `manner.n.01 EQU ?` (13) |
| how many / how much | `Quantity ?` on the counted concept (`person.n.01` 12); "how much" also `Unit ?` on `measure.n.02` |
| how + adjective | the adjective with `Value ?` (`old.a.01` 8, `tall.a.01` 5) |
| what time | `time.n.08 ClockTime ?` (5) |

Two points to keep apart:

- **Yes/no questions have no `?`.** "Is Tom retired?" is
  `time.n.08 EQU now male.n.02 Name "Tom" retired.a.01 Time -2 AttributeOf -1`,
  the same as the statement. The question mark in the sentence is not what
  triggers `?`; a questioned slot is.
- **Embedded questions do have `?`**, though the sentence is not a question:
  "Nobody knows where he lives." is
  `NEGATION <1 person.n.01 know.v.01 Proposition >1 Experiencer -1 Time +1 time.n.08 EQU now CONTINUATION <0 location.n.01 EQU ? male.n.02 live.v.01 Location -2 Theme -1 Time +1 time.n.08 EQU now`.

`?` is a constant, so it is an anchor (§2.5), never an index target.

---

## 5. Errors to avoid

Each line below is **wrong**. Some make the SBN reader fail; others are read
without complaint and give a wrong graph, which is worse because nothing flags them.

| Wrong | Why | Right |
|---|---|---|
| `new.a.05 AttributeOf+1` | no space between role and index; read as one unknown token, the SBN reader fails | `new.a.05 AttributeOf +1` |
| `time.n.08 Day "Monday"` | there is no `Day` role, and weekdays are bare lower-case constants | `time.n.08 DayOfWeek monday` |
| `male.n.02 NameOf "Tom"` | a role whose target is a constant cannot be inverted; the SBN reader fails | `male.n.02 Name "Tom"` |
| `NEGATION person.n.01 …` | separator without connector | `NEGATION <1 person.n.01 …` |
| `… Agent -12` | two-digit index, silently read as `-1` | reorder or restructure so the index stays within ±9 |
| `buy.v.01 Agent -1 Time +1 Theme +2 time.n.08 TPR now new.a.05 AttributeOf +1 pickup.n.01` | is read, but `Theme +2` lands on `new.a.05`, the adjective, instead of `pickup.n.01` (`+3`) | count every concept between the two, adjectives included |
| `… CONTRAST <1 male.n.02 ANA -1 …` (in the "tired" example of §2.5) | is read, but counts as if the new box restarted the count, so `ANA` points to `tired.a.01` | `ANA -3`, counting across the separator |
| `NEGATION <2` where `<1` is meant | is read, but attaches the new box to the wrong parent, giving two independent negations instead of a nested one | count boxes back from the new box |
