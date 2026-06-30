# Annotation Guideline: When does *be* get its own SBN node?

*Scope: this guideline covers **only** the binary decision — whether a `be` verb is
realised as an independent `be.v.*` node in the SBN sequence or is absorbed by another
constituent. **Which** sense (`be.v.01/02/03/…`) to assign is a separate question,
decided later. Derived from PMB 5.1.0 `en/train/gold.sbn` (3,483 be-form samples).*

## Core principle

SBN is **predicate-centred**: every clause needs one concept to carry the predication.
English *be* sometimes **is** that predicate, and sometimes is only grammatical glue
while another word predicates.

> **A `be.v.*` node is introduced if and only if no other concept in the clause can
> serve as the predicate.**
> If the complement supplies its own predicating concept (an adjective, an adverb, or a
> lexical verb), *be* is absorbed and does **not** appear in the SBN sequence.

**Operational test.** Look at how the material *after* `be` is analysed in SBN, **not**
at its surface part-of-speech:

- Is there an **adjective / adverb** attached by `AttributeOf` to the subject, or a
  **lexical (non-`be`) verb**? → *be* is absorbed, **no node**.
- Is the complement a **noun**, a **location**, a bare **existence** claim, or a
  **measure / price** — none of which can predicate on its own? → insert an independent
  **`be.v.*` node**.

The two sets below are **complementary and mutually exclusive**: every *be* falls into
exactly one.

---

## Set A — *be* is ABSORBED (no independent node)

Another concept already carries the predicate.

| # | Case | Why absorbed | Example | SBN (no `be.*`) |
|---|---|---|---|---|
| A1 | **Predicative adjective** | The adjective predicates via `AttributeOf` → subject | *The cause is unknown.* | `cause.n.01 … unknown.a.01 AttributeOf -3 Time -1` |
| A2 | **Comparative** (`-er / more / less / as … as`) | The adjective predicates; `more/less/as.r` only grades it | *Tokyo is bigger than Rome.* | `… big.a.01 AttributeOf -2 … more.r.01 … Co-Theme +1 city.n.01 Name "Rome"` |
| A3 | **Auxiliary — passive** | The lexical verb is the predicate; *be* marks voice | *The rock was moved by dynamite.* | `rock.n.01 … move.v.02 Theme -2 … Causer +1 dynamite.n.01` |
| A4 | **Auxiliary — progressive / perfect** | The lexical verb is the predicate; *be* marks aspect | *We are working for you.* | `… work.v.02 Agent -2 … Co-Agent +1 …` |
| A5 | **Predicative adverb** (`away`, `back`, `on board` …) | The adverb predicates via `AttributeOf` | *He is away.* | `… away.r.* AttributeOf …` (no `be.*`) |

---

## Set B — *be* is an INDEPENDENT NODE (`be.v.*` appears)

Nothing else can predicate, so *be* itself carries it.

| # | Case | Why a node is needed | Example | SBN (`be.*` present) |
|---|---|---|---|---|
| B1 | **Predicate noun** (class / identity / role) | A noun cannot predicate; *be* links subject to predicate noun via `Co-Theme` | *Turtles are reptiles.* / *Riga is the capital of Latvia.* | `… be.v.* Theme -1 … Co-Theme +2 … reptile.n.01` |
| B2 | **Locative** (*be somewhere*) | Position is asserted; no predicating word | *My house is on the south bank.* | `… be.v.03 Theme -1 … Location +3 …` |
| B3 | **Existential** (*there is / are*) | Bare existence; subject introduced by *be* | *There is a book on the table.* | `entity.n.01 be.v.03 … EQU -1 Theme +2 Location +3 … book.n.02 …` |
| B4 | **Measure / price** | A quantity / price is predicated | *How much is this watch?* | `… be.v.13 …` |

---

## Important caveat — judge the SBN analysis, not the surface POS

A surface adjective does **not** automatically mean Set A. Some "adjectives" are
**reanalysed** into a relation that cannot predicate, and they then take a *be* node:

- **Nationality / origin adjectives** → an origin relation to a country (`Source`),
  patterning like a locative → **Set B**.
  *My wife is Swedish.* → `… be.v.03 Theme -3 … Source +1 country.n.02 Name "sweden"`
  ✅ *be* node present.

> **Rule of thumb:** if the post-*be* word ends up attached by **`AttributeOf`** (a true
> predicate adjective / adverb) → Set A, no node. If it ends up as a **noun**,
> **`Location`**, **`Source`**, **existence**, or **measure** → Set B, insert a node.

---

## Empirical support

On the 3,483 be-form samples, the rule "insert a `be` node iff the predicate is nominal
or locative / existential / measure (no adjective, adverb, or lexical verb to carry the
predicate)" reaches **95.5% accuracy** (precision 94.1%, recall 89.0%). The residual
mismatches are dominated by an extraction artefact — **attributive** adjectives inside
locative / existential clauses (e.g. *There was a beautiful woman in the park*) being
miscounted as **predicative** — which the surface-vs-analysis caveat above resolves; the
true rule is cleaner than the raw figure.

