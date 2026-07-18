# General Pattern Observed
Co-Authored by Hongxu Zhou and Claude [16/July/2026]

## Data Source
The samples are selected from Gemma 2 9b parser's parsing result on the pilot dataset. 

The 107 rows span three groups:

- A_longitudinal (rows 2–50): 5 base sentences, each probed across all 7 conditions (gold, 3 repair variants, 3 interregnum variants). This is the systematic longitudinal view — how does the parser behave sentence-by-sentence as disfluency is introduced and compounded?
- B_failure_survey (rows 51–87): A curated set of failure cases from the broader run — heterogeneous sentences, selected because something went wrong.
- C_manual_repairs_known_defect (rows 88–108): Sentences where the repair structure is known to interact with a specific annotation quirk or ambiguity.

## Observation

Qualitative Analysis: How the Model Scores High Without Understanding Self-Repair

### The core observation

The model has no systematic understanding of self-repair. When presented with disfluent input, it does not identify and discard the reparandum. Instead, it strives to output a formally legal SBN sequence by glueing and sewing the disfluent material together — at the cost of the actual meaning. This behaviour, not repair competence, is what sustains a reasonably high score.

### Vanilla repair

When no interregnum is present, the model almost always retains both the reparandum and the repair in the output. The specific glueing device depends on where the reparandum appears:

- Head position: the reparandum becomes a dangling or weakly-connected node. The repair takes its normal structural role. The gold structure is largely intact; the reparandum adds unconnected material.
- Mid position: the model treats the two mentions as coordinated items, using entity.n.01 Sub -1 Sub +1 or parallel attribute chains. Both concepts are present; the repair lands in the correct structural slot alongside the reparandum.
- Tail position: the model serialises both via CONTINUATION, appending the repair after the reparandum.

In all three cases, the gold triples are a subset of the model's output — which is why precision drops but recall holds, and F1 remains acceptable.

### Interregnum repair

When "I mean" is present, two distinct behaviours emerge, and neither reflects an understanding of the interregnum's metalinguistic function.

Glueing: The model parses "I mean" as a propositional speech act (mean.v.01 Proposition <1 Agent speaker) and wraps the whole disfluency in a tripling structure: [reparandum subgraph] → CONTINUATION → mean.v.01 → CONTINUATION → [repair subgraph]. Both reparandum and repair are present, connected by discourse machinery that does not belong. When the surrounding sentence is structurally complex, this tripling corrupts the relative index counting and tips the output into a parse error — the glueing fails.

Selective omission: In a notable subset of cases, the repair is simply dropped and the model outputs clean SBN built from the reparandum alone. Counterintuitively, the score does not collapse, because the reparandum and repair share nearly the same structural frame — they differ in one substituted element while the surrounding event structure is the same. The wrong concept fills the right role; most triples still match.

### Why scores hold

The model's two strategies — additive glueing and selective omission — converge on the same outcome: a formally valid (or near-valid) SBN sequence that contains enough of the gold's structure to score reasonably. The score does not reflect repair competence. It reflects the fact that the gold meaning is either present inside a bloated, incoherent output, or approximated by a structurally similar but semantically wrong one. The gap between a decent F1 and an actual understanding of self-repair is only visible at the structural level: the spurious nodes, the misplaced discourse connectors, the cascading index errors, and the parse failures are the true signal.

## Possible Connection & Limitations
The quantitate analysis reports a highly complex pattern. It suggests that the "vanilla repair" vs. "interregnum repair" is more of a parallel structure instead of a stepwise one. Inserting "I mean" drastically the pattern observed on vanilla repair. My (Hongxu's) assumption is this may support my argue that interregna enact on the meta-linguistic layer, and should not be represented in MR. However, without more concrete evidence, the connection is clearly outstretching. 
