"""Lower bound of the candidate band: reject reparanda that are synonyms.

Why not a cosine.  The obvious instrument -- embed both sentences and take the
cosine, as LARD does -- cannot work here, and fails in the direction that hurts
most.  Distributional similarity does not separate synonymy from antonymy:
`hot`/`cold`, `cheap`/`expensive` and `big`/`small` occur in near-identical
contexts because they *are* the same semantic dimension, so every embedding
space scores them as highly similar.  LARD is unaffected because it maximises
similarity for coherence.  We need the opposite discrimination: an antonym is
one of the best reparanda we can produce (`I can only afford the big, small
room`), a synonym is the one we must reject.

So the instrument has to test the *relation*, not the distance.  Natural
language inference does exactly that, and its three labels map onto the
contrast-set criterion directly:

    mutual entailment   the two are interchangeable      -> synonym, reject
    contradiction       incompatible values, one slot    -> keep, this is the
                                                            contrast we want
    neutral             different but compatible         -> keep

"Same slot, incompatible value" is, in NLI terms, "not entailment".

Model: `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (184M, MIT), trained on
MNLI + FEVER + ANLI.

Caveat worth keeping in view: NLI models are trained on naturally occurring
premise/hypothesis pairs, not on lexical-substitution minimal pairs, so this is
mildly out of distribution.  The scores must be validated against a
hand-labelled sample before any threshold is trusted -- which is the same
calibration step DESIGN.md §5.2 already requires.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

MODEL_ID = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Provisional, NOT calibrated.  Kept as a module constant so the calibration
# step has one place to write its result.
SYNONYMY_REJECT_ABOVE = 0.5


@dataclass(frozen=True)
class Verdict:
    """NLI reading of one (clean, substituted) sentence pair."""

    entail_fwd: float       # P(clean entails substituted)
    entail_bwd: float       # P(substituted entails clean)
    contradiction: float    # strongest contradiction reading of either direction
    neutral: float          # weakest neutral reading of either direction

    @property
    def synonymy(self) -> float:
        """Interchangeability: high only when entailment holds *both* ways.

        One-way entailment is hyponymy, not synonymy -- "I saw a dachshund"
        entails "I saw a dog" but not the reverse -- and a hyponym is a
        perfectly good reparandum.
        """
        return min(self.entail_fwd, self.entail_bwd)

    @property
    def too_similar(self) -> bool:
        return self.synonymy > SYNONYMY_REJECT_ABOVE


@lru_cache(maxsize=1)
def _model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device).eval()
    # Read the label order off the config rather than assuming it; checkpoints
    # differ, and getting this wrong inverts the filter silently.
    order = {v.lower(): k for k, v in mdl.config.id2label.items()}
    return tok, mdl, device, order


@torch.no_grad()
def _probs(pairs: list[tuple[str, str]], batch_size: int) -> torch.Tensor:
    """Softmax over (entailment, neutral, contradiction) for each pair.

    Batches are formed after sorting by length.  Padding is dynamic, so a
    batch costs whatever its longest member costs; mixing a 75-word sentence
    from `test/long.sbn` with 5-word ones from `test/standard.sbn` pads the
    whole batch to 75.  Lowering `max_length` would not help -- it is only a
    cap -- so the saving has to come from grouping similar lengths together.
    """
    if not pairs:
        return torch.empty(0, 3)
    tok, mdl, device, _ = _model()

    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    out = torch.empty(len(pairs), 3)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        chunk = [pairs[j] for j in idx]
        enc = tok([p for p, _ in chunk], [h for _, h in chunk],
                  return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(device)
        probs = torch.softmax(mdl(**enc).logits, dim=-1).cpu()
        for k, j in enumerate(idx):
            out[j] = probs[k]
    return out


def judge_pairs(pairs: list[tuple[str, str]],
                batch_size: int = 128) -> list[Verdict]:
    """Score (clean, substituted) sentence pairs for interchangeability.

    `substituted` is the clean sentence with one word swapped for a candidate
    reparandum -- *not* the repaired sentence.  We are asking whether the
    speaker could have meant the same thing by the other word, which is a
    question about the two words in one context, not about the repair.

    Both entailment directions are run, because synonymy is symmetric while
    entailment is not: "I saw a dachshund" entails "I saw a dog" one way only,
    and a hyponym is a perfectly good reparandum.
    """
    if not pairs:
        return []
    _, _, _, order = _model()
    e, n, c = order["entailment"], order["neutral"], order["contradiction"]

    fwd = _probs(pairs, batch_size)
    bwd = _probs([(h, p) for p, h in pairs], batch_size)

    return [
        Verdict(
            entail_fwd=float(fwd[i, e]),
            entail_bwd=float(bwd[i, e]),
            contradiction=float(max(fwd[i, c], bwd[i, c])),
            neutral=float(min(fwd[i, n], bwd[i, n])),
        )
        for i in range(len(pairs))
    ]


def judge(clean: str, variants: list[str], batch_size: int = 128) -> list[Verdict]:
    """Convenience wrapper: one clean sentence, many substituted variants."""
    return judge_pairs([(clean, v) for v in variants], batch_size)
