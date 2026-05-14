"""
truncation_detect.py
──────────────────────────────────────────────────────────────────────────────
Heuristic detector for obvious mid-token truncation in byT5-generated SBN
sequences.

'Obvious truncation' is defined as: the last whitespace-delimited token of
the SBN string does not match any complete, valid SBN token type.  This is a
conservative lower-bound proxy for all truncation: it catches only cases where
the 512-byte byT5 limit fires in the middle of a token, leaving a visible
fragment (e.g. 'R', 'Nam', 'Colo', '"Eng').  Clean token-boundary truncations
(where the output happens to end on a valid token but the sequence is
structurally incomplete) are NOT detected and are out of scope.

Token categories checked (derived from SBNSpec):
  1. Synset          word.pos.nn          e.g. person.n.01
  2. Role            fixed vocabulary     e.g. Agent, Theme
  3. DRS operator    fixed vocabulary     e.g. EQU, TPR, Name
  4. Box connector   fixed vocabulary     e.g. CONTINUATION, CONTRAST
  5. Relative index  (+/-/</>)\d       e.g. +1, -2, <0
  6. Complete quoted name  ^".*"$         e.g. "England"
  7. Pure integer    ^\d+$                e.g. 1993, 30, 4
  8. SBNSpec.CONSTANTS                    now, speaker, hearer, weekdays,
                                          unknown_ref

Note: VALUE_CONSTANT (^[A-Z]$, single uppercase letter) is deliberately
excluded.  A lone 'R' is far more likely to be a fragment of Role/Recipient
than a genuine value constant; including it would suppress detection of the
most common mid-token truncation pattern.

-- Hongxu Zhou, 14/May/2026
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# ── SBNSpec import (must be co-located with the PMB evaluation scripts) ────────
try:
    from sbn_spec import SBNSpec
except ImportError:
    sys.exit(
        "ImportError: sbn_spec not found.\n"
        "Place this file in the same directory as the PMB evaluation scripts."
    )

# Pre-compile the integer pattern once at module load.
_INTEGER_RE = re.compile(r"^\d+$")


def is_obvious_truncation(sbn_str: str) -> bool:
    """
    Return True when the last token of *sbn_str* is not a recognisable,
    complete SBN token — indicating that the sequence was cut mid-token.

    Parameters
    ----------
    sbn_str : str
        A single-line SBN string as produced by the byT5 parser.

    Returns
    -------
    bool
        True  → obvious (mid-token) truncation detected.
        False → last token is a complete, valid SBN token; the sequence may
                still be truncated at a token boundary (silent truncation),
                but that is not detectable by this heuristic.
    """
    tokens = sbn_str.strip().split()
    if not tokens:
        # Empty sequence — treat as not obviously truncated; the caller's
        # parse_error status already captures this case.
        return False

    last = tokens[-1]

    is_valid_token = (
        bool(SBNSpec.SYNSET_PATTERN.match(last))       # person.n.01
        or last in SBNSpec.ROLES                        # Agent, Theme, Time …
        or last in SBNSpec.DRS_OPERATORS                # EQU, TPR, Name …
        or last in SBNSpec.NEW_BOX_INDICATORS           # CONTINUATION, CONTRAST …
        or bool(SBNSpec.INDEX_PATTERN.match(last))      # +1, -2, <0, >1
        or bool(re.match(r'^".*"$', last))              # "England"  (both quotes)
        or bool(_INTEGER_RE.match(last))                # 1993, 30, 4
        or last in SBNSpec.CONSTANTS                    # now, speaker, hearer …
    )

    return not is_valid_token
