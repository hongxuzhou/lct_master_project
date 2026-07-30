#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Locate the canonical PMB SBN modules and put them on sys.path.
-- Hongxu Zhou, 2026

Every module here needs `sbn_smatch` / `sbn_spec` / `graph_base` /
`penman_model`. There is exactly ONE authoritative copy, in
`data/pmb-5.1.0/src/sbn/`, and this package deliberately does not vendor a
second one.

Why it matters: `sbn_spec.py` carries the project's own edits -- the
`CORRECTION` separator, the extra roles, and the `INVERTIBLE_ROLES` additions
that scoring depends on. A vendored duplicate drifts the moment a role is added
to one copy and not the other, and the symptom is a silent scoring difference,
not an error. `colloquium_prep/pilot_eval/sbn_lib/` is a frozen snapshot kept
only so the pilot's published numbers stay reproducible; never point new code
at it.

Resolution order:
    1. $SBN_SRC, if set -- for HPC runs where the repo layout differs
    2. ./vendor/sbn/ next to this file, if someone shipped a snapshot
    3. <repo root>/data/pmb-5.1.0/src/sbn/  (the normal case)
"""

import os
import sys
from pathlib import Path

__all__ = ["sbn_src_dir", "ensure_on_path"]

_REQUIRED = ("sbn_smatch.py", "sbn_spec.py", "graph_base.py", "penman_model.py")

_HERE = Path(__file__).resolve().parent
_CANDIDATES = (
    lambda: Path(os.environ["SBN_SRC"]) if os.environ.get("SBN_SRC") else None,
    lambda: _HERE / "vendor" / "sbn",
    lambda: _HERE.parent / "data" / "pmb-5.1.0" / "src" / "sbn",
)


def _is_complete(path: Path) -> bool:
    return path.is_dir() and all((path / f).exists() for f in _REQUIRED)


def sbn_src_dir() -> Path:
    """The directory holding the canonical SBN modules."""
    tried = []
    for candidate in _CANDIDATES:
        path = candidate()
        if path is None:
            continue
        path = path.resolve()
        tried.append(str(path))
        if _is_complete(path):
            return path
    raise ImportError(
        "Cannot locate the PMB SBN modules ("
        + ", ".join(_REQUIRED)
        + ").\nTried:\n  "
        + "\n  ".join(tried)
        + "\nSet $SBN_SRC to the directory containing them."
    )


def ensure_on_path() -> Path:
    """Put the canonical SBN directory on sys.path (idempotent)."""
    path = sbn_src_dir()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path
