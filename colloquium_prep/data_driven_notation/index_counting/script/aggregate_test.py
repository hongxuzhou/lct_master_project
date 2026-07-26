#!/usr/bin/env python3
"""Corpus-level discrimination between the LINEAR and REGISTER counting models.

Two independent aggregate tests over every negative index in gold:
  1. WELL-FORMEDNESS -- does the index resolve to an existing concept at all?
  2. TYPE AGREEMENT  -- does a `Time` role land on a time concept, a `Name`-
     bearing role on a nameable one, etc.?  A wrong counting domain shows up
     as systematic type violations.
"""
import re
from collections import Counter
from probe_index_domain import read_records, parse, load_roles

from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
gold = f"{ROOT}/data/pmb-5.1.0/split/en/train/gold.sbn"
roles = load_roles(f"{ROOT}/data/pmb-5.1.0/src/sbn/sbn_spec.py")

TIME_ROLES = {"Time"}
stats = Counter()

for block in read_records(gold):
    if len(block) < 3:
        continue
    sbn = " ".join(block[2:])
    try:
        concepts, edges, boxes, snap = parse(sbn, roles)
    except Exception:
        continue

    for e in edges:
        if e["kind"] != "concept_ptr" or e["n"] >= 0:
            continue
        n, src = e["n"], e["src"]
        stats["neg_total"] += 1

        lin = src + n
        lin_tok = concepts[lin][1] if 0 <= lin < len(concepts) else None

        L = snap.get(src, [])
        here = L.index(src) if src in L else len(L) - 1
        rp = here + n
        reg = L[rp] if 0 <= rp < len(L) else None
        reg_tok = concepts[reg][1] if reg is not None else None

        # test 1: resolvable?
        if lin_tok is None:
            stats["LINEAR_unresolvable"] += 1
        if reg_tok is None:
            stats["REGISTER_unresolvable"] += 1

        # test 2: does a Time role land on a time concept?
        if e["role"] in TIME_ROLES:
            stats["time_role_total"] += 1
            if lin_tok and lin_tok.startswith("time.n"):
                stats["LINEAR_time_ok"] += 1
            if reg_tok and reg_tok.startswith("time.n"):
                stats["REGISTER_time_ok"] += 1

print("=" * 70)
print("AGGREGATE MODEL COMPARISON  (all negative indices, en/train/gold)")
print("=" * 70)
t = stats["neg_total"]
print(f"  negative indices examined            {t:6d}")
print()
print("  TEST 1 -- index fails to resolve to any concept")
for m in ("LINEAR", "REGISTER"):
    u = stats[f"{m}_unresolvable"]
    print(f"    {m:9s} unresolvable  {u:6d}   ({100*u/t:5.2f}%)")
print()
print("  TEST 2 -- `Time -n` must land on a time.n.* concept")
tt = stats["time_role_total"]
for m in ("LINEAR", "REGISTER"):
    ok = stats[f"{m}_time_ok"]
    print(f"    {m:9s} correct type  {ok:6d} / {tt}   ({100*ok/tt:6.2f}%)")
