#!/usr/bin/env python3
"""Render the divergent cases so the index target can be checked by eye."""
import json, sys
from collections import Counter

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "output"
cases = json.load(open(OUT / "divergent_cases.json"))

limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(cases)

# group by the separator configuration that produced the divergence
by_sep = Counter()
for c in cases:
    seps = tuple(b[2] for b in c["boxes"][1:])
    by_sep[seps] += 1
print("divergence by separator sequence:")
for k, v in by_sep.most_common():
    print(f"   {v:4d}  {' > '.join(k)}")
print()

for c in cases[:limit]:
    print("=" * 78)
    print(f"{c['doc']}   {c['sent']}")
    print(f"SBN: {c['sbn']}")
    print("boxes: " + ", ".join(
        f"B{b[0]}" + (f"({b[2]} -> parent B{b[1]})" if b[1] is not None else "(top)")
        for b in c["boxes"]))
    print("concepts:")
    for pos, tok, box in c["concepts"]:
        mark = ""
        if pos == c["src"]:
            mark = "   <== SOURCE (the concept bearing the index)"
        elif pos == c["lin"]:
            mark = "   <== LINEAR model says this"
        elif c["reg"] is not None and pos == c["reg"]:
            mark = "   <== REGISTER model says this"
        print(f"   [{pos:2d}] B{box}  {tok}{mark}")
    print(f"EDGE: {c['src_tok']} --{c['role']} {c['raw']}-->")
    print(f"   LINEAR  : [{c['lin']}] {c['lin_tok']}")
    print(f"   REGISTER: [{c['reg']}] {c['reg_tok']}")
    print()
