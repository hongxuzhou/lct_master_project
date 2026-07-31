"""
Round-trip every generated repair through the project's own SBN parser
(data/pmb-5.1.0/src/sbn/sbn_smatch.py), which is what the evaluation pipeline
uses.  A splice that our arithmetic calls "legal" is worthless if SBNGraph
refuses it, silently drops an edge, or produces a non-DAG.

Checks per sample:
  parse      -- SBNGraph.from_string does not raise
  dag        -- graph is acyclic
  edges      -- number of role/operator edges matches what we wrote
                (catches indices that fall off the end and get dropped)
  boxes      -- CORRECTION and CONJUNCTION end up as *siblings* off the same
                parent box, which is what the notation requires

Run:  python3 validate_with_pmb_parser.py [n_samples]
"""
from __future__ import annotations

import collections
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data/pmb-5.1.0/src/sbn"))

import networkx as nx  # noqa: E402

from sbn_smatch import SBNGraph, SBNSource  # noqa: E402
from sbn_spec import SBN_EDGE_TYPE, SBN_NODE_TYPE, SBNError  # noqa: E402

from sbn_lin import IDX_PATTERN, read_split  # noqa: E402
from repair_transform import build_repair  # noqa: E402
from wn_candidates import candidates  # noqa: E402
from analyse_feasibility import NON_LEXICAL, POS_OK, WORD_RE, surface_forms, token_match  # noqa: E402

GOLD = ROOT / "data/pmb-5.1.0/split/en/train/gold.sbn"


def expected_edge_count(line: str) -> int:
    """Role/operator edges we intended to write (index-valued ones only)."""
    from sbn_lin import BOX_PTR_PATTERN, parse_sbn_line
    d = parse_sbn_line(line)
    n = 0
    for c in d.concepts:
        for _, val in c.roles:
            if BOX_PTR_PATTERN.match(val):
                continue                      # syn-box edge, counted separately
            n += 1                            # index or constant: both an edge
    return n


def box_shape(G) -> tuple[str, ...]:
    """Sequence of box-box connector labels, with parent box ids."""
    out = []
    for u, v, data in G.edges(data=True):
        if data.get("type") == SBN_EDGE_TYPE.BOX_BOX_CONNECT:
            out.append((data["token"], u[1], v[1]))
    return tuple(sorted(out, key=lambda x: x[2]))


def main() -> None:
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    random.seed(0)
    docs = read_split(GOLD)

    results = collections.Counter()
    failures = collections.defaultdict(list)
    sample_lines = []

    for doc in docs:
        lowered = [t.lower() for t in WORD_RE.findall(doc.sentence)]
        for c in doc.concepts:
            if c.pos_tag not in POS_OK or c.synset in NON_LEXICAL:
                continue
            if token_match(surface_forms(c.synset), lowered) is None:
                continue
            cands = candidates(c.synset)
            if not cands:
                continue
            res = build_repair(doc, c.pos, cands[0])
            if res.ok:
                sample_lines.append((doc, c.pos, cands[0], res))
    random.shuffle(sample_lines)
    sample_lines = sample_lines[:n_samples]
    print(f"validating {len(sample_lines)} generated repairs "
          f"against sbn_smatch.SBNGraph\n")

    for doc, pos, cand, res in sample_lines:
        try:
            G = SBNGraph(source=SBNSource.INFERENCE).from_string(res.sbn, is_single_line=True)
        except SBNError as e:
            results["parse_error"] += 1
            failures["parse_error"].append((doc.doc_id, str(e), res.sbn))
            continue
        except Exception as e:  # noqa: BLE001
            results["crash"] += 1
            failures["crash"].append((doc.doc_id, f"{type(e).__name__}: {e}", res.sbn))
            continue
        results["parsed"] += 1

        if not nx.is_directed_acyclic_graph(G):
            results["not_dag"] += 1
            failures["not_dag"].append((doc.doc_id, "cycle", res.sbn))

        got = sum(1 for _, _, d in G.edges(data=True)
                  if d.get("type") in (SBN_EDGE_TYPE.ROLE,
                                       SBN_EDGE_TYPE.DRS_OPERATOR))
        want = expected_edge_count(res.sbn)
        if got != want:
            results["edge_loss"] += 1
            failures["edge_loss"].append(
                (doc.doc_id, f"wrote {want} index edges, graph has {got}", res.sbn))
        else:
            results["edges_ok"] += 1

        shape = box_shape(G)
        corr = next(((par, new) for tok, par, new in shape
                     if tok == "CORRECTION"), None)
        # our CONJUNCTION is the box created immediately after the CORRECTION
        conj = next(((par, new) for tok, par, new in shape
                     if tok == "CONJUNCTION" and corr and new == corr[1] + 1),
                    None)
        if corr is None or conj is None:
            results["missing_box_edge"] += 1
            failures["missing_box_edge"].append(
                (doc.doc_id, f"CORRECTION={corr} CONJUNCTION={conj}", res.sbn))
        elif corr[0] != conj[0]:
            results["not_siblings"] += 1
            failures["not_siblings"].append(
                (doc.doc_id,
                 f"CORRECTION parent box {corr[0]}, CONJUNCTION parent {conj[0]}",
                 res.sbn))
        else:
            results["boxes_ok"] += 1

    print("=== results ===")
    for k, v in sorted(results.items()):
        print(f"  {k:20s} {v:6d}")
    print()
    for k, v in failures.items():
        print(f"=== {k} (showing up to 5 of {len(v)}) ===")
        for doc_id, msg, line in v[:5]:
            print(f"  {doc_id}: {msg}")
            print(f"    {line}")
        print()


if __name__ == "__main__":
    main()
