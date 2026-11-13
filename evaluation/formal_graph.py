"""Resolve SBN once and expose complete triples and scope-aware local edges."""

from collections import Counter

import networkx as nx
from smatchpp import Smatchpp, solvers
from smatchpp.formalism.generic.tools import GenericStandardizer

from sbn_env import ensure_on_path

ensure_on_path()
from sbn_smatch import SBNGraph
from sbn_spec import SBNSpec, SBNError, split_synset_id, SBN_NODE_TYPE as Node, SBN_EDGE_TYPE as Edge


class InvalidGraph(ValueError):
    """The input cannot form a valid SBN graph."""


class CompleteReader(SBNGraph):
    """Keep parallel conditions that the PMB reader's DiGraph overwrites."""

    def add_edges_from(self, edges, **attrs):
        self.resolved_edges = tuple(edges)
        super().add_edges_from(self.resolved_edges, **attrs)


class Graph:
    def __init__(self, sbn):
        if not isinstance(sbn, str) or not sbn.strip():
            raise InvalidGraph("missing SBN")
        try:
            raw = CompleteReader().from_string(sbn.strip(), is_single_line=True)
        except (SBNError, IndexError) as exc:
            raise InvalidGraph(str(exc)) from exc
        self.root = (Node.BOX, 0)
        if raw.is_possibly_ill_formed or any(not d for _, d in raw.nodes(data=True)):
            raise InvalidGraph("unresolved relative index")
        if not raw.is_dag:
            raise InvalidGraph("cyclic graph")
        if len(nx.descendants(raw, self.root)) + 1 != len(raw):
            raise InvalidGraph("graph has material unreachable from the root box")
        self.labels = {n: d["token"] for n, d in raw.nodes(data=True)}
        self.concepts = tuple(n for n in raw if n[0] == Node.SYNSET)
        if any(not split_synset_id(self.labels[n]) for n in self.concepts):
            raise InvalidGraph("invalid synset label")
        if not self.concepts:
            raise InvalidGraph("graph has no concepts")
        self.box = {}
        self.parents = {}
        self.box_links = []
        self.relations = []
        for a, b, d in raw.resolved_edges:
            kind, label = d["type"], d["token"]
            if kind == Edge.BOX_CONNECT:
                self.box[b] = a
            elif kind == Edge.BOX_BOX_CONNECT:
                self.parents[b] = (a, label)
                self.box_links.append((a, label, b))
            else:
                # Preserve the written owner as well as semantic direction.
                owner = self.box[a]
                if label in SBNSpec.INVERTIBLE_ROLES:
                    if b[0] != Node.SYNSET:
                        raise InvalidGraph("inverse role does not target a concept")
                    self.relations.append((b, label[:-2], a, owner))
                else:
                    self.relations.append((a, label, b, owner))

    def pairs(self):
        # Box order is part of the notation, never an alignment to gold.
        by_parent = {}
        for parent, label, box in sorted(self.box_links, key=lambda e: e[2][1]):
            if label == "CORRECTION":
                by_parent[parent] = box
            elif label == "CONJUNCTION" and parent in by_parent:
                yield parent, by_parent.pop(parent), box

    def triples(self):
        """Same graph vocabulary as PMB PENMAN, without a lossy serialisation."""
        def var(n):
            return f"{n[0].value[0]}{n[1]}"

        triples = [("ROOT_OF_GRAPH", ":root", var(self.root))]
        for n, label in self.labels.items():
            if n[0] == Node.BOX:
                triples.append((var(n), ":instance", '"box"'))
            elif n[0] == Node.SYNSET:
                triples.append((var(n), ":instance", '"' + label + '"'))
        triples.extend((var(box), ":member", var(n)) for n, box in self.box.items())
        triples.extend((var(a), f":{r}", var(b)) for a, r, b in self.box_links)
        for a, label, b, _ in self.relations:
            if b[0] == Node.CONSTANT:
                value = self.labels[b]
                triples.append((var(a), f":{label}", '"' + value.strip('"') + '"'))
            else:
                triples.append((var(a), f":{label}", var(b)))
        return triples


_STRUCTURAL = {":instance", ":member", ":root"}


def make_scorer():
    return Smatchpp(alignmentsolver=solvers.ILP(),
                    graph_standardizer=GenericStandardizer())


def full_score(scorer, gold, pred, relation_recall=False):
    """Use one full-graph SMATCH++ alignment, also for omission diagnostics."""
    standardize = scorer.graph_standardizer.standardize
    a, b, va, vb = scorer.graph_pair_preparer.prepare_get_vars(
        standardize(gold.triples()), standardize(pred.triples()))
    alignment, indices, (lower, upper) = scorer.graph_aligner.align(a, b, va, vb)
    if abs(min(len(a), len(b), upper) - lower) > 1e-6:
        raise RuntimeError("SMATCH++ did not certify an optimal alignment")
    match, _, na, nb = scorer.graph_scorer.score(a, b, alignment, indices)
    result = {"precision": float(match / nb), "recall": float(match / na),
              "f1": float(2 * match / (na + nb))}
    if relation_recall:
        # SMATCH++ namespaces second-graph variables as bb_.
        target = {i: v for v, i in indices.items() if v in vb}
        mapping = {v: target.get(alignment[i], v) for v, i in indices.items() if v in va}
        ar = Counter((mapping.get(x, x), r, mapping.get(y, y))
                     for x, r, y in a if r not in _STRUCTURAL)
        br = Counter(t for t in b if t[1] not in _STRUCTURAL)
        total = ar.total()
        hit = (ar & br).total()
        result.update(relation_matched=hit, relation_total=total,
                      relation_recall=hit / total if total else None)
    return result
