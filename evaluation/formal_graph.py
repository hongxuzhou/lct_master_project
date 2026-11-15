"""Resolve SBN once and expose complete triples and scope-aware local edges."""

from collections import Counter

import networkx as nx
from smatchpp import Smatchpp, solvers
from smatchpp.formalism.generic.tools import DoNothingStandardizer

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

    @staticmethod
    def var(n):
        return f"{n[0].value[0]}{n[1]}"

    def condition_scope(self, box):
        """Bos (2023), Appendix A: CONJUNCTION adds to its target DRS."""
        while box in self.parents and self.parents[box][1] == "CONJUNCTION":
            box = self.parents[box][0]
        return box

    def relation_facts(self, prefix=""):
        """One fact per original relation, retaining scope and literal case."""
        var = lambda n: prefix + self.var(n)
        facts = [(None, var(a), label, var(b)) for a, label, b in self.box_links]
        for a, label, b, owner in self.relations:
            target = ('"' + self.labels[b].strip('"') + '"'
                      if b[0] == Node.CONSTANT else var(b))
            facts.append((var(self.condition_scope(owner)), var(a), label, target))
        return facts

    def triples(self):
        """Encode scoped conditions; inversion never changes their owner.

        Hooks/anchors enter their host context's conditions (Bos 2023,
        Appendix A). A condition node retains this after role inversion.
        These are scoring nodes, not additions to the SBN notation.
        """
        var = self.var

        triples = [("ROOT_OF_GRAPH", ":root", var(self.root))]
        for n, label in self.labels.items():
            if n[0] == Node.BOX:
                triples.append((var(n), ":instance", '"box"'))
            elif n[0] == Node.SYNSET:
                triples.append((var(n), ":instance", '"' + label + '"'))
        triples.extend((var(box), ":member", var(n)) for n, box in self.box.items())
        triples.extend((var(a), f":{r}", var(b)) for a, r, b in self.box_links)
        for i, (a, label, b, owner) in enumerate(self.relations):
            condition = f"r{i}"
            triples.extend(((condition, ":instance", '"relation:' + label + '"'),
                            (condition, ":source", var(a)),
                            (condition, ":scope", var(self.condition_scope(owner)))))
            if b[0] == Node.CONSTANT:
                value = self.labels[b]
                triples.append((condition, ":target", '"' + value.strip('"') + '"'))
            else:
                triples.append((condition, ":target", var(b)))
        return triples


def make_scorer():
    return Smatchpp(alignmentsolver=solvers.ILP(),
                    graph_standardizer=DoNothingStandardizer())


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
        # Count original semantic/discourse relations, not the four encoding
        # triples for each condition. Reuse the full graph's node alignment.
        ar = Counter(tuple(mapping.get(v, v) for v in fact)
                     for fact in gold.relation_facts("aa_"))
        br = Counter(pred.relation_facts("bb_"))
        total = ar.total()
        hit = (ar & br).total()
        result.update(relation_matched=hit, relation_total=total,
                      relation_recall=hit / total if total else None)
    return result
