"""Conservative gold extraction for released RepairSBN slices."""

from numbers import Integral

from formal_graph import Node
from formal_local import Pattern, Reference, check_reference, check_repair, scope_path, value


class NeedsReview(ValueError):
    """The released row does not determine a supported local pattern."""


class GoldError(ValueError):
    """Gold metadata disagrees with the graph or its published contract."""


def _require(condition, message):
    if not condition:
        raise GoldError(message)


def _pattern(g, pair, selected, relations):
    parent, w, a = pair
    try:
        path = scope_path(g, parent)
    except ValueError as exc:
        raise NeedsReview(str(exc)) from exc
    boxes = {b: f"p{i}" for i, (_, b) in enumerate(path)}
    boxes.update({g.root: "root", parent: "parent", w: "withdrawn", a: "accepted"})
    nodes = set(selected)
    for x, _, y, _ in relations:
        nodes.add(x)
        if y[0] == Node.SYNSET:
            nodes.add(y)
        elif y[0] != Node.CONSTANT:
            raise NeedsReview("local relation targets a box")
    if any(g.box[n] not in boxes for n in nodes) or any(o not in boxes for _, _, _, o in relations):
        raise NeedsReview("local material reaches outside supported scope")
    # Names/values of shared anchors can distinguish repeated synsets.
    relations = set(relations)
    relations.update((x, r, y, o) for x, r, y, o in g.relations
                     if x in nodes and y[0] == Node.CONSTANT)
    return Pattern({n: g.labels[n] for n in sorted(nodes)},
                   tuple((n, boxes[g.box[n]]) for n in sorted(nodes)),
                   tuple(step for step, _ in path),
                   tuple((x, r, value(g.labels[y]) if y[0] == Node.CONSTANT else y, boxes[o])
                         for x, r, y, o in sorted(relations)))


def core_pattern(g, row):
    """Only simple lexical/value pairs with one concept in each repair box."""
    pairs = list(g.pairs())
    if len(pairs) != 1:
        raise NeedsReview("expected one unambiguous repair pair")
    if row["repair_type"] not in {"content_word", "constant_name", "constant_number", "constant_temporal"}:
        raise NeedsReview("repair requires explicit local requirements for its construction")
    pair = pairs[0]
    _, w, a = pair
    left, right = ([n for n in g.concepts if g.box[n] == b] for b in (w, a))
    if len(left) != 1 or len(right) != 1:
        raise NeedsReview("gold boxes do not isolate both contributions from continuation")
    x, y = left[0], right[0]
    if (g.labels[x], g.labels[y]) != (row["reparandum_synset"], row["repair_synset"]):
        raise NeedsReview("gold concepts do not match lexical metadata")
    relations = [e for e in g.relations if x in (e[0], e[2]) or y in (e[0], e[2])]
    if row["repair_type"] != "content_word":
        def constants(n):
            return {(r, value(g.labels[b])) for s, r, b, _ in relations
                    if s == n and b[0] == Node.CONSTANT}
        lv, rv = constants(x), constants(y)
        expected = row.get("reparandum_value")
        if not isinstance(expected, str) or not any(v == value(expected) for _, v in lv - rv) or not rv - lv:
            raise NeedsReview("value contrast cannot be isolated from gold and metadata")
    return _pattern(g, pair, (x, y), relations)


def _gold_node(g, i):
    _require(isinstance(i, Integral) and 0 <= i < len(g.concepts), "invalid gold concept index")
    return g.concepts[i]


def later_spec(g, row):
    try:
        group = _gold_node(g, row["reference_group_index"])
        members = [_gold_node(g, i) for i in row["antecedent_indices"]]
        withdrawn = [_gold_node(g, i) for i in row["withdrawn_antecedent_indices"]]
    except TypeError as exc:
        raise GoldError("invalid gold reference metadata") from exc
    _require(len(members) == 2 and len(set(members)) == 2 and group not in members,
             "expected two distinct gold antecedents")
    _require(set(withdrawn) <= set(members) and len(withdrawn) == (row["language_condition"] == "rsr"),
             "inconsistent gold withdrawal metadata")
    _require({b for a, r, b, _ in g.relations if a == group and r == "Sub"} == set(members),
             "gold Sub edges disagree with metadata")
    anchors = {(a == group, r, g.labels[b if a == group else a])
               for a, r, b, _ in g.relations if group in (a, b)
               and r in {"Attribute", "Theme"} and a in g.box and b in g.box
               and g.box[a] == g.box[b]}
    if len(anchors) != 1:
        raise NeedsReview("later predicate anchor is not unique in gold")
    descriptions = {}
    modifiers = {}
    for n in members:
        descriptions[n] = (g.labels[n], tuple(sorted(
            (r, value(g.labels[b])) for a, r, b, _ in g.relations
            if a == n and r in {"Name", "EQU"} and b[0] == Node.CONSTANT)))
        # Descriptive modifiers are independent of repair status and Sub.
        modifiers[n] = tuple(sorted(
            (a == n, r, g.labels[b if a == n else a]) for a, r, b, _ in g.relations
            if n in (a, b) and a in g.box and b in g.box and r in {"Colour", "Attribute"}
            and g.labels[b if a == n else a].rsplit(".", 2)[1] == "a"))
    spec = Reference(g.labels[group], descriptions, anchors.pop(), modifiers)
    result, _ = check_reference(g, spec)
    if result.status != "pass":
        raise NeedsReview("gold reference identity: " + result.reason)
    return spec, withdrawn


def later_pattern(g, spec, withdrawn):
    w = withdrawn[0]
    a = next(n for n in spec.members if n != w)
    pairs = [pair for pair in g.pairs() if pair[1:] == (g.box[w], g.box[a])]
    _require(len(pairs) == 1, "gold withdrawal metadata disagrees with C–C scopes")
    pair = pairs[0]
    local = {n for n in g.concepts if g.box[n] in pair[1:]}
    # Later reference is not itself a repair requirement.
    relations = [e for e in g.relations
                 if (e[0] in local or e[2] in local) and e[1] != "Sub"]
    # Only two validated templates: shared event, or dummy subject + event.
    event_roles = {}
    for x, r, y, _ in relations:
        if y in {w, a} and x not in local and g.labels[x].rsplit(".", 2)[1] in {"v", "a"}:
            event_roles.setdefault((x, r), set()).add(y)
    shared_event = any(ys == {w, a} for ys in event_roles.values())
    shared_anchor = {y for x, r, y, _ in relations if x == w and r == "EQU" and y in g.box} & {
        y for x, r, y, _ in relations if x == a and r == "EQU" and y in g.box}
    anchored_event = any(y in shared_anchor and x in local and g.labels[x].rsplit(".", 2)[1] == "v"
                         for x, _, y, _ in relations)
    if not (shared_event or anchored_event):
        raise NeedsReview("later repair does not match a supported argument-replacement template")
    return _pattern(g, pair, local, relations)


def validate_pattern(g, pattern):
    result, _ = check_repair(g, pattern)
    if result.status != "pass":
        raise NeedsReview("gold repair correspondence: " + result.reason)
    return pattern


def annotated_pattern(g, annotation):
    """Read reviewed local requirements expressed against this gold graph."""
    pairs = list(g.pairs())
    if len(pairs) != 1:
        raise GoldError("reviewed pattern requires one gold repair pair")
    try:
        nodes = [_gold_node(g, i) for i in annotation["concept_indices"]]
        edges = []
        for src, role, target in annotation["relations"]:
            a = _gold_node(g, src)
            if isinstance(target, str):
                matches = [e for e in g.relations if e[:2] == (a, role)
                           and e[2][0] == Node.CONSTANT and value(g.labels[e[2]]) == value(target)]
            else:
                b = _gold_node(g, target)
                matches = [e for e in g.relations if e[:3] == (a, role, b)]
            _require(len(matches) == 1, "reviewed relation is absent or ambiguous in gold")
            edges.extend(matches)
        _require(nodes and edges, "reviewed pattern needs concepts and relations")
        pair = pairs[0]
        covered = set(nodes) | {n for x, _, y, _ in edges for n in (x, y) if n in g.box}
        _require(set(pair[1:]) <= {g.box[n] for n in covered},
                 "reviewed pattern must identify both repair contributions")
        return validate_pattern(g, _pattern(g, pair, nodes, edges))
    except (KeyError, TypeError, ValueError) as exc:
        raise GoldError(str(exc)) from exc
