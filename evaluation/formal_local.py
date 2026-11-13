"""Local checks over one consistent, position-independent correspondence."""

from dataclasses import dataclass
from itertools import product

from formal_graph import Node


@dataclass(frozen=True)
class Result:
    status: str
    reason: str = ""


PASS = Result("pass")


@dataclass(frozen=True)
class Pattern:
    # Keys are local names (or gold node IDs); their values are synsets.
    nodes: dict
    # (key, scope alias), with aliases root/parent/withdrawn/accepted or p0...
    placement: tuple
    # Root-to-parent path of (separator, optional embedding role, governor synset).
    scope: tuple
    # (source, role, target, owner scope); string constants use ("value", value).
    edges: tuple
    # Optional conditions that must not be withdrawn. An endpoint can be a
    # required node key, a constant, or ("optional_synset", label) for an optional node.
    protected_edges: tuple = ()


def value(token):
    return ("value", token.strip('"'))


def scope_path(g, box):
    steps = []
    visited = set()
    while box != g.root:
        if box in visited:
            raise ValueError("cyclic scope")
        visited.add(box)
        if box in g.parents:
            parent, label = g.parents[box]
            steps.append(((label, None, None), box))
            box = parent
        else:
            links = [(a, r, owner) for a, r, b, owner in g.relations if b == box]
            if len(links) != 1:
                raise ValueError("scope has no unique embedding relation")
            a, r, box_owner = links[0]
            steps.append(((None, r, g.labels[a]), box))
            box = box_owner
    return tuple(reversed(steps))


def _withdraws_protected_edge(g, edges, pattern, binding, withdrawn):
    # Nested contexts inside CORRECTION are withdrawn too.
    inside = {withdrawn}
    pending = [withdrawn]
    children = {}
    for parent, _, child in g.box_links:
        children.setdefault(parent, set()).add(child)
    for source, _, target, owner in g.relations:
        if target[0] == Node.BOX:
            children.setdefault(owner, set()).add(target)
    while pending:
        nested = children.get(pending.pop(), set()) - inside
        inside.update(nested)
        pending.extend(nested)

    def matches(expected, actual):
        if isinstance(expected, tuple) and expected[0] == "optional_synset":
            return actual in g.box and g.labels[actual] == expected[1]
        return binding.get(expected, expected) == actual

    return any(
        role == required_role and matches(src, a) and matches(dst, b)
        and (owner in inside or g.box.get(a) in inside or g.box.get(b) in inside)
        for src, required_role, dst in pattern.protected_edges
        for a, role, b, owner in edges
    )


def check_repair(g, pattern):
    if pattern is None:
        return Result("review", "local repair requirements unavailable"), []
    pairs = list(g.pairs())
    if not pairs:
        return Result("fail", "no paired CORRECTION and CONJUNCTION"), []
    edges = {(a, r, value(g.labels[b]) if b[0] == Node.CONSTANT else b, owner)
             for a, r, b, owner in g.relations}
    stage = 0
    hits = []
    placements = dict(pattern.placement)
    tested = 0
    for parent, withdrawn, accepted in pairs:
        try:
            path = scope_path(g, parent)
        except ValueError:
            continue
        if tuple(step for step, _ in path) != pattern.scope:
            continue
        stage = max(stage, 1)
        scopes = dict(root=g.root, parent=parent, withdrawn=withdrawn, accepted=accepted)
        scopes.update({f"p{i}": b for i, (_, b) in enumerate(path)})
        # CONJUNCTION merges its conditions into the parent. The withdrawn
        # box does not merge, even if an inverse role points back to the parent.
        owner_scopes = {alias: {box} for alias, box in scopes.items()}
        for alias, box in scopes.items():
            if box in (parent, accepted):
                owner_scopes[alias] = {parent, accepted}
        keys = tuple(pattern.nodes)
        candidates = [[n for n in g.concepts if g.labels[n] == pattern.nodes[k]]
                      for k in keys]
        if not all(candidates):
            continue
        stage = max(stage, 2)
        domains = [[n for n in ns if k not in placements or g.box[n] == scopes[placements[k]]]
                   for k, ns in zip(keys, candidates)]
        if not all(domains):
            continue
        stage = max(stage, 3)
        for ns in product(*domains):
            tested += 1
            if tested > 100_000:
                return Result("review", "too many candidate repair correspondences"), []
            if len(set(ns)) != len(ns):
                continue
            stage = max(stage, 4)
            binding = dict(zip(keys, ns))
            if all(any((binding[a], r, binding.get(b, b), scope) in edges
                       for scope in owner_scopes[owner])
                   for a, r, b, owner in pattern.edges):
                withdrawn_edges = {(binding[a], r, binding.get(b, b))
                                   for a, r, b, owner in pattern.edges
                                   if owner == "withdrawn" and (placements.get(a) == "withdrawn"
                                                                or placements.get(b) == "withdrawn")}
                if any((a, r, b) in withdrawn_edges and o != withdrawn
                       for a, r, b, o in edges):
                    stage = max(stage, 5)
                    continue
                if pattern.protected_edges and _withdraws_protected_edge(
                    g, edges, pattern, binding, withdrawn
                ):
                    stage = max(stage, 6)
                    continue
                hits.append(binding)
                if len(hits) > 1:
                    return Result("review", "multiple matching repair correspondences"), hits
    if hits:
        return PASS, hits
    return Result("fail", (
        "repair pair has a different enclosing scope",
        "required repair object absent",
        "required object outside its expected status or scope",
        "required objects cannot be distinguished",
        "required relation or value absent in its expected scope",
        "withdrawn relation also occurs outside the withdrawn box",
        "protected relation is inside the withdrawn scope",
    )[stage]), []


@dataclass(frozen=True)
class Reference:
    group_label: str
    # gold member key -> (synset, distinguishing constant conditions)
    members: dict
    # A semantic continuation anchor, used to identify which group is later.
    anchor: tuple  # (group is source, role, predicate synset)
    modifiers: dict | None = None  # member key -> (direction, role, synset) conditions


def inspect_reference(g, spec):
    """Return a verdict, exact correspondence, and partial member evidence."""
    evidence = dict(member_recovered=[None, None], extra_members=None, sub_targets=None)
    if spec is None:
        return Result("review", "group or antecedent gold requirements unavailable"), [], evidence
    forward, role, predicate = spec.anchor
    groups = set()
    for a, r, b, _ in g.relations:
        group, other = (a, b) if forward else (b, a)
        if (r == role and group in g.box and other in g.box
                and g.labels[group] == spec.group_label and g.labels[other] == predicate):
            groups.add(group)
    if not groups:
        if any(r == "Sub" and g.labels[a] == spec.group_label for a, r, _, _ in g.relations):
            return Result("review", "a group is present but the later predicate does not identify it"), [], evidence
        evidence["member_recovered"] = [False] * len(spec.members)
        return Result("fail", "no later group with the expected concept and links"), [], evidence
    if len(groups) != 1:
        return Result("review", "multiple groups linked to the later predicate"), [], evidence
    group = groups.pop()
    actual = {b for a, r, b, _ in g.relations if a == group and r == "Sub"}
    evidence["sub_targets"] = [
        dict(index=n[1] if n in g.box else None, label=g.labels[n],
             box=g.box[n][1] if n in g.box else None,
             box_relation=g.parents.get(g.box.get(n), (None, None))[1])
        for n in sorted(actual)
    ]
    attrs = {(a, r, value(g.labels[b])) for a, r, b, _ in g.relations
             if b[0] == Node.CONSTANT}
    modifiers = set()
    for a, r, b, _ in g.relations:
        if a in g.box and b in g.box:
            modifiers.update(((a, True, r, g.labels[b]), (b, False, r, g.labels[a])))
    domains = [[n for n in g.concepts if n != group and g.labels[n] == label
                and all((n, r, v) in attrs for r, v in conditions)
                and all((n, *m) in modifiers for m in (spec.modifiers or {}).get(key, ()))]
               for key, (label, conditions) in spec.members.items()]
    # Establish identity independently of which Sub edges were predicted.
    # Keep one-to-one identity where it is unique, even if Sub is incomplete.
    # If a gold member has no candidate, still check the other member.
    keys = tuple(spec.members)
    mappings = []
    for ns in product(*(domain or [None] for domain in domains)):
        assigned = [n for n in ns if n is not None]
        if len(set(assigned)) == len(assigned):
            mappings.append(dict(zip(keys, ns)))
            if len(mappings) > 1:
                return Result("review", "member identities have multiple correspondences"), [], evidence
    if not mappings:
        return Result("review", "member identities cannot be distinguished"), [], evidence
    binding = mappings[0]
    evidence["member_recovered"] = [binding[k] in actual for k in keys]
    evidence["extra_members"] = len(actual - set(binding.values()))
    if not all(evidence["member_recovered"]) or evidence["extra_members"]:
        return Result("fail", "later group does not link to exactly the intended members"), [], evidence
    return PASS, mappings, evidence


def check_reference(g, spec):
    result, bindings, _ = inspect_reference(g, spec)
    return result, bindings


def joint(repair, reference):
    rr, repairs = repair
    mr, members = reference
    if any(x.status == "review" for x in (rr, mr)):
        return Result("review", "repair or member correspondence needs review")
    if any(x.status != "pass" for x in (rr, mr)):
        return Result("fail", "repair and exact group membership are not both present")
    shared = repairs[0].keys() & members[0].keys()
    if shared != members[0].keys():
        return Result("review", "repair requirements do not identify both gold members")
    if any(repairs[0][key] != members[0][key] for key in shared):
        return Result("fail", "repair and group link to different referents")
    return PASS
