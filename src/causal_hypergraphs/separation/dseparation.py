"""d*-separation on the bipartite blowup.

`d_separated` is a conditional-independence oracle: a ``True`` verdict licenses an
independence claim downstream, so the only acceptable failure direction is refusing to
answer -- never a false ``True``.

Two design commitments follow from that.

Reachability, not path enumeration
    Separation is decided by the Bayes-Ball reachability algorithm (Shachter 1998; see
    also Koller & Friedman 2009, Alg. 3.1), which visits each (node, direction) state at
    most once and therefore runs in O(V + E). Enumerating simple paths is exponential --
    the bipartite blowup of a chain of k branching mechanisms has 2**k of them -- and any
    cap on that enumeration silently converts "I stopped looking" into "separated". T1
    claims a polynomial-time conditional-independence oracle; this is that oracle.

Determination is handled set-wise
    A variable functionally determined by the conditioning set is constant given it, so
    it may be added to the conditioning set and removed from the query sets. These are
    Steps 4 and 5 of THEOREM_T1.md 3, and both are exact rather than conservative:
    conditioning on a deterministic function of Z leaves every conditional law unchanged.
    What is *not* sound is concluding separation because *some* element of X is
    determined -- the undetermined remainder of X can still be d-connected to Y.

    Both steps rest on the declared rules being **valid** (every declared equality really
    holds in the model). That is a hypothesis on the model description, not something
    this module can check, since it never sees the structural functions. It is also the
    only place validity is used: how *complete* the rule set is affects what can be
    proved, never whether what is proved is true.
"""
from __future__ import annotations

from collections.abc import Iterable

from causal_hypergraphs.graph import MechanismGraph

from .closure import deterministic_closure

# Direction of arrival at a node, in the Bayes-Ball sense.
_FROM_CHILD = True  # arrived moving "up" against an arrow
_FROM_PARENT = False  # arrived moving "down" along an arrow


def _build_adjacency(graph: MechanismGraph) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    parents: dict[str, set[str]] = {node: set() for node in graph.bipartite_nodes()}
    children: dict[str, set[str]] = {node: set() for node in graph.bipartite_nodes()}
    for parent, child in graph.bipartite_edges():
        parents[child].add(parent)
        children[parent].add(child)
    return parents, children


def _with_ancestors(nodes: Iterable[str], parents: dict[str, set[str]]) -> frozenset[str]:
    """``nodes`` together with all their ancestors."""
    stack = list(nodes)
    seen: set[str] = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(parents[node])
    return frozenset(seen)


def _reachable(
    sources: Iterable[str],
    conditioned: frozenset[str],
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
) -> frozenset[str]:
    """Nodes connected to `sources` by a trail left active by `conditioned`."""
    activated_colliders = _with_ancestors(conditioned, parents)
    frontier: list[tuple[str, bool]] = [(source, _FROM_CHILD) for source in sources]
    visited: set[tuple[str, bool]] = set()
    reached: set[str] = set()

    while frontier:
        state = frontier.pop()
        if state in visited:
            continue
        visited.add(state)
        node, direction = state
        blocked = node in conditioned
        if not blocked:
            reached.add(node)

        if direction is _FROM_CHILD:
            # Non-collider at `node`: open unless `node` is conditioned on.
            if not blocked:
                frontier.extend((parent, _FROM_CHILD) for parent in parents[node])
                frontier.extend((child, _FROM_PARENT) for child in children[node])
        else:
            # Chain through `node`: open unless `node` is conditioned on.
            if not blocked:
                frontier.extend((child, _FROM_PARENT) for child in children[node])
            # Collider at `node`: open iff `node` or one of its descendants is conditioned on.
            if node in activated_colliders:
                frontier.extend((parent, _FROM_CHILD) for parent in parents[node])

    return frozenset(reached)


def _as_set(value: object) -> frozenset[str]:
    if isinstance(value, str):
        return frozenset({value})
    return frozenset(str(item) for item in value)  # type: ignore[union-attr]


def d_separated(
    graph: MechanismGraph,
    x: object,
    y: object,
    given: object = (),
) -> bool:
    """Return True iff X is d*-separated from Y given `given` in the bipartite blowup.

    `given` is closed under the graph's declared determination rules before use, so a
    variable fixed by the conditioning set blocks paths through it even when the caller
    did not name it.
    """
    if not graph.is_mechanism_acyclic():
        raise ValueError(
            "d*-separation has no soundness guarantee on a cyclic mechanism graph. "
            f"Mechanisms on a cycle: {sorted(graph.cyclic_mechanisms)}. THEOREM_T1.md "
            "Lemma 2.1 establishes that the noise-augmented blowup is a DAG, and its "
            "proof ends 'C1 forbids cycles in G_E'; Step 1 of the soundness proof then "
            "relies on ancestral sampling over that DAG for the Markov property. Without "
            "C1 the lemma is false and a returned verdict would have nothing behind it -- "
            "in either direction, since callers read a separation oracle both ways. "
            "sigma-separation (Forre & Mooij) is the cyclic replacement and is not "
            "implemented."
        )
    x_set = _as_set(x)
    y_set = _as_set(y)
    z_set = _as_set(given)
    if x_set & y_set or x_set & z_set or y_set & z_set:
        raise ValueError("X, Y, and conditioning set must be pairwise disjoint.")

    conditioned = deterministic_closure(graph, z_set)

    # Determined query variables are constant given the conditioning set, so they carry no
    # information and are dropped. Only if *all* of X (or all of Y) is determined does the
    # independence follow immediately.
    x_effective = x_set - conditioned
    y_effective = y_set - conditioned
    if not x_effective or not y_effective:
        return True

    parents, children = _build_adjacency(graph)
    return not (_reachable(x_effective, conditioned, parents, children) & y_effective)
