"""d*-separation in the bipartite blowup of a Hypergraph SCM.

Implements Theorem T1 (THEOREM_T1.md): standard Pearl d-separation on the
bipartite blowup, augmented with the deterministic-relations rule when
mechanisms have structurally-equal outputs (Geiger & Pearl 1990).
"""
from __future__ import annotations

from typing import Iterable

from .scm import HypergraphSCM


def deterministic_closure(scm: HypergraphSCM, Z: Iterable[str]) -> set[str]:
    """Smallest superset of Z closed under structural-equation determination.

    For v1 we only model the equality case: if a mechanism declares an output
    equality group (v_1, ..., v_k), then any v_i in the closure forces all v_j
    into the closure. Richer determination (e.g., invertible joint mechanisms)
    is a v2 extension.
    """
    closure = set(Z)
    changed = True
    while changed:
        changed = False
        for m in scm.mechanisms:
            for group in m.output_equalities:
                if any(v in closure for v in group):
                    new = set(group) - closure
                    if new:
                        closure |= new
                        changed = True
    return closure


def _build_adjacency(scm: HypergraphSCM) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    nodes, edges = scm.bipartite_blowup()
    parents: dict[str, set[str]] = {n: set() for n in nodes}
    children: dict[str, set[str]] = {n: set() for n in nodes}
    for u, v in edges:
        parents[v].add(u)
        children[u].add(v)
    return parents, children


def _descendants(start: str, children: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    stack = list(children[start])
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(children[n])
    return seen


def _simple_paths(
    source: str,
    target: str,
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
    max_paths: int = 1024,
) -> list[list[str]]:
    """All simple (no repeated nodes) undirected paths from source to target.

    Capped at `max_paths` to avoid pathological blowups; the bipartite blowup
    of a small hypergraph is small, so this is generous.
    """
    results: list[list[str]] = []
    path: list[str] = [source]
    visited: set[str] = {source}

    def dfs(node: str) -> None:
        if len(results) >= max_paths:
            return
        if node == target:
            results.append(list(path))
            return
        for nb in parents[node] | children[node]:
            if nb in visited:
                continue
            visited.add(nb)
            path.append(nb)
            dfs(nb)
            path.pop()
            visited.discard(nb)

    dfs(source)
    return results


def _path_open(
    path: list[str],
    Z_eff: set[str],
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
) -> bool:
    """Standard Pearl d-connection rules: open iff every intermediate triple is open."""
    if len(path) < 3:
        # A direct edge is open iff its endpoints are not in Z_eff.
        # (Endpoints in Z_eff is the degenerate "X intersects Z" case; caller should disallow.)
        return True
    for i in range(1, len(path) - 1):
        a, b, c = path[i - 1], path[i], path[i + 1]
        is_collider = (a in parents[b]) and (c in parents[b])
        if is_collider:
            if b not in Z_eff and not (_descendants(b, children) & Z_eff):
                return False
        else:
            if b in Z_eff:
                return False
    return True


def d_separated(
    scm: HypergraphSCM,
    X: Iterable[str],
    Y: Iterable[str],
    Z: Iterable[str] = (),
) -> bool:
    """d*-separation in B(scm) with deterministic-relations augmentation.

    Returns True iff X is d*-separated from Y given Z.
    Requires X, Y, Z pairwise disjoint subsets of scm.variables.
    """
    X_set, Y_set, Z_set = set(X), set(Y), set(Z)
    if X_set & Y_set or X_set & Z_set or Y_set & Z_set:
        raise ValueError("X, Y, Z must be pairwise disjoint.")
    Z_eff = deterministic_closure(scm, Z_set)
    # If the determination closure overlaps X or Y, the query is degenerate
    # (a variable functionally determined by Z is constant given Z).
    if X_set & Z_eff or Y_set & Z_eff:
        return True

    parents, children = _build_adjacency(scm)

    for x in X_set:
        for y in Y_set:
            for path in _simple_paths(x, y, parents, children):
                if _path_open(path, Z_eff, parents, children):
                    return False
    return True
