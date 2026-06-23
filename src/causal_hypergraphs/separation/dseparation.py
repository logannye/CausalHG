from __future__ import annotations

from causal_hypergraphs.graph import MechanismGraph

from .closure import deterministic_closure


def _build_adjacency(graph: MechanismGraph) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    parents = {node: set() for node in graph.bipartite_nodes()}
    children = {node: set() for node in graph.bipartite_nodes()}
    for parent, child in graph.bipartite_edges():
        parents[child].add(parent)
        children[parent].add(child)
    return parents, children


def _descendants(start: str, children: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    stack = list(children[start])
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(children[node])
    return seen


def _simple_paths(
    source: str,
    target: str,
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
    max_paths: int,
) -> list[list[str]]:
    paths: list[list[str]] = []
    path = [source]
    visited = {source}

    def dfs(node: str) -> None:
        if len(paths) >= max_paths:
            return
        if node == target:
            paths.append(list(path))
            return
        for neighbor in parents[node] | children[node]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            path.append(neighbor)
            dfs(neighbor)
            path.pop()
            visited.remove(neighbor)

    dfs(source)
    return paths


def _path_open(
    path: list[str],
    conditioned: frozenset[str],
    parents: dict[str, set[str]],
    children: dict[str, set[str]],
) -> bool:
    if len(path) < 3:
        return True
    for index in range(1, len(path) - 1):
        previous, node, nxt = path[index - 1], path[index], path[index + 1]
        collider = previous in parents[node] and nxt in parents[node]
        if collider:
            if node not in conditioned and not (_descendants(node, children) & set(conditioned)):
                return False
        elif node in conditioned:
            return False
    return True


def d_separated(
    graph: MechanismGraph,
    x: object,
    y: object,
    given: object = (),
    max_paths: int = 1024,
) -> bool:
    """d*-separation in the bipartite blowup with equality-based closure."""

    x_set = frozenset({x} if isinstance(x, str) else set(x))  # type: ignore[arg-type]
    y_set = frozenset({y} if isinstance(y, str) else set(y))  # type: ignore[arg-type]
    z_set = frozenset({given} if isinstance(given, str) else set(given))  # type: ignore[arg-type]
    if x_set & y_set or x_set & z_set or y_set & z_set:
        raise ValueError("X, Y, and conditioning set must be pairwise disjoint.")

    conditioned = deterministic_closure(graph, z_set)
    if x_set & conditioned or y_set & conditioned:
        return True

    parents, children = _build_adjacency(graph)
    for source in x_set:
        for target in y_set:
            for path in _simple_paths(source, target, parents, children, max_paths):
                if _path_open(path, conditioned, parents, children):
                    return False
    return True
