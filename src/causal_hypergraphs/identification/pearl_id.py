from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass

from causal_hypergraphs.expression import Expression, Probability, Product, SumOut

from .results import Assumption, IdentificationResult, Identified, ProofStep, Unidentified


def _ordered(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(sorted(str(v) for v in values))


def _bidirected_edge(edge: tuple[str, str]) -> tuple[str, str]:
    left, right = edge
    if left == right:
        raise ValueError("Bidirected self-edges are not allowed.")
    return tuple(sorted((str(left), str(right))))  # type: ignore[return-value]


@dataclass(frozen=True)
class ADMG:
    """Pearl-style acyclic directed mixed graph over observed random variables."""

    nodes: tuple[str, ...]
    directed_edges: tuple[tuple[str, str], ...] = ()
    bidirected_edges: tuple[tuple[str, str], ...] = ()

    def __init__(
        self,
        nodes: object,
        directed_edges: Iterable[tuple[str, str]] = (),
        bidirected_edges: Iterable[tuple[str, str]] = (),
    ) -> None:
        node_tuple = _ordered(nodes)
        node_set = set(node_tuple)
        directed = tuple(sorted((str(a), str(b)) for a, b in directed_edges))
        bidirected = tuple(sorted({_bidirected_edge(edge) for edge in bidirected_edges}))
        for source, target in directed:
            if source == target:
                raise ValueError("Directed self-edges are not allowed.")
            if source not in node_set or target not in node_set:
                raise ValueError(f"Directed edge {source!r}->{target!r} references unknown node.")
        for left, right in bidirected:
            if left not in node_set or right not in node_set:
                raise ValueError(
                    f"Bidirected edge {left!r}<->{right!r} references unknown node."
                )
        object.__setattr__(self, "nodes", node_tuple)
        object.__setattr__(self, "directed_edges", directed)
        object.__setattr__(self, "bidirected_edges", bidirected)
        self.topological_order()

    @property
    def node_set(self) -> frozenset[str]:
        return frozenset(self.nodes)

    def parents(self, node: str) -> frozenset[str]:
        return frozenset(source for source, target in self.directed_edges if target == node)

    def children(self, node: str) -> frozenset[str]:
        return frozenset(target for source, target in self.directed_edges if source == node)

    def topological_order(self) -> tuple[str, ...]:
        in_degree = {node: 0 for node in self.nodes}
        children = {node: set() for node in self.nodes}
        for source, target in self.directed_edges:
            in_degree[target] += 1
            children[source].add(target)
        queue = [node for node in self.nodes if in_degree[node] == 0]
        order: list[str] = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for child in sorted(children[current]):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(order) != len(self.nodes):
            raise ValueError("ADMG directed component must be acyclic.")
        return tuple(order)

    def ancestors(
        self,
        nodes: object,
        remove_incoming_to: object = (),
    ) -> frozenset[str]:
        targets = set(_ordered(nodes))
        blocked_incoming = set(_ordered(remove_incoming_to))
        ancestors = set(targets)
        stack = list(targets)
        while stack:
            node = stack.pop()
            for parent, child in self.directed_edges:
                if child != node or child in blocked_incoming:
                    continue
                if parent not in ancestors:
                    ancestors.add(parent)
                    stack.append(parent)
        return frozenset(ancestors)

    def induced(self, nodes: object) -> ADMG:
        keep = set(_ordered(nodes))
        return ADMG(
            nodes=keep,
            directed_edges=[
                edge for edge in self.directed_edges if edge[0] in keep and edge[1] in keep
            ],
            bidirected_edges=[
                edge for edge in self.bidirected_edges if edge[0] in keep and edge[1] in keep
            ],
        )

    def districts(self, nodes: object | None = None) -> tuple[tuple[str, ...], ...]:
        active = set(self.nodes if nodes is None else _ordered(nodes))
        adjacency = {node: set() for node in active}
        for left, right in self.bidirected_edges:
            if left in active and right in active:
                adjacency[left].add(right)
                adjacency[right].add(left)
        districts: list[tuple[str, ...]] = []
        seen: set[str] = set()
        for node in sorted(active):
            if node in seen:
                continue
            component: set[str] = set()
            stack = [node]
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(adjacency[current] - component)
            seen.update(component)
            districts.append(tuple(sorted(component)))
        return tuple(districts)

    def m_separated(self, source: str, target: str, given: Iterable[str]) -> bool:
        """Is `source` m-separated from `target` by `given`?

        m-separation is d-separation for ADMGs. It is decided here by the standard route:
        expand each bidirected edge into an explicit latent common parent, which is what a
        bidirected edge abbreviates, then apply the ancestral-moral-graph criterion to the
        resulting DAG (Lauritzen et al. 1990). Two nodes are separated when every path
        between them is blocked, which after moralization is ordinary graph connectivity
        with the conditioning set removed.
        """
        conditioning = set(_ordered(given))
        latents = {
            f"__u{index}": edge for index, edge in enumerate(self.bidirected_edges)
        }
        parents: dict[str, set[str]] = {node: set() for node in self.nodes}
        parents.update({name: set() for name in latents})
        for origin, destination in self.directed_edges:
            parents[destination].add(origin)
        for name, (left, right) in latents.items():
            parents[left].add(name)
            parents[right].add(name)

        # Ancestral subgraph over the query, then moralize it.
        wanted = {source, target} | conditioning
        ancestors: set[str] = set()
        stack = list(wanted)
        while stack:
            node = stack.pop()
            if node in ancestors:
                continue
            ancestors.add(node)
            stack.extend(parents[node])

        adjacency: dict[str, set[str]] = {node: set() for node in ancestors}
        for node in ancestors:
            kin = sorted(parents[node] & ancestors)
            for parent in kin:
                adjacency[node].add(parent)
                adjacency[parent].add(node)
            for first, second in itertools.combinations(kin, 2):
                adjacency[first].add(second)
                adjacency[second].add(first)

        blocked = conditioning
        seen = {source}
        stack = [source]
        while stack:
            node = stack.pop()
            if node == target:
                return False
            for neighbour in adjacency.get(node, ()):
                if neighbour in blocked or neighbour in seen:
                    continue
                seen.add(neighbour)
                stack.append(neighbour)
        return target not in seen

    def has_bidirected_path(self, source: str, target: str) -> bool:
        if source == target:
            return True
        districts = self.districts()
        return any(source in district and target in district for district in districts)

    def directed_paths(
        self,
        source: str,
        target: str,
        max_paths: int = 256,
    ) -> tuple[tuple[str, ...], ...]:
        paths: list[tuple[str, ...]] = []
        stack: list[tuple[str, tuple[str, ...]]] = [(source, (source,))]
        while stack and len(paths) < max_paths:
            current, path = stack.pop()
            if current == target:
                paths.append(path)
                continue
            for child in sorted(self.children(current), reverse=True):
                if child not in path:
                    stack.append((child, (*path, child)))
        return tuple(paths)


@dataclass(frozen=True)
class PearlIDQuery:
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...] = ()

    def __init__(self, outcomes: object, interventions: object = ()) -> None:
        object.__setattr__(self, "outcomes", _ordered(outcomes))
        object.__setattr__(self, "interventions", _ordered(interventions))


@dataclass(frozen=True)
class PearlHedgeWitness:
    districts: tuple[tuple[str, ...], ...]
    explanation: str


PEARL_ASSUMPTIONS = (
    Assumption("Pearl ADMG", "Input graph is a Pearl-style acyclic directed mixed graph."),
    Assumption("Observed nodes", "ADMG nodes are the variables available to the backend."),
)


class PearlIDBackend:
    """Small isolated Pearl-ID backend.

    This backend is intentionally conservative. It identifies base observational
    queries, Markovian truncated-factorization queries, and the canonical
    front-door pattern. Other cases return `Unidentified` with a witness-like
    explanation instead of fabricating a formula.
    """

    def identify(self, graph: ADMG, query: PearlIDQuery) -> IdentificationResult:
        outcomes = set(query.outcomes)
        interventions = set(query.interventions)
        if not outcomes:
            raise ValueError("Pearl ID query must contain at least one outcome.")
        unknown = (outcomes | interventions) - graph.node_set
        if unknown:
            raise ValueError(f"Pearl ID query references unknown nodes: {sorted(unknown)}")
        if outcomes & interventions:
            raise ValueError("Outcomes and interventions must be disjoint.")

        from .shpitser import Hedge, identify_admg_effect

        outcome_names = tuple(sorted(outcomes))
        intervention_names = tuple(sorted(interventions))
        answer = identify_admg_effect(graph, outcome_names, intervention_names)

        if isinstance(answer, Hedge):
            return Unidentified(
                reason=(
                    f"P({','.join(outcome_names)} | do({','.join(intervention_names)})) is "
                    "not identifiable in this ADMG: the recursion reaches a hedge."
                ),
                witness=answer,
                assumptions=PEARL_ASSUMPTIONS,
                derivation=(
                    ProofStep("Shpitser-Pearl ID", "Ran the seven-line recursion."),
                    ProofStep("Hedge", str(answer)),
                ),
            )

        expression, aliases = answer
        return Identified(
            expression=expression,
            theorem="Pearl-ID (Shpitser-Pearl)",
            assumptions=PEARL_ASSUMPTIONS,
            aliases=aliases,
            derivation=(
                ProofStep(
                    "Shpitser-Pearl ID",
                    "Sound and complete for P(y | do(x)) in a semi-Markovian model; a "
                    "formula returned here is correct, and a failure exhibits a hedge.",
                ),
            )
            + (
                (
                    ProofStep(
                        "Copied variables",
                        f"{sorted(aliases)} are fresh names for {sorted(set(aliases.values()))}, "
                        "introduced so an inner sum does not capture a value bound outside it.",
                    ),
                )
                if aliases
                else ()
            ),
        )

    def _markovian_truncated_factorization(
        self,
        graph: ADMG,
        outcomes: set[str],
        interventions: set[str],
    ) -> Expression:
        factors = [
            Probability(node, given=graph.parents(node))
            for node in graph.topological_order()
            if node not in interventions
        ]
        eliminate = graph.node_set - outcomes - interventions
        return SumOut(eliminate, Product(factors))

    def _frontdoor_expression(
        self,
        graph: ADMG,
        outcomes: set[str],
        interventions: set[str],
    ) -> Expression | None:
        if len(outcomes) != 1 or len(interventions) != 1:
            return None
        x = next(iter(interventions))
        y = next(iter(outcomes))
        candidates = sorted(graph.node_set - {x, y})
        for z in candidates:
            if (x, z) not in graph.directed_edges or (z, y) not in graph.directed_edges:
                continue
            if graph.has_bidirected_path(x, z):
                continue
            paths = graph.directed_paths(x, y)
            if not paths or any(z not in path[1:-1] for path in paths):
                continue
            if not graph.has_bidirected_path(x, y):
                continue
            x_prime = f"{x}_prime"
            inner = SumOut(
                x_prime,
                Product([Probability(y, given=(x_prime, z)), Probability(x_prime)]),
            )
            return SumOut(z, Product([Probability(z, given=x), inner]))
        return None


def identify_effect(
    graph: ADMG,
    outcomes: object,
    interventions: object = (),
) -> IdentificationResult:
    return PearlIDBackend().identify(graph, PearlIDQuery(outcomes, interventions))
