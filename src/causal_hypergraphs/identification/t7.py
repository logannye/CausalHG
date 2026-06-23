from __future__ import annotations

from dataclasses import dataclass

from causal_hypergraphs.graph import MechanismGraph

from .pearl_id import ADMG
from .queries import DeleteMechanism, ReplaceMechanism
from .results import Unknown


def variable_node(variable: str) -> str:
    return f"V:{variable}"


def mechanism_node(mechanism: str) -> str:
    return f"M:{mechanism}"


@dataclass(frozen=True)
class BipartiteDAG:
    """Typed bipartite DAG before hidden-node projection."""

    observed_nodes: tuple[str, ...]
    hidden_nodes: tuple[str, ...]
    directed_edges: tuple[tuple[str, str], ...]

    def __init__(
        self,
        observed_nodes: object,
        hidden_nodes: object = (),
        directed_edges: object = (),
    ) -> None:
        object.__setattr__(self, "observed_nodes", _ordered(observed_nodes))
        object.__setattr__(self, "hidden_nodes", _ordered(hidden_nodes))
        edges = tuple(sorted((str(a), str(b)) for a, b in directed_edges))  # type: ignore[arg-type]
        object.__setattr__(self, "directed_edges", edges)

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset(self.observed_nodes) | frozenset(self.hidden_nodes)

    def children(self, node: str) -> frozenset[str]:
        return frozenset(target for source, target in self.directed_edges if source == node)


@dataclass(frozen=True)
class BipartiteADMG:
    """Interface placeholder for the latent-projected bipartite ADMG used by T7."""

    observed_nodes: tuple[str, ...] = ()
    hidden_nodes: tuple[str, ...] = ()
    directed_edges: tuple[tuple[str, str], ...] = ()
    bidirected_edges: tuple[tuple[str, str], ...] = ()

    def __init__(
        self,
        observed_nodes: object = (),
        hidden_nodes: object = (),
        directed_edges: object = (),
        bidirected_edges: object = (),
    ) -> None:
        object.__setattr__(self, "observed_nodes", _ordered(observed_nodes))
        object.__setattr__(self, "hidden_nodes", _ordered(hidden_nodes))
        directed = tuple(
            sorted((str(a), str(b)) for a, b in directed_edges)  # type: ignore[arg-type]
        )
        bidirected = tuple(
            sorted({_bidirected_edge(edge) for edge in bidirected_edges})  # type: ignore[arg-type]
        )
        object.__setattr__(self, "directed_edges", directed)
        object.__setattr__(self, "bidirected_edges", bidirected)

    def to_admg(self) -> ADMG:
        return ADMG(
            nodes=self.observed_nodes,
            directed_edges=self.directed_edges,
            bidirected_edges=self.bidirected_edges,
        )


@dataclass(frozen=True)
class LatentProjectionPlan:
    """Planned transform from a typed mechanism graph to a Pearl-style ADMG."""

    source_graph: object
    target_admg: BipartiteADMG | None = None
    status: str = "not_implemented"


@dataclass(frozen=True)
class StochasticInterventionReduction:
    """Planned reduction of mechanism deletion/replacement to Pearl ID inputs."""

    target_outputs: tuple[str, ...]
    conditioning_inputs: tuple[str, ...]
    target_mechanism: str = ""
    query_type: str = ""
    replacement_kernel: str | None = None
    admg: BipartiteADMG | None = None
    status: str = "reduced"

    def __init__(
        self,
        target_outputs: object,
        conditioning_inputs: object,
        replacement_kernel: str | None = None,
        target_mechanism: str = "",
        query_type: str = "",
        admg: BipartiteADMG | None = None,
        status: str = "reduced",
    ) -> None:
        object.__setattr__(self, "target_mechanism", str(target_mechanism))
        object.__setattr__(self, "query_type", str(query_type))
        object.__setattr__(self, "target_outputs", _ordered(target_outputs))
        object.__setattr__(self, "conditioning_inputs", _ordered(conditioning_inputs))
        object.__setattr__(self, "replacement_kernel", replacement_kernel)
        object.__setattr__(self, "admg", admg)
        object.__setattr__(self, "status", str(status))


@dataclass(frozen=True)
class HedgeWitness:
    """Pearl hedge witness returned by future non-identification paths."""

    districts: tuple[tuple[str, ...], ...] = ()
    explanation: str = ""


@dataclass(frozen=True)
class HyperHedgeWitness:
    """Mechanism-level lift of a Pearl hedge witness."""

    mechanisms: tuple[str, ...] = ()
    pearl_witness: HedgeWitness | None = None
    explanation: str = ""


@dataclass(frozen=True)
class T7ReductionPlaceholder:
    """Future hook for Pearl-ID reduction of boundary-violating mechanism queries."""

    reason: str = "T7 Pearl-ID reduction is not implemented in milestone 1."
    admg: BipartiteADMG | None = None
    stochastic_intervention: StochasticInterventionReduction | None = None


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


def build_bipartite_dag(graph: MechanismGraph) -> BipartiteDAG:
    hidden_nodes = {variable_node(v) for v in graph.hidden_variables}
    hidden_nodes.update(mechanism_node(m) for m in graph.latent_mechanism_names)

    observed_nodes = {variable_node(v) for v in graph.variable_set - graph.hidden_variables}
    observed_nodes.update(
        mechanism_node(name)
        for name in graph.mechanisms
        if name not in graph.latent_mechanism_names
    )

    directed_edges: set[tuple[str, str]] = set()
    for name, mechanism in graph.mechanisms.items():
        m_node = mechanism_node(name)
        for variable in mechanism.inputs:
            directed_edges.add((variable_node(variable), m_node))
        for variable in mechanism.outputs:
            directed_edges.add((m_node, variable_node(variable)))

    return BipartiteDAG(
        observed_nodes=observed_nodes,
        hidden_nodes=hidden_nodes,
        directed_edges=directed_edges,
    )


def _observed_reachable_through_hidden(
    dag: BipartiteDAG,
    source: str,
    observed: frozenset[str],
    hidden: frozenset[str],
) -> frozenset[str]:
    reached: set[str] = set()
    stack = list(dag.children(source))
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in observed:
            reached.add(current)
            continue
        if current in hidden:
            stack.extend(dag.children(current))
    return frozenset(reached)


def latent_project(graph: MechanismGraph) -> BipartiteADMG:
    dag = build_bipartite_dag(graph)
    observed = frozenset(dag.observed_nodes)
    hidden = frozenset(dag.hidden_nodes)

    directed_edges: set[tuple[str, str]] = set()
    for source in observed:
        for target in _observed_reachable_through_hidden(dag, source, observed, hidden):
            if source != target:
                directed_edges.add((source, target))

    bidirected_edges: set[tuple[str, str]] = set()
    for hidden_node in hidden:
        descendants = sorted(_observed_reachable_through_hidden(dag, hidden_node, observed, hidden))
        for index, left in enumerate(descendants):
            for right in descendants[index + 1 :]:
                bidirected_edges.add(_bidirected_edge((left, right)))

    return BipartiteADMG(
        observed_nodes=observed,
        hidden_nodes=hidden,
        directed_edges=directed_edges,
        bidirected_edges=bidirected_edges,
    )


def reduce_mechanism_query_to_stochastic_intervention(
    graph: MechanismGraph,
    query: DeleteMechanism | ReplaceMechanism,
) -> StochasticInterventionReduction:
    target = graph.get_mechanism(query.target)
    replacement_kernel = None
    query_type = "delete"
    if isinstance(query, ReplaceMechanism):
        query_type = "replace"
        replacement_kernel = f"P_{query.replacement}"
    return StochasticInterventionReduction(
        target_mechanism=query.target,
        query_type=query_type,
        target_outputs=target.outputs,
        conditioning_inputs=target.inputs,
        replacement_kernel=replacement_kernel,
        admg=latent_project(graph),
    )


def identify_via_t7(
    graph: MechanismGraph | None = None,
    query: DeleteMechanism | ReplaceMechanism | None = None,
) -> Unknown:
    """Return an honest placeholder until ADMG latent projection and Pearl ID exist."""

    suggestions = ("Build bipartite ADMG, reduce to stochastic intervention, run Pearl ID.",)
    if graph is not None and query is not None:
        reduction = reduce_mechanism_query_to_stochastic_intervention(graph, query)
        suggestions = (
            f"Reduced {reduction.query_type}({reduction.target_mechanism}) to a "
            "stochastic intervention object.",
            *suggestions,
        )
    return Unknown(
        reason="T7 Pearl-ID reduction is not implemented.",
        next_algorithm="Build bipartite ADMG, reduce to stochastic intervention, run Pearl ID.",
        suggestions=suggestions,
    )
