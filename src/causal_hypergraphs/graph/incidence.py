"""Immutable directed incidence, separate from a supported causal interpretation.

Parallel named hyperedges, overlapping producers, and feedback are valid structure.
Only an explicit conversion to the independent-mechanism profile applies C3/C4.
An ordinary Pearl ADMG is a separate model family, not an interpretation selected
by attaching bidirected metadata to these edges.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .model import MechanismGraph, _fingerprint, _identifier, _ordered

INDEPENDENT_MECHANISMS = "independent-mechanisms"


class UnsupportedModelError(ValueError):
    """Valid incidence cannot be interpreted by the requested causal profile."""

    code = "unsupported_model"


def _freeze_metadata(value: object, active: set[int] | None = None) -> object:
    """Copy finite JSON-like metadata into immutable nested collections."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Metadata numbers must be finite.")
        return value
    if not isinstance(value, (Mapping, tuple, list)):
        raise TypeError("Metadata must contain only JSON values, lists, and string-keyed mappings.")
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError("Metadata must not contain recursive collections.")
    active.add(id(value))
    try:
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("Metadata mapping keys must be strings.")
            return MappingProxyType({
                key: _freeze_metadata(item, active) for key, item in sorted(value.items())
            })
        return tuple(_freeze_metadata(item, active) for item in value)
    finally:
        active.remove(id(value))


def _metadata(value: Mapping[str, object] | None) -> Mapping[str, object]:
    frozen = _freeze_metadata({} if value is None else value)
    if not isinstance(frozen, Mapping):
        raise TypeError("Metadata must be a string-keyed mapping.")
    return frozen


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _hashable(value: object) -> Any:
    if isinstance(value, Mapping):
        return tuple((key, _hashable(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_hashable(item) for item in value)
    return value


@dataclass(frozen=True, init=False)
class DirectedHyperedge:
    """A named pair of incidence sets without implied causal semantics.

    A node may occur in both roles here. The causal mechanism profile rejects that
    structure, but the generic representation preserves it without rewriting it.
    """

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    metadata: Mapping[str, object]

    def __init__(
        self,
        name: str,
        inputs: Iterable[str] | str = (),
        outputs: Iterable[str] | str = (),
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        object.__setattr__(self, "name", _identifier(name))
        object.__setattr__(self, "inputs", _ordered(inputs))
        object.__setattr__(self, "outputs", _ordered(outputs))
        object.__setattr__(self, "metadata", _metadata(metadata))

    @classmethod
    def from_spec(
        cls, name: str, spec: DirectedHyperedge | Mapping[str, Any]
    ) -> DirectedHyperedge:
        name = _identifier(name)
        if isinstance(spec, DirectedHyperedge):
            if name != spec.name:
                raise ValueError(f"Edge key {name!r} disagrees with edge name {spec.name!r}.")
            return spec
        if not isinstance(spec, Mapping):
            raise TypeError("An edge specification must be a DirectedHyperedge or mapping.")
        extra = set(spec) - {"inputs", "outputs", "metadata"}
        if extra:
            raise ValueError(f"Unknown edge specification fields: {sorted(extra)}")
        return cls(name, spec.get("inputs", ()), spec.get("outputs", ()), spec.get("metadata"))

    def __hash__(self) -> int:
        return hash((self.name, self.inputs, self.outputs, _hashable(self.metadata)))


@dataclass(frozen=True, init=False)
class DirectedHypergraph:
    """Generic directed hypergraph with immutable, explicitly named incidence.

    Nodes and edges use separate namespaces. Native IDs must already be nonempty
    strings; adapters must provide an explicit codec for other external ID types.
    Distinct named edges may have identical incidence. Isolated nodes are retained.
    """

    nodes: tuple[str, ...]
    edges: Mapping[str, DirectedHyperedge]
    metadata: Mapping[str, object]

    def __init__(
        self,
        nodes: Iterable[str] | str,
        edges: Mapping[str, DirectedHyperedge | Mapping[str, Any]],
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        normalized_nodes = _ordered(nodes)
        if not isinstance(edges, Mapping):
            raise TypeError("Edges must be a mapping from string IDs to specifications.")
        normalized_edges = {
            _identifier(name): DirectedHyperedge.from_spec(name, spec)
            for name, spec in edges.items()
        }
        node_set = frozenset(normalized_nodes)
        for name, edge in normalized_edges.items():
            missing = (set(edge.inputs) | set(edge.outputs)) - node_set
            if missing:
                raise ValueError(f"Edge {name!r} references missing nodes: {sorted(missing)}")
        object.__setattr__(self, "nodes", normalized_nodes)
        object.__setattr__(self, "edges", MappingProxyType(dict(sorted(normalized_edges.items()))))
        object.__setattr__(self, "metadata", _metadata(metadata))

    @property
    def node_set(self) -> frozenset[str]:
        return frozenset(self.nodes)

    def fingerprint(self) -> str:
        """Process-independent identity for the full incidence and metadata snapshot."""
        return _fingerprint([
            "DirectedHypergraph-v1", self.nodes,
            [
                [name, edge.inputs, edge.outputs, _plain(edge.metadata)]
                for name, edge in self.edges.items()
            ],
            _plain(self.metadata),
        ])

    def __hash__(self) -> int:
        return hash((self.nodes, tuple(self.edges.items()), _hashable(self.metadata)))

    def to_mechanism_graph(
        self,
        observed_variables: Iterable[str] | str | None = None,
        fallback_variables: Iterable[str] | str | None = None,
        assumptions: Iterable[str] = (),
        *,
        profile: str = INDEPENDENT_MECHANISMS,
    ) -> MechanismGraph:
        """Explicitly interpret incidence using a supported causal model profile.

        Metadata stays on this graph; the compiler consumes only the incidence and
        declared causal profile. Metadata cannot assert additional causal properties.
        """
        return CausalModelSpec(
            self, profile, observed_variables, fallback_variables, assumptions
        ).to_mechanism_graph()


@dataclass(frozen=True, init=False)
class CausalModelSpec:
    """A structural graph with an explicit causal interpretation and observations.

    The initial profile assumes independent mechanism noises and requires one
    producer per variable. It allows cycles as stored structure, while algorithms
    decide whether a particular query is supported. Choosing the profile declares
    its noise assumptions; it cannot verify them from incidence or observations.
    """

    graph: DirectedHypergraph
    profile: str
    observed_variables: tuple[str, ...]
    fallback_variables: tuple[str, ...]
    assumptions: frozenset[str]

    def __init__(
        self,
        graph: DirectedHypergraph,
        profile: str = INDEPENDENT_MECHANISMS,
        observed_variables: Iterable[str] | str | None = None,
        fallback_variables: Iterable[str] | str | None = None,
        assumptions: Iterable[str] = (),
    ) -> None:
        if not isinstance(graph, DirectedHypergraph):
            raise TypeError("A CausalModelSpec requires a DirectedHypergraph.")
        observed = graph.nodes if observed_variables is None else _ordered(observed_variables)
        fallback = graph.nodes if fallback_variables is None else _ordered(fallback_variables)
        for role, names in (("Observed", observed), ("Fallback", fallback)):
            missing = set(names) - graph.node_set
            if missing:
                raise ValueError(f"{role} variables not in graph: {sorted(missing)}")
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "profile", _identifier(profile))
        object.__setattr__(self, "observed_variables", observed)
        object.__setattr__(self, "fallback_variables", fallback)
        object.__setattr__(self, "assumptions", frozenset(_ordered(assumptions)))

    def to_mechanism_graph(self) -> MechanismGraph:
        if self.profile != INDEPENDENT_MECHANISMS:
            raise UnsupportedModelError(
                f"Unsupported causal profile {self.profile!r}; use {INDEPENDENT_MECHANISMS!r} "
                "for this conversion. Pearl ADMGs have their own interface."
            )
        collisions = set(self.graph.edges) & self.graph.node_set
        if collisions:
            raise UnsupportedModelError(
                "The mechanism compiler requires distinct node and mechanism IDs for its "
                f"bipartite representation; rename these edge IDs explicitly: {sorted(collisions)}"
            )
        try:
            return MechanismGraph(
                self.graph.nodes,
                {
                    name: {"inputs": edge.inputs, "outputs": edge.outputs}
                    for name, edge in self.graph.edges.items()
                },
                self.observed_variables,
                self.fallback_variables,
                self.assumptions,
            )
        except ValueError as error:
            raise UnsupportedModelError(
                f"Incidence is valid, but profile {self.profile!r} cannot use it: {error}"
            ) from error

    def fingerprint(self) -> str:
        return _fingerprint([
            "CausalModelSpec-v1", self.graph.fingerprint(), self.profile,
            self.observed_variables, self.fallback_variables, sorted(self.assumptions),
        ])
