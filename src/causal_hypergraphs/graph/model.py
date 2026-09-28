from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def _ordered(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (_identifier(values),)
    if not isinstance(values, Iterable):
        raise TypeError("Identifiers must be an iterable of nonempty strings.")
    names = tuple(_identifier(value) for value in values)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate identifiers are not valid incidence entries.")
    return tuple(sorted(names))


def _identifier(value: object) -> str:
    """Validate native IDs without coercing distinct external IDs to one string."""
    if not isinstance(value, str):
        raise TypeError("Identifiers must be nonempty strings; convert external IDs explicitly.")
    if not value or not value.strip():
        raise ValueError("Identifiers must be nonempty strings.")
    return value


def _fingerprint(value: object) -> str:
    """A process-independent digest of a canonical, JSON-compatible payload."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_equalities(groups: object) -> tuple[tuple[str, ...], ...]:
    if groups is None:
        return ()
    if isinstance(groups, str) or not isinstance(groups, Iterable):
        raise TypeError("Output equalities must be an iterable of identifier groups.")
    return tuple(sorted(_ordered(group) for group in groups))


@dataclass(frozen=True, init=False)
class Mechanism:
    """Pure typed incidence for one mechanism.

    Structural functions/noise live outside the identification compiler. The compiler only needs
    typed incidence plus optional metadata such as latent status and declared output equalities.
    """

    name: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    latent: bool = False
    output_equalities: tuple[tuple[str, ...], ...] = ()

    def __init__(
        self,
        name: str,
        inputs: Iterable[str] | str = (),
        outputs: Iterable[str] | str = (),
        latent: bool = False,
        output_equalities: Iterable[Iterable[str] | str] = (),
    ) -> None:
        if not isinstance(latent, bool):
            raise TypeError("Mechanism latent status must be a boolean.")
        object.__setattr__(self, "name", _identifier(name))
        object.__setattr__(self, "inputs", _ordered(inputs))
        object.__setattr__(self, "outputs", _ordered(outputs))
        object.__setattr__(self, "latent", latent)
        object.__setattr__(self, "output_equalities", _normalize_equalities(output_equalities))

    @classmethod
    def from_spec(cls, name: str, spec: Mechanism | Mapping[str, Any]) -> Mechanism:
        name = _identifier(name)
        if isinstance(spec, Mechanism):
            if spec.name != name:
                return cls(
                    name=name,
                    inputs=spec.inputs,
                    outputs=spec.outputs,
                    latent=spec.latent,
                    output_equalities=spec.output_equalities,
                )
            return spec
        if not isinstance(spec, Mapping):
            raise TypeError("A mechanism specification must be a Mechanism or a mapping.")
        return cls(
            name=name,
            inputs=_ordered(spec.get("inputs", ())),
            outputs=_ordered(spec.get("outputs", ())),
            latent=spec.get("latent", False),
            output_equalities=_normalize_equalities(spec.get("output_equalities", ())),
        )

    @property
    def boundary(self) -> frozenset[str]:
        return frozenset(self.inputs) | frozenset(self.outputs)


@dataclass(frozen=True, init=False)
class MechanismGraph:
    """Typed mechanism hypergraph used by the identification compiler.

    `fallback_variables=None` declares symbolic joint fallback coverage for every mechanism.
    Passing an explicit set makes fallback policy strict, which lets callers force refusal when
    a mechanism deletion would orphan an output without an intervention policy.

    Inputs are copied and normalized. Stored identifiers are nonempty strings, incidence axes
    are sorted tuples, and the mechanism mapping is immutable. Cycles are valid structure;
    each inference algorithm checks its own acyclicity requirements.
    """

    variables: tuple[str, ...]
    mechanisms: Mapping[str, Mechanism]
    observed_variables: tuple[str, ...]
    fallback_variables: tuple[str, ...]
    assumptions: frozenset[str]

    def __init__(
        self,
        variables: Iterable[str] | str,
        mechanisms: Mapping[str, Mechanism | Mapping[str, Any]],
        observed_variables: Iterable[str] | str | None = None,
        fallback_variables: Iterable[str] | str | None = None,
        assumptions: Iterable[str] = frozenset(),
    ) -> None:
        ordered_variables = _ordered(variables)
        if not isinstance(mechanisms, Mapping):
            raise TypeError("Mechanisms must be a mapping from string IDs to specifications.")
        normalized = {
            _identifier(name): Mechanism.from_spec(name, spec)
            for name, spec in mechanisms.items()
        }
        observed = ordered_variables if observed_variables is None else _ordered(observed_variables)
        fallback = ordered_variables if fallback_variables is None else _ordered(fallback_variables)

        object.__setattr__(self, "variables", ordered_variables)
        object.__setattr__(self, "mechanisms", MappingProxyType(dict(sorted(normalized.items()))))
        object.__setattr__(self, "observed_variables", observed)
        object.__setattr__(self, "fallback_variables", fallback)
        object.__setattr__(self, "assumptions", frozenset(_ordered(assumptions)))

        self.validate()

    def fingerprint(self) -> str:
        """Stable content identity, including observation and policy declarations."""
        return _fingerprint([
            "MechanismGraph-v1",
            self.variables,
            [
                [name, m.inputs, m.outputs, m.latent, m.output_equalities]
                for name, m in self.mechanisms.items()
            ],
            self.observed_variables,
            self.fallback_variables,
            sorted(self.assumptions),
        ])

    def __hash__(self) -> int:
        return hash((
            self.variables, tuple(self.mechanisms.items()), self.observed_variables,
            self.fallback_variables, self.assumptions,
        ))

    @property
    def variable_set(self) -> frozenset[str]:
        return frozenset(self.variables)

    @property
    def observed_set(self) -> frozenset[str]:
        return frozenset(self.observed_variables)

    @property
    def hidden_variables(self) -> frozenset[str]:
        return self.variable_set - self.observed_set

    @property
    def fallback_set(self) -> frozenset[str]:
        return frozenset(self.fallback_variables)

    @property
    def latent_mechanism_names(self) -> frozenset[str]:
        return frozenset(name for name, mechanism in self.mechanisms.items() if mechanism.latent)

    @property
    def produced_variables(self) -> frozenset[str]:
        produced: set[str] = set()
        for mechanism in self.mechanisms.values():
            produced.update(mechanism.outputs)
        return frozenset(produced)

    @property
    def exogenous_variables(self) -> frozenset[str]:
        return self.variable_set - self.produced_variables

    def validate(self) -> None:
        var_set = set(self.variables)
        observed = set(self.observed_variables)
        fallback = set(self.fallback_variables)

        if not observed <= var_set:
            missing = sorted(observed - var_set)
            raise ValueError(f"Observed variables not in graph: {missing}")
        if not fallback <= var_set:
            missing = sorted(fallback - var_set)
            raise ValueError(f"Fallback variables not in graph: {missing}")

        producer: dict[str, list[str]] = {}
        for name, mechanism in self.mechanisms.items():
            inputs = set(mechanism.inputs)
            outputs = set(mechanism.outputs)
            if not inputs <= var_set:
                raise ValueError(
                    f"Mechanism {name!r} has inputs outside V: {sorted(inputs - var_set)}"
                )
            if not outputs <= var_set:
                raise ValueError(
                    f"Mechanism {name!r} has outputs outside V: {sorted(outputs - var_set)}"
                )
            if inputs & outputs:
                raise ValueError(
                    f"Mechanism {name!r} has overlapping inputs and outputs: "
                    f"{sorted(inputs & outputs)}"
                )
            for group in mechanism.output_equalities:
                if not set(group) <= outputs:
                    raise ValueError(
                        f"Mechanism {name!r} declares output equality outside outputs: {group}"
                    )
            for output in outputs:
                producer.setdefault(output, []).append(name)

        duplicates = {v: names for v, names in producer.items() if len(names) > 1}
        if duplicates:
            detail = ", ".join(f"{v}: {names}" for v, names in sorted(duplicates.items()))
            raise ValueError(f"C4 violation: variables with multiple producers ({detail})")

        # C1 is a query condition, not a construction-time restriction. An unrelated
        # feedback component must not prevent queries whose required kernels are acyclic.
        # `identify` checks the relevant closure and records semantic assumptions.

    def get_mechanism(self, name: str) -> Mechanism:
        try:
            return self.mechanisms[name]
        except KeyError as exc:
            raise KeyError(f"No mechanism named {name!r}") from exc

    def consumers(self) -> dict[str, tuple[str, ...]]:
        """The mechanisms reading each variable, indexed by variable."""
        index: dict[str, list[str]] = {}
        for name in self.mechanisms:
            for variable in self.get_mechanism(name).inputs:
                index.setdefault(variable, []).append(name)
        return {variable: tuple(names) for variable, names in index.items()}

    def observed_closure(
        self, variables: object, observed: frozenset[str] | None = None
    ) -> frozenset[str]:
        """Observed variables reached from `variables`, through hidden ones only.

        One step follows a mechanism: from a variable, every mechanism consuming it
        produces its outputs. An observed variable is a stopping point rather than a node
        to pass through, which is the "interior nodes are all latent" condition of the
        standard latent projection -- and the same walk answers "can this hidden variable
        move anything anyone measured", which is what decides whether it obstructs
        identification or is simply removable.

        Members of `variables` that are themselves observed are returned. `observed`
        overrides the graph's own observed set, for callers asking what would be
        identifiable under a different measurement plan.
        """
        observed = self.observed_set if observed is None else observed
        consumers = self.consumers()
        reached: set[str] = set()
        seen: set[str] = set()
        stack = list(_ordered(variables))
        while stack:
            variable = stack.pop()
            if variable in seen:
                continue
            seen.add(variable)
            if variable in observed:
                reached.add(variable)
                continue
            for name in consumers.get(variable, ()):
                stack.extend(self.get_mechanism(name).outputs)
        return frozenset(reached)

    def removable_outputs(
        self, mechanism_name: str, observed: frozenset[str] | None = None
    ) -> tuple[str, ...]:
        """Hidden outputs of `mechanism_name` that no observable depends on.

        These do not obstruct identification. `delete(m)` installs a joint policy over
        every output, supplied by the caller, so a hidden coordinate that reaches nothing
        observed is summed out of that declared table -- its domain is part of the
        intervention rather than something the data must supply.

        A hidden output that *does* reach an observation is the opposite case and is not
        identifiable at all: relabelling it preserves every observed distribution and
        changes the policy defined on its values.
        """
        resolved = self.observed_set if observed is None else observed
        mechanism = self.get_mechanism(mechanism_name)
        hidden = set(mechanism.outputs) - resolved
        return tuple(
            sorted(
                name
                for name in hidden
                if not self.observed_closure((name,), resolved)
            )
        )

    def mechanism_dependencies(self) -> dict[str, set[str]]:
        """`m -> m'` whenever an output of `m` is an input of `m'`.

        Indexed by variable rather than by comparing every pair of mechanisms. Both give
        the same edges, but this runs in the number of incidences instead of the square of
        the number of mechanisms. Construction does not call this method; algorithms
        request dependency information only when they need it.
        """
        consumers: dict[str, list[str]] = {}
        for name, mechanism in self.mechanisms.items():
            for variable in mechanism.inputs:
                consumers.setdefault(variable, []).append(name)

        edges: dict[str, set[str]] = {name: set() for name in self.mechanisms}
        for name, mechanism in self.mechanisms.items():
            for variable in mechanism.outputs:
                edges[name].update(
                    other for other in consumers.get(variable, ()) if other != name
                )
        return edges

    def mechanism_components(self) -> tuple[tuple[str, ...], ...]:
        """The strongly connected components of the mechanism dependency graph.

        Sorted, with each component's members sorted, so the answer is stable between runs
        -- a cost or a refusal that varied with dict ordering could not be checked against
        anything. Components are returned for acyclic graphs too, as singletons.

        Tarjan's algorithm, iterative rather than recursive: a chain of twenty thousand
        mechanisms is an ordinary size for this library and would overflow the interpreter
        stack.
        """
        edges = self.mechanism_dependencies()
        index: dict[str, int] = {}
        low: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        components: list[tuple[str, ...]] = []
        counter = 0

        for root in sorted(edges):
            if root in index:
                continue
            work: list[tuple[str, list[str]]] = [(root, sorted(edges[root]))]
            index[root] = low[root] = counter
            counter += 1
            stack.append(root)
            on_stack.add(root)
            while work:
                node, pending = work[-1]
                if pending:
                    child = pending.pop()
                    if child not in index:
                        index[child] = low[child] = counter
                        counter += 1
                        stack.append(child)
                        on_stack.add(child)
                        work.append((child, sorted(edges[child])))
                    elif child in on_stack:
                        low[node] = min(low[node], index[child])
                    continue
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    component: list[str] = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == node:
                            break
                    components.append(tuple(sorted(component)))
        return tuple(sorted(components))

    @property
    def cyclic_mechanisms(self) -> frozenset[str]:
        """Mechanisms that lie on a cycle.

        Exactly the members of a strongly connected component with more than one member.
        A component of size one is never cyclic here: a self-edge would need a mechanism's
        output to be one of its own inputs, which C3 already forbids.
        """
        return frozenset(
            name
            for component in self.mechanism_components()
            if len(component) > 1
            for name in component
        )

    def is_mechanism_acyclic(self) -> bool:
        edges = self.mechanism_dependencies()
        in_degree = {name: 0 for name in edges}
        for successors in edges.values():
            for successor in successors:
                in_degree[successor] += 1
        queue = [name for name, degree in in_degree.items() if degree == 0]
        visited = 0
        while queue:
            current = queue.pop()
            visited += 1
            for successor in edges[current]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)
        return visited == len(edges)

    def bipartite_edges(self) -> frozenset[tuple[str, str]]:
        edges: set[tuple[str, str]] = set()
        for name, mechanism in self.mechanisms.items():
            for variable in mechanism.inputs:
                edges.add((variable, name))
            for variable in mechanism.outputs:
                edges.add((name, variable))
        return frozenset(edges)

    def bipartite_nodes(self) -> frozenset[str]:
        return self.variable_set | frozenset(self.mechanisms)

    def missing_boundary_variables(
        self,
        mechanism_name: str,
        observed_variables: object | None = None,
    ) -> tuple[str, ...]:
        mechanism = self.get_mechanism(mechanism_name)
        observed = (
            self.observed_set
            if observed_variables is None
            else frozenset(_ordered(observed_variables))
        )
        return tuple(sorted(mechanism.boundary - observed))

    def missing_fallback_variables(self, mechanism_name: str) -> tuple[str, ...]:
        mechanism = self.get_mechanism(mechanism_name)
        return tuple(sorted(set(mechanism.outputs) - self.fallback_set))
