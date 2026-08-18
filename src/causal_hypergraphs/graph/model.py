from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


def _ordered(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(sorted(str(v) for v in values))


def _normalize_equalities(groups: object) -> tuple[tuple[str, ...], ...]:
    if groups is None:
        return ()
    return tuple(_ordered(group) for group in groups)  # type: ignore[arg-type]


@dataclass(frozen=True)
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "inputs", _ordered(self.inputs))
        object.__setattr__(self, "outputs", _ordered(self.outputs))
        object.__setattr__(self, "output_equalities", _normalize_equalities(self.output_equalities))

    @classmethod
    def from_spec(cls, name: str, spec: Mechanism | Mapping[str, Any]) -> Mechanism:
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
        return cls(
            name=name,
            inputs=_ordered(spec.get("inputs", ())),
            outputs=_ordered(spec.get("outputs", ())),
            latent=bool(spec.get("latent", False)),
            output_equalities=_normalize_equalities(spec.get("output_equalities", ())),
        )

    @property
    def boundary(self) -> frozenset[str]:
        return frozenset(self.inputs) | frozenset(self.outputs)


@dataclass(frozen=True)
class MechanismGraph:
    """Typed mechanism hypergraph used by the identification compiler.

    `fallback_variables=None` means every variable has a named fallback distribution `P0(v)`.
    Passing an explicit set makes fallback policy strict, which lets callers force refusal when
    a mechanism deletion would orphan an output without an intervention policy.
    """

    variables: object
    mechanisms: Mapping[str, Mechanism | Mapping[str, Any]]
    observed_variables: object | None = None
    fallback_variables: object | None = None
    assumptions: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        variables = _ordered(self.variables)
        mechanisms = {
            str(name): Mechanism.from_spec(str(name), spec)
            for name, spec in self.mechanisms.items()
        }
        observed = (
            variables if self.observed_variables is None else _ordered(self.observed_variables)
        )
        fallback = (
            variables if self.fallback_variables is None else _ordered(self.fallback_variables)
        )

        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "mechanisms", mechanisms)
        object.__setattr__(self, "observed_variables", observed)
        object.__setattr__(self, "fallback_variables", fallback)

        self.validate()

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

        if not self.is_mechanism_acyclic():
            raise ValueError("C1 violation: mechanism dependency graph is cyclic")

    def get_mechanism(self, name: str) -> Mechanism:
        try:
            return self.mechanisms[name]
        except KeyError as exc:
            raise KeyError(f"No mechanism named {name!r}") from exc

    def mechanism_dependencies(self) -> dict[str, set[str]]:
        """`m -> m'` whenever an output of `m` is an input of `m'`.

        Indexed by variable rather than by comparing every pair of mechanisms. Both give
        the same edges, but this runs in the number of incidences instead of the square of
        the number of mechanisms -- and since every graph validates acyclicity at
        construction, the difference is the cost of *loading* a network at all. A
        20,000-mechanism graph is seconds one way and tens of seconds the other.
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
