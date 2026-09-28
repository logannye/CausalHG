"""Immutable intervention and quantity specifications shared by library backends.

These objects describe a question, not a statistical model or an identification
claim. A backend must separately validate that the graph and available data support
the requested operation. JointPolicy is a finite probability mass function; it is
never interpreted as a density or split into independent marginal policies.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
    from causal_hypergraphs.graph import MechanismGraph

Scalar: TypeAlias = str | bool | int | float


def _name(value: object, *, label: str = "identifier") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string, got {type(value).__name__}.")
    if not value.strip():
        raise ValueError(f"{label} must be nonempty.")
    return value


def _names(values: Iterable[str] | str, *, label: str) -> tuple[str, ...]:
    names = (values,) if isinstance(values, str) else tuple(values)
    for name in names:
        _name(name, label=label)
    if len(set(names)) != len(names):
        raise ValueError(f"{label} contains duplicate identifiers.")
    return names


def _scalar(value: object) -> Scalar:
    if not isinstance(value, str | bool | int | float):
        raise TypeError(
            "Intervention values must be strings, booleans, integers, or finite floats."
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Intervention values must be finite.")
    return value


@dataclass(frozen=True)
class HardIntervention:
    """Replace exactly the named variable equations by fixed scalar values.

For a variable that belongs to a joint-output mechanism, the other output
equations retain their original structural function and shared exogenous noise.
An empty assignment is an observational/no-op intervention.
"""

    assignments: Mapping[str, Scalar]

    def __post_init__(self) -> None:
        if not isinstance(self.assignments, Mapping):
            raise TypeError("assignments must be a mapping.")
        snapshot = {
            _name(name, label="Variable identifier"): _scalar(value)
            for name, value in self.assignments.items()
        }
        object.__setattr__(self, "assignments", MappingProxyType(snapshot))

    def __hash__(self) -> int:
        return hash((type(self), frozenset(self.assignments.items())))


@dataclass(frozen=True)
class JointPolicy:
    """An unconditional finite joint policy, independent of original model noise.

    Tuple coordinates follow ``variables`` exactly, including a nonalphabetical
    order. Omitted tuples have zero probability. The listed support must contain
    all nonzero masses and sum to one; no Cartesian completion is assumed.
    """

    variables: tuple[str, ...]
    probabilities: Mapping[tuple[Scalar, ...], float]
    name: str = "policy"

    def __init__(
        self,
        variables: tuple[str, ...],
        probabilities: Mapping[tuple[Any, ...], float],
        name: str = "policy",
    ) -> None:
        # Mapping's key type is invariant. Accept ordinary typed tuples here, then
        # validate each coordinate before exposing the immutable scalar-only table.
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "name", name)
        self.__post_init__()

    def __post_init__(self) -> None:
        if isinstance(self.variables, set | frozenset | Mapping):
            raise TypeError("Policy variables require an explicit axis order, such as a tuple.")
        variables = _names(self.variables, label="Policy variables")
        if not variables:
            raise ValueError("A joint policy must target at least one variable.")
        name = _name(self.name, label="Policy name")
        if not isinstance(self.probabilities, Mapping):
            raise TypeError("probabilities must be a mapping keyed by value tuples.")
        snapshot: dict[tuple[Scalar, ...], float] = {}
        for values, probability in self.probabilities.items():
            if not isinstance(values, tuple) or len(values) != len(variables):
                raise ValueError("Policy keys must be tuples matching the declared variable axes.")
            key = tuple(_scalar(value) for value in values)
            if not isinstance(probability, int | float):
                raise TypeError("Policy probabilities must be finite real numbers.")
            mass = float(probability)
            if not math.isfinite(mass) or mass < 0:
                raise ValueError("Policy probabilities must be finite and nonnegative.")
            snapshot[key] = mass
        if not math.isclose(math.fsum(snapshot.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Policy probabilities must sum to one.")
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "probabilities", MappingProxyType(snapshot))
        object.__setattr__(self, "name", name)

    def __hash__(self) -> int:
        return hash((type(self), self.variables, frozenset(self.probabilities.items()), self.name))

    def probability(self, assignment: Mapping[str, Scalar]) -> float:
        """Read the joint mass at an assignment; unlisted value tuples have mass zero."""
        return self.probabilities.get(tuple(assignment[name] for name in self.variables), 0.0)


@dataclass(frozen=True)
class Delete:
    """Delete a mechanism and jointly reset all its outputs using ``policy``."""

    target: str
    policy: JointPolicy

    def __post_init__(self) -> None:
        _name(self.target, label="Mechanism identifier")
        if not isinstance(self.policy, JointPolicy):
            raise TypeError("Deletion requires an explicit JointPolicy.")


@dataclass(frozen=True)
class Replace:
    """Replace a mechanism by a separately bound kernel with the same incidence.

    The name denotes a symbolic kernel during identification. Simulation and
    evaluation must bind that name explicitly and validate its input/output axes.
    """

    target: str
    replacement: str

    def __post_init__(self) -> None:
        _name(self.target, label="Mechanism identifier")
        _name(self.replacement, label="Replacement name")


AtomicIntervention: TypeAlias = HardIntervention | JointPolicy | Delete | Replace


def _declared_targets(operation: AtomicIntervention) -> frozenset[str]:
    if isinstance(operation, HardIntervention):
        return frozenset(operation.assignments)
    if isinstance(operation, JointPolicy):
        return frozenset(operation.variables)
    if isinstance(operation, Delete):
        return frozenset(operation.policy.variables)
    return frozenset()


def _check_conflicts(operations: tuple[AtomicIntervention, ...]) -> None:
    mechanisms: set[str] = set()
    variables: set[str] = set()
    for operation in operations:
        if isinstance(operation, Delete | Replace):
            if operation.target in mechanisms:
                raise ValueError(f"Conflicting operations on mechanism {operation.target!r}.")
            mechanisms.add(operation.target)
        targets = _declared_targets(operation)
        overlap = targets & variables
        if overlap:
            raise ValueError(f"Conflicting interventions on variables {sorted(overlap)}.")
        variables.update(targets)


@dataclass(frozen=True)
class Composite:
    """Simultaneously apply compatible operations, without order-dependent overrides.

    Nested composites are flattened. An empty composite denotes no intervention.
    Graph-dependent conflicts (for example replacing a producer while fixing one
    of its outputs) are checked by :func:`validate_intervention`.
    """

    operations: tuple[AtomicIntervention, ...]

    def __post_init__(self) -> None:
        flattened: list[AtomicIntervention] = []
        for operation in self.operations:
            if isinstance(operation, Composite):
                flattened.extend(operation.operations)
            elif isinstance(operation, HardIntervention | JointPolicy | Delete | Replace):
                flattened.append(operation)
            else:
                raise TypeError(f"Unsupported intervention operation: {type(operation).__name__}.")
        operations = tuple(flattened)
        _check_conflicts(operations)
        object.__setattr__(self, "operations", operations)


Intervention: TypeAlias = AtomicIntervention | Composite


def intervention_operations(intervention: Intervention) -> tuple[AtomicIntervention, ...]:
    """Return the atomic operations of a valid intervention in declaration order."""
    if isinstance(intervention, Composite):
        return intervention.operations
    if isinstance(intervention, HardIntervention | JointPolicy | Delete | Replace):
        return (intervention,)
    raise TypeError(f"Unsupported intervention type: {type(intervention).__name__}.")


def validate_intervention(
    graph: MechanismGraph, intervention: Intervention
) -> tuple[AtomicIntervention, ...]:
    """Check targets and disjoint intervention boundaries in a mechanism graph.

    This validates the specification only. It does not establish identification,
    positivity, acyclicity, or compatibility with a numerical backend.
    """
    operations = intervention_operations(intervention)
    _check_conflicts(operations)
    occupied: set[str] = set()
    for operation in operations:
        targets = _declared_targets(operation)
        if isinstance(operation, Delete | Replace):
            mechanism = graph.get_mechanism(operation.target)
            targets = frozenset(mechanism.outputs)
            if isinstance(operation, Delete) and targets != frozenset(operation.policy.variables):
                raise ValueError(
                    f"Deletion policy for {operation.target!r} must cover exactly its outputs "
                    f"{sorted(targets)}."
                )
        unknown = targets - graph.variable_set
        if unknown:
            raise ValueError(f"Intervention references unknown variables: {sorted(unknown)}.")
        overlap = targets & occupied
        if overlap:
            raise ValueError(f"Conflicting interventions on variables {sorted(overlap)}.")
        occupied.update(targets)
    return operations


@dataclass(frozen=True)
class CausalQuery:
    """Request a distribution or mean under an intervention.

    ``given`` names baseline conditioning variables; backends must verify that
    they are unaffected by the intervention before claiming baseline semantics.
    Their values are supplied to evaluation, separately from this specification.
    """

    outcomes: tuple[str, ...]
    intervention: Intervention
    kind: Literal["distribution", "expectation"] = "distribution"
    given: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        outcomes = _names(self.outcomes, label="Outcomes")
        given = _names(self.given, label="Conditioning variables")
        if not outcomes:
            raise ValueError("A causal query must contain at least one outcome.")
        if self.kind not in {"distribution", "expectation"}:
            raise ValueError("kind must be 'distribution' or 'expectation'.")
        if self.kind == "expectation" and len(outcomes) != 1:
            raise ValueError("An expectation query requires exactly one outcome.")
        if set(outcomes) & set(given):
            raise ValueError("Outcomes and conditioning variables must be disjoint.")
        intervention_operations(self.intervention)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "given", given)


@dataclass(frozen=True)
class EffectContrast:
    """The difference ``left - right`` for two matching requested quantities."""

    left: CausalQuery
    right: CausalQuery

    def __post_init__(self) -> None:
        if not isinstance(self.left, CausalQuery) or not isinstance(self.right, CausalQuery):
            raise TypeError("An effect contrast requires two CausalQuery objects.")
        if (self.left.outcomes, self.left.kind, self.left.given) != (
            self.right.outcomes, self.right.kind, self.right.given
        ):
            raise ValueError(
                "Contrast queries must have matching outcomes, kind, and conditioning."
            )


__all__ = [
    "AtomicIntervention", "CausalQuery", "Composite", "Delete", "EffectContrast",
    "HardIntervention", "Intervention", "JointPolicy", "Replace", "Scalar",
    "intervention_operations", "validate_intervention",
]
