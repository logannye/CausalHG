"""Executable acyclic mechanism SCMs with explicit joint and cross-world semantics.

Identification does not require this module. These runtimes simulate a specified
structural model; simulated predictions are not identification from observed data.
NumPy is imported only when constructing a linear-Gaussian binding.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias

from causal_hypergraphs.graph import MechanismGraph
from causal_hypergraphs.queries import (
    Composite,
    Delete,
    HardIntervention,
    Intervention,
    JointPolicy,
    Replace,
    Scalar,
    validate_intervention,
)

Values: TypeAlias = Mapping[str, Scalar]
NoiseSampler: TypeAlias = Callable[[random.Random], object]
StructuralFunction: TypeAlias = Callable[[Values, object], Mapping[str, Scalar]]
Abduction: TypeAlias = Callable[[Values, Values], object]


def _axes(values: Sequence[str], label: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if isinstance(values, str | set | frozenset | Mapping):
        raise TypeError(f"{label} requires an ordered sequence of string identifiers.")
    names = tuple(values)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{label} requires nonempty string identifiers.")
    if len(names) != len(set(names)):
        raise ValueError(f"{label} contains duplicate identifiers.")
    if nonempty and not names:
        raise ValueError(f"{label} must not be empty.")
    return names


def _scalar(value: object) -> Scalar:
    if not isinstance(value, str | bool | int | float):
        raise TypeError("Model values must be strings, booleans, integers, or finite floats.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Model values must be finite.")
    return value


def _values(values: Mapping[str, Any], axes: tuple[str, ...], label: str) -> dict[str, Scalar]:
    if not isinstance(values, Mapping) or set(values) != set(axes):
        raise ValueError(f"{label} must have exactly the declared axes {axes}.")
    return {name: _scalar(values[name]) for name in axes}


def _incidence(inputs: Sequence[str], outputs: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    incoming = _axes(inputs, "Inputs")
    outgoing = _axes(outputs, "Outputs", nonempty=True)
    if set(incoming) & set(outgoing):
        raise ValueError("Mechanism inputs and outputs must be disjoint.")
    return incoming, outgoing


@dataclass(frozen=True)
class StructuralMechanism:
    """A joint structural function with one explicit shared noise value.

    ``function(inputs, noise)`` returns exactly the named output coordinates.
    ``abduct``, when supplied, must be a valid deterministic inverse for noise on
    the observed support. Consistency is checked, but uniqueness is the caller's
    model assumption; a many-to-one inverse requires posterior abduction instead.
    """

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    function: StructuralFunction
    noise_sampler: NoiseSampler
    abduct: Abduction | None = None

    def __post_init__(self) -> None:
        incoming, outgoing = _incidence(self.inputs, self.outputs)
        if not callable(self.function) or not callable(self.noise_sampler):
            raise TypeError("Structural function and noise sampler must be callable.")
        if self.abduct is not None and not callable(self.abduct):
            raise TypeError("abduct must be callable when supplied.")
        object.__setattr__(self, "inputs", incoming)
        object.__setattr__(self, "outputs", outgoing)

    @property
    def supports_counterfactual(self) -> bool:
        return True

    def sample_noise(self, rng: random.Random) -> object:
        return self.noise_sampler(rng)

    def evaluate(self, inputs: Values, noise: object) -> dict[str, Scalar]:
        checked = _values(inputs, self.inputs, "Mechanism input values")
        return _values(self.function(checked, noise), self.outputs, "Mechanism output values")


@dataclass(frozen=True, init=False)
class FiniteKernel:
    """A conditional table of joint output probability masses.

    Rows are indexed by ``inputs`` in their declared order; columns are tuples
    ordered by ``outputs``. Conditioned kernels require explicit ``input_domains``
    so that missing input rows are rejected. Within a row, omitted output tuples
    mean zero probability. A kernel alone supplies no cross-world coupling.
    """

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    table: Mapping[tuple[Scalar, ...], Mapping[tuple[Scalar, ...], float]]
    input_domains: Mapping[str, tuple[Scalar, ...]]
    abduct: None = field(default=None, init=False)

    def __init__(
        self,
        inputs: Sequence[str],
        outputs: Sequence[str],
        table: Mapping[tuple[Any, ...], Mapping[tuple[Any, ...], float]],
        *,
        input_domains: Mapping[str, Sequence[Scalar]] | None = None,
    ) -> None:
        incoming, outgoing = _incidence(inputs, outputs)
        supplied_domains = {} if input_domains is None else input_domains
        if set(supplied_domains) != set(incoming):
            raise ValueError("Finite kernels require an explicit domain for every input axis.")
        domains = {
            name: tuple(_scalar(value) for value in supplied_domains[name]) for name in incoming
        }
        if any(not values or len(set(values)) != len(values) for values in domains.values()):
            raise ValueError("Input domains must be nonempty and contain no duplicate values.")
        expected = set(itertools.product(*(domains[name] for name in incoming)))
        if set(table) != expected:
            raise ValueError("Finite kernel table must contain exactly all input-domain rows.")
        rows: dict[tuple[Scalar, ...], Mapping[tuple[Scalar, ...], float]] = {}
        for key, masses in table.items():
            if not isinstance(key, tuple) or len(key) != len(incoming):
                raise ValueError("Finite kernel row keys must match the input axes.")
            policy = JointPolicy(outgoing, masses)
            rows[tuple(_scalar(value) for value in key)] = policy.probabilities
        object.__setattr__(self, "inputs", incoming)
        object.__setattr__(self, "outputs", outgoing)
        object.__setattr__(self, "table", MappingProxyType(rows))
        object.__setattr__(self, "input_domains", MappingProxyType(domains))

    @property
    def supports_counterfactual(self) -> bool:
        return False

    def sample_noise(self, rng: random.Random) -> object:
        return rng.random()

    def evaluate(self, inputs: Values, noise: object) -> dict[str, Scalar]:
        checked = _values(inputs, self.inputs, "Kernel input values")
        key = tuple(checked[name] for name in self.inputs)
        if key not in self.table:
            raise ValueError(f"No finite kernel row for input configuration {key!r}.")
        if not isinstance(noise, int | float) or not math.isfinite(noise) or not 0 <= noise < 1:
            raise ValueError("Finite kernel sampling noise must lie in [0, 1).")
        values = _draw_tuple(self.table[key], float(noise))
        return dict(zip(self.outputs, values, strict=True))

    def as_structural(self) -> StructuralMechanism:
        """Explicitly declare the table's ordered inverse-CDF cross-world coupling.

        Row insertion order determines the intervals assigned to output tuples.
        The returned SCM assumption is stronger than the kernel law: the same
        uniform noise is reused across input configurations and intervention worlds.
        This declaration is user-selected, not learned from the conditional table.
        """
        return StructuralMechanism(self.inputs, self.outputs, self.evaluate, self.sample_noise)


@dataclass(frozen=True, init=False)
class LinearGaussianMechanism:
    """Joint affine outputs with Gaussian residuals, including singular covariance.

    Rows of ``coefficients`` and ``intercept`` follow ``outputs``; coefficient
    columns follow ``inputs``. Residual vectors are the structural noise, so
    abduction is ``observed outputs - affine mean`` even for singular covariance.
    NumPy is required only to validate/factor the covariance at construction.
    """

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    coefficients: tuple[tuple[float, ...], ...]
    intercept: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    _transform: tuple[tuple[float, ...], ...] = field(repr=False)
    _null_vectors: tuple[tuple[float, ...], ...] = field(repr=False)

    def __init__(
        self,
        inputs: Sequence[str],
        outputs: Sequence[str],
        coefficients: Sequence[Sequence[float]],
        intercept: Sequence[float],
        covariance: Sequence[Sequence[float]],
    ) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "LinearGaussianMechanism requires the 'numerical' extra: "
                "pip install 'causal-hypergraphs[numerical]'."
            ) from exc
        incoming, outgoing = _incidence(inputs, outputs)
        beta = np.asarray(coefficients, dtype=float)
        offset = np.asarray(intercept, dtype=float)
        sigma = np.asarray(covariance, dtype=float)
        dimension = len(outgoing)
        if beta.shape != (dimension, len(incoming)):
            raise ValueError("Coefficient dimensions must be outputs by inputs.")
        if offset.shape != (dimension,) or sigma.shape != (dimension, dimension):
            raise ValueError("Intercept and covariance dimensions must match the output axes.")
        if not all(np.all(np.isfinite(value)) for value in (beta, offset, sigma)):
            raise ValueError("Gaussian parameters must be finite.")
        tolerance = 1e-12 * max(1.0, float(np.max(np.abs(sigma))))
        if not np.allclose(sigma, sigma.T, atol=tolerance, rtol=0):
            raise ValueError("Covariance must be symmetric.")
        eigenvalues, eigenvectors = np.linalg.eigh((sigma + sigma.T) / 2)
        if float(np.min(eigenvalues)) < -tolerance:
            raise ValueError("Covariance must be positive semidefinite.")
        eigenvalues = np.maximum(eigenvalues, 0)
        transform = eigenvectors * np.sqrt(eigenvalues)
        null_vectors = eigenvectors[:, eigenvalues == 0].T
        object.__setattr__(self, "inputs", incoming)
        object.__setattr__(self, "outputs", outgoing)
        object.__setattr__(self, "coefficients", tuple(tuple(map(float, row)) for row in beta))
        object.__setattr__(self, "intercept", tuple(map(float, offset)))
        object.__setattr__(self, "covariance", tuple(tuple(map(float, row)) for row in sigma))
        object.__setattr__(self, "_transform", tuple(tuple(map(float, row)) for row in transform))
        object.__setattr__(
            self, "_null_vectors", tuple(tuple(map(float, row)) for row in null_vectors)
        )

    @property
    def supports_counterfactual(self) -> bool:
        return True

    def sample_noise(self, rng: random.Random) -> object:
        independent = tuple(rng.gauss(0, 1) for _ in self.outputs)
        return tuple(
            math.fsum(a * z for a, z in zip(row, independent, strict=True))
            for row in self._transform
        )

    def _mean(self, inputs: Values) -> tuple[float, ...]:
        checked = _values(inputs, self.inputs, "Gaussian input values")
        if any(not isinstance(value, int | float) for value in checked.values()):
            raise TypeError("Gaussian mechanisms require numeric input values.")
        x = tuple(float(checked[name]) for name in self.inputs)
        return tuple(
            intercept + math.fsum(a * value for a, value in zip(row, x, strict=True))
            for intercept, row in zip(self.intercept, self.coefficients, strict=True)
        )

    def _residual(self, noise: object) -> tuple[float, ...]:
        if not isinstance(noise, tuple | list) or len(noise) != len(self.outputs):
            raise ValueError("Gaussian noise must be a residual vector matching the output axes.")
        residual = tuple(float(_scalar(value)) for value in noise)
        if any(not math.isfinite(value) for value in residual):
            raise ValueError("Gaussian residual values must be finite.")
        tolerance = 1e-9 * max(1.0, *(abs(value) for value in residual))
        if any(
            abs(math.fsum(a * value for a, value in zip(row, residual, strict=True))) > tolerance
            for row in self._null_vectors
        ):
            raise ValueError("Residual lies outside the support of the singular covariance.")
        return residual

    def evaluate(self, inputs: Values, noise: object) -> dict[str, Scalar]:
        mean = self._mean(inputs)
        residual = self._residual(noise)
        return {
            name: _scalar(mu + error)
            for name, mu, error in zip(self.outputs, mean, residual, strict=True)
        }

    def abduct(self, inputs: Values, outputs: Values) -> object:
        values = _values(outputs, self.outputs, "Gaussian observed outputs")
        if any(not isinstance(value, int | float) for value in values.values()):
            raise TypeError("Gaussian mechanisms require numeric output values.")
        return self._residual(
            tuple(
                float(values[name]) - mean
                for name, mean in zip(self.outputs, self._mean(inputs), strict=True)
            )
        )


MechanismBinding: TypeAlias = StructuralMechanism | FiniteKernel | LinearGaussianMechanism


@dataclass(frozen=True)
class NoiseRecord:
    """Replayable draws and intervention provenance from one SCM execution.

    Mappings are immutable snapshots. Custom structural noise objects are copied;
    callers should use immutable payloads when records are shared across callers.
    Policy records store selected joint values, not independent coordinate draws.
    """

    exogenous: Mapping[str, Scalar]
    mechanisms: Mapping[str, object]
    policies: Mapping[str, Mapping[str, Scalar]] = field(default_factory=dict)
    intervention: Intervention = field(default_factory=lambda: Composite(()))

    def __post_init__(self) -> None:
        object.__setattr__(self, "exogenous", MappingProxyType(dict(self.exogenous)))
        object.__setattr__(
            self, "mechanisms", MappingProxyType(copy.deepcopy(dict(self.mechanisms)))
        )
        object.__setattr__(
            self,
            "policies",
            MappingProxyType(
                {key: MappingProxyType(dict(values)) for key, values in self.policies.items()}
            ),
        )


@dataclass(frozen=True)
class _Plan:
    intervention: Intervention
    fixed: Mapping[str, Scalar]
    policies: tuple[tuple[str, JointPolicy], ...]
    deleted: frozenset[str]
    bindings: Mapping[str, MechanismBinding]
    targets: frozenset[str]


def _draw_tuple(masses: Mapping[tuple[Scalar, ...], float], uniform: float) -> tuple[Scalar, ...]:
    cumulative = 0.0
    last_positive: tuple[Scalar, ...] | None = None
    for values, mass in masses.items():
        if mass == 0:
            continue
        last_positive = values
        cumulative += mass
        if uniform < cumulative:
            return values
    # Validated rows sum to one within rounding tolerance. Falling beyond the last
    # endpoint only reflects floating-point summation, never a missing table row.
    if last_positive is None:
        raise ValueError("Cannot sample an empty probability distribution.")
    return last_positive


@dataclass(frozen=True, init=False)
class HypergraphSCM:
    """Bind an acyclic, single-producer mechanism graph to executable mechanisms.

    Distinct root samplers and mechanism noise are interpreted as independent.
    ``sample`` returns rows in graph variable order. All randomness flows through
    one explicit ``random.Random`` instance, supplied directly or created by seed.
    """

    graph: MechanismGraph
    mechanisms: Mapping[str, MechanismBinding]
    exogenous: Mapping[str, NoiseSampler]
    _order: tuple[str, ...] = field(repr=False)

    def __init__(
        self,
        graph: MechanismGraph,
        mechanisms: Mapping[str, MechanismBinding],
        exogenous: Mapping[str, NoiseSampler] | None = None,
    ) -> None:
        if not isinstance(graph, MechanismGraph):
            raise TypeError("HypergraphSCM requires a MechanismGraph.")
        if not graph.is_mechanism_acyclic():
            raise ValueError("Simulation requires an acyclic mechanism graph.")
        if set(mechanisms) != set(graph.mechanisms):
            raise ValueError("Provide exactly one executable binding for every graph mechanism.")
        roots = {} if exogenous is None else dict(exogenous)
        if set(roots) != graph.exogenous_variables or any(not callable(f) for f in roots.values()):
            raise ValueError("Provide exactly one callable sampler for every exogenous variable.")
        for name, binding in mechanisms.items():
            self._check_binding(graph, name, binding)
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "mechanisms", MappingProxyType(dict(mechanisms)))
        object.__setattr__(self, "exogenous", MappingProxyType(roots))
        object.__setattr__(self, "_order", self._topological_order())

    @staticmethod
    def _check_binding(graph: MechanismGraph, name: str, binding: MechanismBinding) -> None:
        if not isinstance(binding, StructuralMechanism | FiniteKernel | LinearGaussianMechanism):
            raise TypeError("Bindings must be structural mechanisms, finite kernels, or Gaussians.")
        mechanism = graph.get_mechanism(name)
        if set(binding.inputs) != set(mechanism.inputs) or set(binding.outputs) != set(
            mechanism.outputs
        ):
            raise ValueError(
                f"Binding {name!r} must have exactly its graph input/output incidence."
            )

    def _topological_order(self) -> tuple[str, ...]:
        edges = self.graph.mechanism_dependencies()
        degrees = dict.fromkeys(edges, 0)
        for children in edges.values():
            for child in children:
                degrees[child] += 1
        queue = sorted(name for name, degree in degrees.items() if degree == 0)
        ordered = []
        while queue:
            name = queue.pop(0)
            ordered.append(name)
            for child in sorted(edges[name]):
                degrees[child] -= 1
                if degrees[child] == 0:
                    queue.append(child)
        return tuple(ordered)

    def _plan(
        self,
        intervention: Intervention | None,
        replacements: Mapping[str, MechanismBinding] | None,
    ) -> _Plan:
        resolved = Composite(()) if intervention is None else intervention
        operations = validate_intervention(self.graph, resolved)
        supplied = {} if replacements is None else replacements
        fixed: dict[str, Scalar] = {}
        policies: list[tuple[str, JointPolicy]] = []
        deleted: set[str] = set()
        bindings = dict(self.mechanisms)
        targets: set[str] = set()
        used_replacements: set[str] = set()
        for index, operation in enumerate(operations):
            if isinstance(operation, HardIntervention):
                fixed.update(operation.assignments)
                targets.update(operation.assignments)
            elif isinstance(operation, JointPolicy):
                policies.append((str(index), operation))
                targets.update(operation.variables)
            elif isinstance(operation, Delete):
                deleted.add(operation.target)
                policies.append((str(index), operation.policy))
                targets.update(operation.policy.variables)
            elif isinstance(operation, Replace):
                if operation.replacement not in supplied:
                    raise ValueError(f"Missing executable replacement {operation.replacement!r}.")
                binding = supplied[operation.replacement]
                self._check_binding(self.graph, operation.target, binding)
                bindings[operation.target] = binding
                used_replacements.add(operation.replacement)
        if set(supplied) != used_replacements:
            raise ValueError(
                "Replacement bindings must match exactly the requested replacement names."
            )
        return _Plan(
            resolved, fixed, tuple(policies), frozenset(deleted), bindings, frozenset(targets)
        )

    @staticmethod
    def _rng(seed: int | None, rng: random.Random | None) -> random.Random:
        if seed is not None and rng is not None:
            raise ValueError("Supply either seed or rng, not both.")
        if rng is not None and not isinstance(rng, random.Random):
            raise TypeError("rng must be an instance of random.Random.")
        return random.Random(seed) if rng is None else rng

    def sample(
        self,
        n: int = 1,
        *,
        seed: int | None = None,
        rng: random.Random | None = None,
        intervention: Intervention | None = None,
        replacements: Mapping[str, MechanismBinding] | None = None,
    ) -> list[dict[str, Scalar]]:
        """Draw independent rows under the requested intervention."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("n must be a nonnegative integer.")
        generator = self._rng(seed, rng)
        plan = self._plan(intervention, replacements)
        return [self._sample_with_noise(generator, plan)[0] for _ in range(n)]

    def sample_with_noise(
        self,
        *,
        seed: int | None = None,
        rng: random.Random | None = None,
        intervention: Intervention | None = None,
        replacements: Mapping[str, MechanismBinding] | None = None,
    ) -> tuple[dict[str, Scalar], NoiseRecord]:
        """Draw one row and return all draws needed to reproduce that execution."""
        return self._sample_with_noise(self._rng(seed, rng), self._plan(intervention, replacements))

    def _sample_with_noise(
        self, rng: random.Random, plan: _Plan
    ) -> tuple[dict[str, Scalar], NoiseRecord]:
        roots = {
            name: _scalar(self.exogenous[name](rng))
            for name in sorted(self.exogenous)
            if name not in plan.targets
        }
        noise = {
            name: plan.bindings[name].sample_noise(rng)
            for name in self._order
            if name not in plan.deleted and set(plan.bindings[name].outputs) - plan.targets
        }
        policies = {
            key: dict(
                zip(policy.variables, _draw_tuple(policy.probabilities, rng.random()), strict=True)
            )
            for key, policy in plan.policies
        }
        record = NoiseRecord(roots, noise, policies, plan.intervention)
        return self._evaluate(record, plan), record

    def evaluate_with_noise(
        self,
        noise: NoiseRecord,
        *,
        intervention: Intervention | None = None,
        replacements: Mapping[str, MechanismBinding] | None = None,
    ) -> dict[str, Scalar]:
        """Replay complete supplied draws without sampling or advancing any RNG.

        By default the recorded intervention is replayed. Replacement executions
        additionally require their original executable replacement bindings.
        Changing the recorded intervention is a counterfactual request and obeys
        the same structural-coupling restrictions as :meth:`counterfactual`.
        """
        if intervention is not None and intervention != noise.intervention:
            if replacements:
                raise ValueError("Changed-intervention replay cannot bind replacement mechanisms.")
            return self.counterfactual(intervention, noise=noise)
        resolved = noise.intervention if intervention is None else intervention
        return self._evaluate(noise, self._plan(resolved, replacements))

    def _evaluate(self, noise: NoiseRecord, plan: _Plan) -> dict[str, Scalar]:
        values = dict(plan.fixed)
        for key, policy in plan.policies:
            if key not in noise.policies:
                raise ValueError(f"Missing selected joint policy values for operation {key}.")
            selected = _values(noise.policies[key], policy.variables, "Selected policy values")
            if policy.probability(selected) <= 0:
                raise ValueError("Selected policy values must lie in the policy's support.")
            values.update(selected)
        for name in sorted(self.exogenous):
            if name not in values:
                if name not in noise.exogenous:
                    raise ValueError(f"Missing exogenous noise/value for {name!r}.")
                values[name] = _scalar(noise.exogenous[name])
        for name in self._order:
            if name in plan.deleted:
                continue
            binding = plan.bindings[name]
            retained = set(binding.outputs) - plan.targets
            if not retained:
                continue
            if name not in noise.mechanisms:
                raise ValueError(f"Missing mechanism noise for {name!r}.")
            inputs = {variable: values[variable] for variable in binding.inputs}
            # User structural functions may consume a mutable noise payload. Each
            # evaluation gets its own copy so replay cannot mutate the saved draw.
            outputs = binding.evaluate(inputs, copy.deepcopy(noise.mechanisms[name]))
            values.update({variable: outputs[variable] for variable in retained})
        return {name: values[name] for name in self.graph.variables}

    def abduct(self, observation: Values) -> NoiseRecord:
        """Recover noise from a complete observed state using declared valid inverses."""
        values = _values(observation, tuple(self.graph.variables), "Complete observed state")
        noise: dict[str, object] = {}
        for name in self._order:
            binding = self.mechanisms[name]
            if not binding.supports_counterfactual or binding.abduct is None:
                raise ValueError(
                    f"Mechanism {name!r} has no valid deterministic abduction binding."
                )
            inputs = {variable: values[variable] for variable in binding.inputs}
            outputs = {variable: values[variable] for variable in binding.outputs}
            noise[name] = binding.abduct(inputs, outputs)
        record = NoiseRecord({name: values[name] for name in self.exogenous}, noise)
        replay = self.evaluate_with_noise(record)
        for name, value in values.items():
            reproduced = replay[name]
            if isinstance(value, int | float) and isinstance(reproduced, int | float):
                consistent = math.isclose(value, reproduced, rel_tol=1e-9, abs_tol=1e-9)
            else:
                consistent = value == reproduced
            if not consistent:
                raise ValueError(
                    f"Abducted noise does not reproduce the observed value of {name!r}."
                )
        return record

    def counterfactual(
        self,
        intervention: Intervention,
        *,
        noise: NoiseRecord | None = None,
        observation: Values | None = None,
    ) -> dict[str, Scalar]:
        """Evaluate a hard-intervention world using the same factual exogenous noise.

        Supply either an observational noise record or a complete observed state
        with valid deterministic abduction bindings. Stochastic policies, deletion,
        and replacement require cross-world coupling beyond this stable API.
        """
        operations = validate_intervention(self.graph, intervention)
        if any(not isinstance(operation, HardIntervention) for operation in operations):
            raise ValueError(
                "Stochastic or replacement counterfactuals need explicit cross-world coupling."
            )
        if (noise is None) == (observation is None):
            raise ValueError("Supply exactly one of factual noise or a complete observation.")
        if any(not binding.supports_counterfactual for binding in self.mechanisms.values()):
            raise ValueError(
                "Kernel-only models do not specify cross-world coupling; explicitly bind a "
                "structural mechanism (or declare FiniteKernel.as_structural())."
            )
        factual = self.abduct(observation) if observation is not None else noise
        assert factual is not None
        if any(
            not isinstance(operation, HardIntervention) or operation.assignments
            for operation in validate_intervention(self.graph, factual.intervention)
        ):
            raise ValueError("Counterfactual noise must come from an observational execution.")
        return self._evaluate(factual, self._plan(intervention, None))


__all__ = [
    "Abduction",
    "FiniteKernel",
    "HypergraphSCM",
    "LinearGaussianMechanism",
    "MechanismBinding",
    "NoiseRecord",
    "NoiseSampler",
    "StructuralMechanism",
    "Values",
]
