"""Scoped parametric inference with continuous conditioning variables.

Joint multivariate least squares fits one affine kernel per mechanism. Independent
root moments and joint residual covariances are propagated analytically. This is a
linear additive model backend, not an evaluator for arbitrary identifying ASTs.
Gaussianity is additionally required for conditioning on baseline variables.
NumPy is imported only when the numerical backend is used.
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from causal_hypergraphs.estimation.dataset import DatasetError, _category
from causal_hypergraphs.graph import MechanismGraph
from causal_hypergraphs.identification import Assumption, Identified
from causal_hypergraphs.queries import (
    CausalQuery,
    Delete,
    EffectContrast,
    HardIntervention,
    JointPolicy,
    Replace,
    intervention_operations,
)


class ContinuousError(ValueError):
    """Base error for the scoped continuous backend."""

    code = "continuous_backend_error"


class ContinuousDataError(ContinuousError):
    """Data are missing, nonnumeric, nonfinite, or insufficient for fitting."""

    code = "invalid_continuous_data"


class RankDeficientFit(ContinuousError):
    """The declared linear predictor model is not estimable from these data."""

    code = "rank_deficient_fit"


class UnsupportedContinuousQuery(ContinuousError):
    """A query is outside this backend's causal or statistical model class."""

    code = "unsupported_continuous_query"


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as error:
        raise ImportError(
            "Continuous estimation requires the optional NumPy dependency."
        ) from error
    return numpy


def _number(value: object, label: str) -> float:
    if not isinstance(value, Real):
        raise ContinuousDataError(f"{label} must be a finite numeric value, not {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise ContinuousDataError(f"{label} must be finite.")
    return number


def _axes(names: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(names, str | set | frozenset | Mapping):
        raise TypeError(f"{label} must be an ordered sequence of names.")
    axes = tuple(names)
    if any(not isinstance(name, str) or not name.strip() for name in axes):
        raise ValueError(f"{label} must contain nonempty string identifiers.")
    if len(set(axes)) != len(axes):
        raise ValueError(f"{label} must not contain duplicates.")
    return axes


@dataclass(frozen=True, init=False)
class LinearGaussianKernel:
    """An affine joint-output kernel with a positive-semidefinite noise covariance.

    Coefficient rows follow ``outputs``; columns follow ``inputs``. Covariance
    axes follow ``outputs``. Singular residual covariance is explicitly supported.
    """

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    coefficients: tuple[tuple[float, ...], ...]
    intercept: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]

    def __init__(
        self,
        inputs: Sequence[str],
        outputs: Sequence[str],
        coefficients: Sequence[Sequence[float]],
        intercept: Sequence[float],
        covariance: Sequence[Sequence[float]],
    ) -> None:
        np = _numpy()
        inputs, outputs = _axes(inputs, "inputs"), _axes(outputs, "outputs")
        if not outputs:
            raise ValueError("A numerical kernel must have at least one output.")
        if set(inputs) & set(outputs):
            raise ValueError("A kernel's inputs and outputs must be disjoint.")
        matrix = np.asarray(coefficients, dtype=float)
        offset = np.asarray(intercept, dtype=float)
        noise = np.asarray(covariance, dtype=float)
        if matrix.shape != (len(outputs), len(inputs)):
            raise ValueError("Coefficient shape must be (number of outputs, number of inputs).")
        if offset.shape != (len(outputs),) or noise.shape != (len(outputs), len(outputs)):
            raise ValueError("Intercept and covariance must use the declared output axes.")
        if not all(np.isfinite(array).all() for array in (matrix, offset, noise)):
            raise ValueError("Kernel coefficients, intercept, and covariance must be finite.")
        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(noise))))
        if not np.allclose(noise, noise.T, rtol=0, atol=tolerance):
            raise ValueError("Kernel covariance must be symmetric.")
        noise = (noise + noise.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(noise)
        if float(eigenvalues.min()) < -tolerance:
            raise ValueError("Kernel covariance must be positive semidefinite.")
        if float(eigenvalues.min()) < 0:
            noise = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(
            self, "coefficients", tuple(tuple(float(v) for v in row) for row in matrix)
        )
        object.__setattr__(self, "intercept", tuple(float(v) for v in offset))
        object.__setattr__(self, "covariance", tuple(tuple(float(v) for v in row) for row in noise))


@dataclass(frozen=True)
class ContinuousCapabilities:
    model_profile: str = "fully-observed-acyclic-independent-mechanisms"
    quantities: tuple[str, ...] = ("expectation", "expectation-contrast")
    integration: str = "analytic linear moments"
    baseline_conditioning: str = "joint Gaussian, nonsingular baseline covariance"
    joint_policies: str = "numeric finite policies for unconditional expectations"


@dataclass(frozen=True)
class ContinuousEstimate:
    """A fitted causal mean or mean contrast, with distinct uncertainty sources.

    ``model_variance`` is outcome variation in the fitted intervention model, not
    a standard error. It is absent for contrasts because no cross-world coupling
    is specified. Bootstrap intervals describe parameter-sampling uncertainty.
    """

    value: float
    interval: tuple[float, float] | None
    standard_error: float | None
    model_variance: float | None
    assumptions: tuple[Assumption, ...]
    diagnostics: tuple[str, ...]
    n_rows: int
    n_units: int
    unit: str
    bootstrap: int = 0
    bootstrap_failures: int = 0
    successful_replicates: int = 0
    level: float = 0.95
    method: str = "linear-gaussian-moments"
    monte_carlo_error: float = 0.0
    numerical_error: float | None = None

    def summary(self) -> str:
        interval = "not requested or unavailable" if self.interval is None else str(self.interval)
        return (
            f"{self.method}: {self.value:g}; {self.level:g} interval {interval}\n"
            f"{self.n_rows} rows, {self.n_units} independent units ({self.unit}); "
            f"{self.bootstrap_failures}/{self.bootstrap} bootstrap fits failed\n"
            "Assumed, not verified: "
            + "; ".join(a.code for a in self.assumptions)
            + "\n"
            + "\n".join(self.diagnostics)
        )


@runtime_checkable
class ContinuousBackend(Protocol):
    capabilities: ContinuousCapabilities

    def estimate(
        self,
        query: CausalQuery | EffectContrast,
        *,
        given: Mapping[str, float] | None = None,
        replacements: Mapping[str, object] | None = None,
        bootstrap: int = 0,
        level: float = 0.95,
        seed: int = 0,
    ) -> ContinuousEstimate: ...


def _order(graph: MechanismGraph) -> tuple[str, ...]:
    dependencies = graph.mechanism_dependencies()
    incoming = {name: 0 for name in dependencies}
    for children in dependencies.values():
        for child in children:
            incoming[child] += 1
    ready = sorted(name for name, degree in incoming.items() if degree == 0)
    ordered = []
    while ready:
        name = ready.pop()
        ordered.append(name)
        for child in sorted(dependencies[name]):
            incoming[child] -= 1
            if incoming[child] == 0:
                ready.append(child)
    if len(ordered) != len(dependencies):
        raise UnsupportedContinuousQuery("Continuous fitting requires an acyclic mechanism graph.")
    return tuple(ordered)


def _kernel_copy(value: object) -> LinearGaussianKernel:
    if isinstance(value, LinearGaussianKernel):
        return value
    try:
        names = ("inputs", "outputs", "coefficients", "intercept", "covariance")
        return LinearGaussianKernel(**{name: getattr(value, name) for name in names})
    except AttributeError as error:
        raise UnsupportedContinuousQuery(
            "Replacement bindings require an affine kernel with named axes and covariance."
        ) from error


@dataclass(frozen=True)
class LinearGaussianFit:
    graph: MechanismGraph
    kernels: Mapping[str, LinearGaussianKernel]
    root_means: Mapping[str, float]
    root_variances: Mapping[str, float]
    ranges: Mapping[str, tuple[float, float]]
    discrete: tuple[str, ...]
    unit: str | None
    _records: tuple[Mapping[str, float], ...] = field(repr=False)
    _groups: tuple[tuple[int, ...], ...] = field(repr=False)
    _order: tuple[str, ...] = field(repr=False)
    capabilities: ContinuousCapabilities = field(default_factory=ContinuousCapabilities)

    def __post_init__(self) -> None:
        for name in ("kernels", "root_means", "root_variances", "ranges"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def n_rows(self) -> int:
        return len(self._records)

    @property
    def n_units(self) -> int:
        return len(self._groups)

    def _bindings(self, supplied: Mapping[str, object] | None) -> dict[str, LinearGaussianKernel]:
        return {name: _kernel_copy(kernel) for name, kernel in (supplied or {}).items()}

    def _moments(
        self, query: CausalQuery, replacements: Mapping[str, LinearGaussianKernel]
    ) -> tuple[Any, Any]:
        np = _numpy()
        names = self.graph.variables
        positions = {name: index for index, name in enumerate(names)}
        size = len(names)
        mean, covariance = np.zeros(size), np.zeros((size, size))
        fixed: dict[str, float] = {}
        policies: list[JointPolicy] = []
        replaced: dict[str, LinearGaussianKernel] = {}
        for operation in intervention_operations(query.intervention):
            if isinstance(operation, HardIntervention):
                fixed.update(
                    {name: _number(value, name) for name, value in operation.assignments.items()}
                )
            elif isinstance(operation, Replace):
                if operation.replacement not in replacements:
                    raise UnsupportedContinuousQuery(
                        f"Supply replacement kernel {operation.replacement!r}."
                    )
                kernel = replacements[operation.replacement]
                target = self.graph.get_mechanism(operation.target)
                if set(kernel.inputs) != set(target.inputs) or set(kernel.outputs) != set(
                    target.outputs
                ):
                    raise UnsupportedContinuousQuery(
                        "Replacement kernel incidence does not match its target."
                    )
                replaced[operation.target] = kernel
            else:
                policies.append(operation.policy if isinstance(operation, Delete) else operation)
        overridden = set(fixed) | {name for policy in policies for name in policy.variables}
        for name in self.graph.exogenous_variables - overridden:
            index = positions[name]
            mean[index] = self.root_means[name]
            covariance[index, index] = self.root_variances[name]
        for name, value in fixed.items():
            mean[positions[name]] = value
        for policy in policies:
            support = [point for point, mass in policy.probabilities.items() if mass > 0]
            values = np.asarray(
                [[_number(value, f"Policy {policy.name}") for value in point] for point in support]
            )
            weights = np.asarray([policy.probabilities[point] for point in support])
            policy_mean = weights @ values
            centered = values - policy_mean
            policy_covariance = (centered.T * weights) @ centered
            axes = [positions[name] for name in policy.variables]
            mean[axes] = policy_mean
            covariance[np.ix_(axes, axes)] = policy_covariance
        for name in self._order:
            if not self.graph.get_mechanism(name).outputs:
                continue
            kernel = replaced.get(name, self.kernels[name])
            retained = [
                index for index, variable in enumerate(kernel.outputs) if variable not in overridden
            ]
            if not retained:
                continue
            outputs = [positions[kernel.outputs[index]] for index in retained]
            inputs = [positions[variable] for variable in kernel.inputs]
            coefficients = np.asarray(kernel.coefficients)[retained, :]
            intercept = np.asarray(kernel.intercept)[retained]
            noise = np.asarray(kernel.covariance)[np.ix_(retained, retained)]
            mean[outputs] = intercept + coefficients @ mean[inputs]
            cross = coefficients @ covariance[inputs, :]
            variance = coefficients @ covariance[np.ix_(inputs, inputs)] @ coefficients.T + noise
            covariance[outputs, :] = cross
            covariance[:, outputs] = cross.T
            covariance[np.ix_(outputs, outputs)] = variance
        return mean, covariance

    def _point(
        self,
        query: CausalQuery | EffectContrast,
        given: Mapping[str, float],
        replacements: Mapping[str, LinearGaussianKernel],
    ) -> tuple[float, float | None]:
        if isinstance(query, EffectContrast):
            left, _ = self._point(query.left, given, replacements)
            right, _ = self._point(query.right, given, replacements)
            return left - right, None
        np = _numpy()
        mean, covariance = self._moments(query, replacements)
        index = self.graph.variables.index(query.outcomes[0])
        expectation = float(mean[index])
        variance = float(covariance[index, index])
        if query.given:
            axes = [self.graph.variables.index(name) for name in query.given]
            baseline_covariance = covariance[np.ix_(axes, axes)]
            if np.linalg.matrix_rank(baseline_covariance) < len(axes):
                raise RankDeficientFit(
                    "Baseline covariance is singular; no pseudoinverse extrapolation is used."
                )
            if np.linalg.cond(baseline_covariance) > 1e12:
                raise RankDeficientFit("Baseline covariance is numerically ill-conditioned.")
            cross = covariance[index, axes]
            shift = np.asarray([given[name] for name in query.given]) - mean[axes]
            expectation += float(cross @ np.linalg.solve(baseline_covariance, shift))
            variance -= float(cross @ np.linalg.solve(baseline_covariance, cross))
        if not math.isfinite(expectation) or not math.isfinite(variance):
            raise ContinuousError("Analytic propagation produced nonfinite moments.")
        tolerance = 1e-10 * max(1.0, abs(float(covariance[index, index])))
        if variance < -tolerance:
            raise ContinuousError("Analytic propagation produced a negative conditional variance.")
        return expectation, max(0.0, variance)

    def _validate(
        self, query: CausalQuery | EffectContrast, given: Mapping[str, float]
    ) -> tuple[Assumption, ...]:
        # Local import avoids coupling the symbolic core's imports to numerical backends.
        from causal_hypergraphs.inference import compile_query

        arms = (query.left, query.right) if isinstance(query, EffectContrast) else (query,)
        for arm in arms:
            if not isinstance(arm, CausalQuery) or arm.kind != "expectation":
                raise UnsupportedContinuousQuery(
                    "This backend supports expectations and expectation contrasts only."
                )
            if set(given) != set(arm.given):
                raise ValueError("given must bind exactly the query's baseline variables.")
            if arm.given and self.discrete:
                raise UnsupportedContinuousQuery(
                    "Baseline conditioning requires the fully Gaussian profile; "
                    "discrete roots were declared."
                )
            for operation in intervention_operations(arm.intervention):
                policy = operation.policy if isinstance(operation, Delete) else operation
                if isinstance(policy, JointPolicy) and arm.given:
                    if sum(mass > 0 for mass in policy.probabilities.values()) > 1:
                        raise UnsupportedContinuousQuery(
                            "Conditional queries with nondegenerate finite policy mixtures "
                            "are unsupported."
                        )
        compiled = compile_query(self.graph, query)
        if not isinstance(compiled.result, Identified):
            raise UnsupportedContinuousQuery(
                getattr(compiled.result, "reason", "Query is not identified.")
            )
        # Finite point-mass positivity labels from the symbolic frontend do not
        # certify continuous conditioning. State the continuous assumptions instead.
        assumptions = tuple(
            item
            for item in compiled.result.assumptions
            if item.code
            not in {"Downstream positivity", "Baseline positivity", "Backend positivity"}
        ) + (
            Assumption(
                "Linear additive kernels",
                "Each conditional mean is affine, with independent mechanism noise "
                "and constant joint residual covariance.",
            ),
            Assumption(
                "Independent roots",
                "Exogenous variables are mutually independent in the declared causal profile.",
            ),
            Assumption(
                "Continuous overlap",
                "Conditional kernels must be justified at intervention and conditioning values; "
                "marginal range diagnostics do not establish joint overlap.",
            ),
        )
        if arms[0].given:
            assumptions += (
                Assumption(
                    "Gaussian conditioning",
                    "All original root and mechanism noises are jointly Gaussian within each "
                    "independent block; the requested baseline covariance is nonsingular.",
                ),
            )
        return assumptions

    def _diagnostics(
        self, query: CausalQuery | EffectContrast, given: Mapping[str, float]
    ) -> tuple[str, ...]:
        checks = list(given.items())
        arms = (query.left, query.right) if isinstance(query, EffectContrast) else (query,)
        for arm in arms:
            for operation in intervention_operations(arm.intervention):
                if isinstance(operation, HardIntervention):
                    checks.extend(
                        (name, _number(value, name))
                        for name, value in operation.assignments.items()
                    )
                policy = operation.policy if isinstance(operation, Delete) else operation
                if isinstance(policy, JointPolicy):
                    for point, mass in policy.probabilities.items():
                        if mass > 0:
                            checks.extend(
                                (name, _number(value, name))
                                for name, value in zip(policy.variables, point, strict=True)
                            )
        diagnostics = [
            "Marginal observed-range checks do not establish joint overlap "
            "or validate causal assumptions."
        ]
        for name, value in sorted(set(checks)):
            low, high = self.ranges[name]
            if name in self.discrete and value not in {row[name] for row in self._records}:
                raise UnsupportedContinuousQuery(
                    f"{name!r} is a declared discrete root; value {value} "
                    "is absent from its observed support."
                )
            if value < low or value > high:
                diagnostics.append(
                    f"Extrapolation: {name}={value:g} lies outside observed range "
                    f"[{low:g}, {high:g}]."
                )
        if self.discrete:
            diagnostics.append(
                "Numeric discrete roots use their empirical moments; linear conditional means "
                "are assumed, not learned nonparametrically."
            )
        return tuple(diagnostics)

    def estimate(
        self,
        query: CausalQuery | EffectContrast,
        *,
        given: Mapping[str, float] | None = None,
        replacements: Mapping[str, object] | None = None,
        bootstrap: int = 0,
        level: float = 0.95,
        seed: int = 0,
    ) -> ContinuousEstimate:
        if not isinstance(bootstrap, int) or isinstance(bootstrap, bool) or bootstrap < 0:
            raise ValueError("bootstrap must be a nonnegative integer.")
        if not math.isfinite(level) or not 0 < level < 1:
            raise ValueError("level must be strictly between zero and one.")
        bound = {name: _number(value, name) for name, value in (given or {}).items()}
        assumptions = self._validate(query, bound)
        bindings = self._bindings(replacements)
        diagnostics = self._diagnostics(query, bound)
        value, variance = self._point(query, bound, bindings)
        samples: list[float] = []
        failures = 0
        if bootstrap:
            if self.n_units < 2:
                raise ContinuousDataError(
                    "Bootstrap uncertainty needs at least two independent sampling units."
                )
            rng = random.Random(seed)
            for _ in range(bootstrap):
                indices = [
                    index for group in rng.choices(self._groups, k=self.n_units) for index in group
                ]
                try:
                    replicate = fit_linear_gaussian(
                        self.graph,
                        (self._records[index] for index in indices),
                        discrete=self.discrete,
                    )
                    sample, _ = replicate._point(query, bound, bindings)
                    samples.append(sample)
                except (RankDeficientFit, ContinuousDataError):
                    failures += 1
        interval = None
        standard_error = None
        if bootstrap > 1 and failures == 0:
            np = _numpy()
            tails = ((1 - level) / 2, (1 + level) / 2)
            lower, upper = np.quantile(samples, tails)
            interval = (float(lower), float(upper))
            standard_error = float(np.std(samples, ddof=1))
        elif bootstrap:
            diagnostics += (
                "Bootstrap interval unavailable: at least two replicates and no failed refits "
                "are required; failures are not silently discarded.",
            )
        return ContinuousEstimate(
            value=value,
            interval=interval,
            standard_error=standard_error,
            model_variance=variance,
            assumptions=assumptions,
            diagnostics=diagnostics,
            n_rows=self.n_rows,
            n_units=self.n_units,
            unit=self.unit or "independent rows",
            bootstrap=bootstrap,
            bootstrap_failures=failures,
            successful_replicates=len(samples),
            level=level,
        )


def fit_linear_gaussian(
    graph: MechanismGraph,
    records: Iterable[Mapping[str, object]],
    *,
    unit: str | None = None,
    discrete: Iterable[str] = (),
) -> LinearGaussianFit:
    """Fit the supported fully observed, acyclic linear mechanism profile.

    Numeric discrete predictors must be explicitly named by ``discrete`` and must
    be exogenous roots. They support unconditional linear means, not a Gaussian
    distribution claim. Missing/nonfinite data and rank-deficient predictor designs
    are rejected. Declaring ``unit`` changes bootstrap resampling, not row weighting.
    """
    np = _numpy()
    if not isinstance(graph, MechanismGraph):
        raise TypeError(
            "Fit a causal MechanismGraph; generic incidence and ADMGs require other backends."
        )
    if graph.hidden_variables:
        raise UnsupportedContinuousQuery(
            "This backend requires every graph variable to be observed."
        )
    order = _order(graph)
    discrete_names = _axes(
        (discrete,) if isinstance(discrete, str) else tuple(discrete), "discrete"
    )
    if len(set(discrete_names)) != len(discrete_names):
        raise ValueError("discrete contains duplicate variables.")
    if not set(discrete_names) <= graph.exogenous_variables:
        raise UnsupportedContinuousQuery(
            "Declared discrete variables must be exogenous roots in this backend."
        )
    if unit is not None and (not isinstance(unit, str) or not unit.strip()):
        raise ValueError("unit must name a nonempty sampling-unit column.")
    rows: list[Mapping[str, float]] = []
    groups: dict[Hashable, list[int]] = {}
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise ContinuousDataError(f"Record {index} is not a mapping.")
        missing = graph.variable_set - set(row)
        if missing:
            raise ContinuousDataError(f"Record {index} is missing variables {sorted(missing)}.")
        values = {name: _number(row[name], f"Record {index}, {name}") for name in graph.variables}
        for mechanism in graph.mechanisms.values():
            for equality in mechanism.output_equalities:
                if equality and any(
                    not math.isclose(
                        values[equality[0]], values[name], rel_tol=1e-12, abs_tol=1e-12
                    )
                    for name in equality[1:]
                ):
                    raise ContinuousDataError(
                        f"Record {index} violates declared output equality {equality} "
                        f"in mechanism {mechanism.name!r}."
                    )
        rows.append(MappingProxyType(values))
        label = index if unit is None else row.get(unit)
        try:
            label = _category(label, f"Record {index}, sampling-unit identifier")
        except DatasetError as error:
            raise ContinuousDataError(str(error)) from error
        groups.setdefault(label, []).append(index)
    if len(rows) < 2:
        raise ContinuousDataError("At least two complete records are required.")
    positions = {name: index for index, name in enumerate(graph.variables)}
    matrix = np.asarray([[row[name] for name in graph.variables] for row in rows])
    kernels = {}
    for name, mechanism in graph.mechanisms.items():
        if not mechanism.outputs:
            continue
        predictors = matrix[:, [positions[v] for v in mechanism.inputs]]
        outputs = matrix[:, [positions[v] for v in mechanism.outputs]]
        n_parameters = len(mechanism.inputs) + 1
        if len(rows) <= n_parameters:
            raise ContinuousDataError(
                f"Mechanism {name!r} needs more than {n_parameters} rows "
                "to estimate residual covariance."
            )
        center = predictors.mean(axis=0)
        scale = predictors.std(axis=0)
        if np.any(scale == 0):
            raise RankDeficientFit(
                f"Mechanism {name!r} has a constant predictor; "
                "no extrapolated coefficient is fitted."
            )
        standardized = (predictors - center) / scale
        design = np.column_stack((np.ones(len(rows)), standardized))
        coefficients, _, rank, singular_values = np.linalg.lstsq(design, outputs, rcond=None)
        if rank != n_parameters or singular_values[0] / singular_values[-1] > 1e12:
            raise RankDeficientFit(
                f"Mechanism {name!r} has rank-deficient or ill-conditioned predictors."
            )
        residual = outputs - design @ coefficients
        residual_covariance = residual.T @ residual / (len(rows) - n_parameters)
        slopes = coefficients[1:, :].T / scale
        intercept = coefficients[0, :] - slopes @ center
        kernels[name] = LinearGaussianKernel(
            mechanism.inputs, mechanism.outputs, slopes, intercept, residual_covariance
        )
    root_means = {
        name: float(matrix[:, positions[name]].mean()) for name in graph.exogenous_variables
    }
    root_variances = {
        name: float(matrix[:, positions[name]].var(ddof=1)) for name in graph.exogenous_variables
    }
    ranges = {
        name: (float(matrix[:, index].min()), float(matrix[:, index].max()))
        for name, index in positions.items()
    }
    return LinearGaussianFit(
        graph=graph,
        kernels=kernels,
        root_means=root_means,
        root_variances=root_variances,
        ranges=ranges,
        discrete=discrete_names,
        unit=unit,
        _records=tuple(rows),
        _groups=tuple(tuple(indices) for indices in groups.values()),
        _order=order,
    )


def estimate_continuous(
    fitted: ContinuousBackend,
    query: CausalQuery | EffectContrast,
    *,
    given: Mapping[str, float] | None = None,
    replacements: Mapping[str, object] | None = None,
    bootstrap: int = 0,
    level: float = 0.95,
    seed: int = 0,
) -> ContinuousEstimate:
    """Delegate to a numerical backend, which validates its supported options."""
    return fitted.estimate(
        query,
        given=given,
        replacements=replacements,
        bootstrap=bootstrap,
        level=level,
        seed=seed,
    )


__all__ = [
    "ContinuousBackend",
    "ContinuousCapabilities",
    "ContinuousDataError",
    "ContinuousError",
    "ContinuousEstimate",
    "LinearGaussianFit",
    "LinearGaussianKernel",
    "RankDeficientFit",
    "UnsupportedContinuousQuery",
    "estimate_continuous",
    "fit_linear_gaussian",
]
