"""Evaluate an identified estimand against data, and discharge its certificates.

`identify` returns an estimand together with a list of assumptions it could not check
from incidence alone -- chiefly positivity, which is a property of the distribution and
not of the graph. Until now those assumptions were prose in a list. Against a dataset
they become checkable, and checking them is the substance of this module.

Design commitments
------------------
The certificate population is *computed, not enumerated*. The set of conditioning strata
an estimand requires is not a hand-written list of kernels; it is whatever the evaluator
touches while evaluating the expression over its scope. `UndefinedEstimand` is the
evaluator's own refusal, so the discharge is exactly as complete as the semantics, and an
estimand that grows a new quotient is covered without anyone remembering to update a list.

Failure is named, not absorbed. A conditioning cell with no observations yields a
`SupportFailure` identifying the kernel and the stratum. The corresponding points are
absent from `values` rather than present as `nan`, so an undefined estimand cannot be
mistaken for a computed zero downstream.

The interval resamples *units*, not rows, and the estimate reports which. A replicate may
push a thin stratum to empty; those are counted rather than discarded silently, because a
high replicate-failure rate is itself the finding.
"""
from __future__ import annotations

import itertools
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from causal_hypergraphs.identification import Assumption, IdentificationResult, Identified
from causal_hypergraphs.semantics import (
    Assignment,
    DiscreteModel,
    MissingKernel,
    SemanticsError,
    UndefinedEstimand,
    evaluate,
)

from .dataset import Dataset, DatasetError, Point


class EstimationError(Exception):
    """Base class for failures of the data-facing path."""


class NotIdentified(EstimationError):
    """There is no estimand to evaluate, because identification did not produce one."""


class UnsupportedEstimand(EstimationError):
    """The estimand references something these data or these kernels do not supply."""


# Assumption codes this module actually discharges. Everything else is reported as *not*
# checked, which is the safe default: a certificate wrongly listed here would be announced
# as verified without being verified, whereas one wrongly omitted is merely conservative.
# `test_discharged_codes_are_codes_the_compiler_emits` pins these against the compiler.
DISCHARGEABLE_CODES = frozenset({"Target positivity", "Downstream positivity"})


@dataclass(frozen=True)
class SupportFailure:
    """One conditioning stratum at which the estimand is undefined in these data."""

    kernel: str
    stratum: Mapping[str, Any]
    points: int
    """How many points of the estimand's scope were lost to this stratum."""

    def __str__(self) -> str:
        cells = ", ".join(f"{name}={value!r}" for name, value in sorted(self.stratum.items()))
        return f"{self.kernel} undefined at {cells} ({self.points} point(s) unreachable)"


@dataclass(frozen=True)
class SupportReport:
    """The outcome of discharging the estimand's positivity certificates."""

    checked: tuple[str, ...]
    not_checked: tuple[Assumption, ...]
    failures: tuple[SupportFailure, ...]
    points_total: int
    points_undefined: int
    min_stratum_count: int | None
    """Rows in the sparsest conditioning cell the estimand used, or None if it used none.

    `None` and `0` mean opposite things -- no conditioning at all versus an empty cell --
    so they must not share a representation.
    """
    thinnest_stratum: Mapping[str, Any] | None

    @property
    def holds(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        if self.holds:
            thinnest = (
                "the estimand conditions on nothing, so no stratum can be empty"
                if self.min_stratum_count is None
                else f"thinnest conditioning stratum has {self.min_stratum_count} row(s)"
            )
            return (
                f"all {len(self.checked)} positivity certificate(s) hold over "
                f"{self.points_total} point(s); {thinnest}"
            )
        return (
            f"{self.points_undefined} of {self.points_total} point(s) undefined across "
            f"{len(self.failures)} empty stratum/strata"
        )


@dataclass(frozen=True)
class Estimate:
    """A post-intervention law estimated from data, with its certificates discharged."""

    variables: tuple[str, ...]
    values: Mapping[Point, float]
    support: SupportReport
    assumptions: tuple[Assumption, ...]
    theorem: str
    unit: str
    n_rows: int
    n_units: int
    interval: Mapping[Point, tuple[float, float] | None] = field(default_factory=dict)
    bootstrap: int = 0
    level: float = 0.95
    replicate_failures: int = 0

    def summary(self) -> str:
        """A human-readable report that leads with what the data cannot check.

        Ordering is deliberate. Once this path returns numbers the numbers get believed,
        and the assumption most likely to be false -- C2, no unmeasured confounding across
        mechanisms -- is not visible anywhere in the output unless it is put there. It
        comes before the discharged certificates, not after them.
        """
        lines = [
            f"Estimate of P({','.join(self.variables)} | intervention) via {self.theorem}",
            f"  {self.n_rows} row(s) in {self.n_units} unit(s); unit = {self.unit}",
        ]
        if self.bootstrap:
            lines.append(
                f"  {int(self.level * 100)}% interval from {self.bootstrap} unit-bootstrap "
                f"replicate(s)"
                + (
                    f"; {self.replicate_failures} replicate-point(s) undefined"
                    if self.replicate_failures
                    else ""
                )
            )
        lines.append("")
        lines.append("  Not checked here -- asserted by the model, not testable from these data:")
        for assumption in self.support.not_checked:
            lines.append(f"    {assumption.code}: {assumption.description}")
        lines.append("")
        lines.append("  Checked against the data:")
        if self.support.checked:
            for code in self.support.checked:
                lines.append(f"    {code}: {'PASS' if self.support.holds else 'FAIL'}")
            lines.append(f"    {self.support.summary()}")
        else:
            lines.append("    (this estimand carries no dischargeable certificate)")
        for failure in self.support.failures:
            lines.append(f"    ! {failure}")
        return "\n".join(lines)


class _AuditingModel:
    """A `Model` that records which cells of the data an estimand actually reads.

    Wrapping rather than instrumenting `DiscreteModel` keeps the audit out of the
    evaluation path that the conformance sweep verifies, and keeps `evaluate` free of any
    knowledge that auditing exists.

    Both cells of a conditional are recorded, not just the conditioning one. `P(y | z)` is
    estimated as a ratio of two counts, and the *numerator* cell is the smaller of the
    two, so it is what bounds the precision. Recording only `z` would report a conditional
    resting on three observations as though it rested on the five hundred that share its
    conditioning stratum.
    """

    def __init__(self, inner: DiscreteModel) -> None:
        self._inner = inner
        self.strata: set[tuple[tuple[str, ...], Point]] = set()

    @property
    def domains(self) -> Mapping[str, tuple[Any, ...]]:
        return self._inner.domains

    def _record(self, names: Sequence[str], assignment: Assignment) -> None:
        ordered = tuple(sorted(names))
        self.strata.add((ordered, tuple(assignment[name] for name in ordered)))

    def conditional(
        self, variables: Sequence[str], given: Sequence[str], assignment: Assignment
    ) -> float:
        if given:
            self._record(given, assignment)
            self._record(tuple(variables) + tuple(given), assignment)
        return self._inner.conditional(variables, given, assignment)

    def fallback(
        self, mechanism: str, variables: Sequence[str], assignment: Assignment
    ) -> float:
        return self._inner.fallback(mechanism, variables, assignment)

    def replacement(
        self,
        mechanism: str,
        variables: Sequence[str],
        given: Sequence[str],
        assignment: Assignment,
    ) -> float:
        return self._inner.replacement(mechanism, variables, given, assignment)


def _as_identified(result: IdentificationResult | Identified) -> Identified:
    if isinstance(result, Identified):
        return result
    reason = getattr(result, "reason", None) or "identification did not return an estimand"
    missing = getattr(result, "missing_variables", ())
    suggestions = getattr(result, "suggestions", ())
    detail = f"{type(result).__name__}: {reason}"
    if missing:
        detail += f" Missing: {list(missing)}."
    if suggestions:
        detail += f" Suggested: {suggestions[0]}"
    raise NotIdentified(
        f"Cannot estimate: there is no identified estimand. {detail}"
    )


def _evaluate_over_scope(
    identified: Identified,
    model: _AuditingModel,
    variables: tuple[str, ...],
) -> tuple[dict[Point, float], dict[tuple[str, Point], SupportFailure]]:
    """Evaluate at every point of the estimand's scope, collecting undefined strata."""
    values: dict[Point, float] = {}
    failures: dict[tuple[str, Point], SupportFailure] = {}
    domains = [model.domains[name] for name in variables]

    for combination in itertools.product(*domains):
        assignment = dict(zip(variables, combination, strict=True))
        try:
            values[combination] = evaluate(identified.expression, model, assignment)
        except UndefinedEstimand as undefined:
            stratum = undefined.stratum
            key = (
                undefined.kernel or "<unknown kernel>",
                tuple(stratum[name] for name in sorted(stratum)),
            )
            previous = failures.get(key)
            failures[key] = SupportFailure(
                kernel=undefined.kernel or "<unknown kernel>",
                stratum=stratum,
                points=(previous.points if previous else 0) + 1,
            )
        except MissingKernel as missing:
            raise UnsupportedEstimand(str(missing)) from missing

    return values, failures


def _thinnest_stratum(
    data: Dataset, strata: set[tuple[tuple[str, ...], Point]]
) -> tuple[int | None, Mapping[str, Any] | None]:
    """The smallest row count among the conditioning cells the estimand actually used.

    An estimand can be perfectly well defined and still be resting on a handful of rows.
    Reporting only pass/fail on positivity would hide that, so the thinnest cell is part
    of the report: it is the number that says how much the quotient can be trusted.
    """
    smallest: int | None = None
    where: Mapping[str, Any] | None = None
    cache: dict[tuple[str, ...], dict[Point, int]] = {}
    for names, point in sorted(strata):
        if names not in cache:
            cache[names] = data.counts(names)
        count = cache[names].get(point, 0)
        if smallest is None or count < smallest:
            smallest = count
            where = dict(zip(names, point, strict=True))
    return smallest, where


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile of an already-sorted sample."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def estimate(
    result: IdentificationResult | Identified,
    data: Dataset,
    *,
    fallbacks: Mapping[str, Mapping[Point, float]] | None = None,
    replacements: Mapping[str, Mapping[tuple[Point, Point], float]] | None = None,
    bootstrap: int = 0,
    level: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Evaluate an identified estimand against `data` and discharge its certificates.

    Parameters
    ----------
    result:
        The output of `identify`. A refusal raises `NotIdentified` carrying the reason,
        rather than being coerced into a number.
    fallbacks:
        Joint deletion policies ``P0^m(out(m))`` keyed by mechanism, as
        ``{mechanism: {output_values: probability}}``. These are *policy*, not data: they
        say what the intervention installs, so they are supplied, never estimated.
    replacements:
        Replacement kernels keyed by the replacement's name, in the form
        `DiscreteModel.replacements` expects.
    bootstrap:
        Number of unit-resampled replicates for the interval. Zero (the default) returns
        a point estimate with no interval rather than a fake one.

    Returns
    -------
    An `Estimate` whose `values` cover exactly the points where the estimand is defined in
    these data. Points lost to an empty conditioning stratum are absent from `values` and
    described in `support.failures`.
    """
    identified = _as_identified(result)
    variables = tuple(sorted(identified.expression.scope()))

    missing = [name for name in variables if name not in data.variables]
    if missing:
        raise UnsupportedEstimand(
            f"The estimand references {missing}, which the dataset does not contain. "
            f"Dataset variables: {list(data.variables)}."
        )

    try:
        inner = data.model(variables, fallbacks=fallbacks, replacements=replacements)
    except (DatasetError, SemanticsError) as error:  # pragma: no cover - defensive
        raise UnsupportedEstimand(str(error)) from error

    model = _AuditingModel(inner)
    values, failures = _evaluate_over_scope(identified, model, variables)
    min_count, thinnest = _thinnest_stratum(data, model.strata)

    checked = tuple(
        sorted(
            {
                assumption.code
                for assumption in identified.assumptions
                if assumption.code in DISCHARGEABLE_CODES
            }
        )
    )
    not_checked = tuple(
        assumption
        for assumption in identified.assumptions
        if assumption.code not in DISCHARGEABLE_CODES
    )
    support = SupportReport(
        checked=checked,
        not_checked=not_checked,
        failures=tuple(sorted(failures.values(), key=str)),
        points_total=len(values) + sum(f.points for f in failures.values()),
        points_undefined=sum(f.points for f in failures.values()),
        min_stratum_count=min_count,
        thinnest_stratum=thinnest,
    )

    interval: dict[Point, tuple[float, float] | None] = {}
    replicate_failures = 0
    if bootstrap > 0:
        interval, replicate_failures = _bootstrap_interval(
            identified,
            data,
            variables,
            fallbacks=fallbacks,
            replacements=replacements,
            replicates=bootstrap,
            level=level,
            seed=seed,
            points=tuple(values),
        )

    return Estimate(
        variables=variables,
        values=values,
        support=support,
        assumptions=identified.assumptions,
        theorem=identified.theorem,
        unit=data.unit_description,
        n_rows=data.n_rows,
        n_units=data.n_units,
        interval=interval,
        bootstrap=bootstrap,
        level=level,
        replicate_failures=replicate_failures,
    )


def _bootstrap_interval(
    identified: Identified,
    data: Dataset,
    variables: tuple[str, ...],
    *,
    fallbacks: Mapping[str, Mapping[Point, float]] | None,
    replacements: Mapping[str, Mapping[tuple[Point, Point], float]] | None,
    replicates: int,
    level: float,
    seed: int,
    points: tuple[Point, ...],
) -> tuple[dict[Point, tuple[float, float] | None], int]:
    """Percentile interval over unit-resampled replicates.

    A replicate can leave a stratum empty that was merely thin in the original sample. The
    affected point contributes no draw for that replicate and is counted; a point with no
    surviving draws gets `None` rather than an interval computed from nothing. Both are
    reported, because a high failure rate says the point estimate is resting on very few
    rows -- which is the finding, not an inconvenience.
    """
    rng = random.Random(seed)
    draws: dict[Point, list[float]] = {point: [] for point in points}
    failures = 0

    for _ in range(replicates):
        replicate = data.resample(rng)
        model = replicate.model(variables, fallbacks=fallbacks, replacements=replacements)
        for point in points:
            assignment = dict(zip(variables, point, strict=True))
            try:
                draws[point].append(evaluate(identified.expression, model, assignment))
            except UndefinedEstimand:
                failures += 1
            except MissingKernel as missing:  # pragma: no cover - caught in the main pass
                raise UnsupportedEstimand(str(missing)) from missing

    tail = (1.0 - level) / 2.0
    interval: dict[Point, tuple[float, float] | None] = {}
    for point, sample in draws.items():
        if not sample:
            interval[point] = None
            continue
        ordered = sorted(sample)
        interval[point] = (_percentile(ordered, tail), _percentile(ordered, 1.0 - tail))
    return interval, failures
