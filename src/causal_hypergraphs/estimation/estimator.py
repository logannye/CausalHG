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

Nothing exponential in the estimand's footprint is ever built. The estimand is evaluated by
variable elimination, and the empirical model counts each factor over that factor's own
variables, so a query whose ancestry has forty variables costs its treewidth rather than
`2**40`. The estimate reports what it cost, because a number with no cost attached invites
the next query to be a thousand times worse.
"""
from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from causal_hypergraphs.expression import Expression
from causal_hypergraphs.identification import Assumption, IdentificationResult, Identified
from causal_hypergraphs.semantics import (
    DEFAULT_MAX_ENTRIES,
    Assignment,
    EliminationPlan,
    MissingKernel,
    Model,
    SemanticsError,
    UndefinedEstimand,
    eliminate,
    evaluate,
    plan_elimination,
    with_aliases,
)

from .dataset import Dataset, DatasetError, Point
from .empirical import EmpiricalModel

Evaluator = Callable[[Expression, Model, Assignment], float]

METHODS = ("eliminate", "enumerate")
"""How to evaluate the estimand.

`eliminate` is the default and is what makes a wide query affordable. `enumerate` is the
reference implementation -- slower by an exponential, and kept reachable because agreement
between the two is what verifies the fast path.
"""


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
DISCHARGEABLE_CODES = frozenset(
    {"Target positivity", "Downstream positivity", "Backend positivity"}
)


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


DEFAULT_POLICY_FLOOR = 30
"""Effective rows below which a policy's leverage is reported as a failure.

A convention, and named as one. It is the point below which a multinomial cell estimate
stops being usable rather than a property of any particular estimand, so it is a keyword
argument on `estimate`. What is *not* a convention is `PolicySupport.effective_n`: that
number is computed and printed whatever the floor is, so the disclosure never depends on
the threshold and a caller who disagrees with 30 still sees what the answer rests on.
"""


@dataclass(frozen=True)
class PolicySupport:
    """How many rows an estimand's answer actually rests on, given the policy's weights.

    A deletion estimand is ``sum_t (downstream)(t) * P0^m(t)``. The factor at ``t`` is
    estimated from the rows showing ``out(m) = t``, so the policy's weights decide which
    rows carry the answer. Concentrate the mass on a level the data barely populates and
    the estimate rests on a handful of rows while the header still reports the whole table.

    Positivity cannot see this. It asks whether a conditioning cell is *empty*; this asks
    whether a non-empty cell is being *leaned on*. They are separate findings and are kept
    separate for the reason `Backend positivity` is not `Downstream positivity`: a report
    that flipped positivity to FAIL here would be describing a stratum that is not empty.
    """

    mechanism: str
    variables: tuple[str, ...]
    weights: Mapping[Point, float]
    rows: Mapping[Point, int]
    n_rows: int
    floor: int
    effective_n: float | None
    """``1 / sum_t (w_t**2 / n_t)`` -- the inverse-variance count.

    `None` when the policy feeds no kernel that reads the data, which is not a small
    number but an inapplicable one: there is no leverage to weigh. The two must not share
    a representation, for the same reason `min_stratum_count` distinguishes `None` from 0.
    """

    @property
    def holds(self) -> bool:
        return self.effective_n is None or self.effective_n >= self.floor

    @property
    def overstatement(self) -> float | None:
        """How many times the header row count exceeds the count behind the answer.

        Disclosed on PASS as well as FAIL, and deliberately so: the floor is a convention
        about when a cell stops being estimable, while this is the size of the gap between
        what the report says and what the answer rests on. A policy can clear any floor and
        still be standing on a ninetieth of the table.
        """
        if self.effective_n is None or self.effective_n <= 0.0:
            return None
        return self.n_rows / self.effective_n

    @property
    def heaviest(self) -> tuple[Point, float] | None:
        """The level carrying the most policy mass, which is the one to report."""
        if not self.weights:
            return None
        return max(self.weights.items(), key=lambda item: item[1])

    def summary(self) -> str:
        if self.effective_n is None:
            return (
                f"P0_{self.mechanism} feeds no kernel estimated from these data, so the "
                f"answer carries no data-backed leverage to weigh"
            )
        heaviest = self.heaviest
        assert heaviest is not None  # effective_n is None when there are no weights
        key, mass = heaviest
        cells = ", ".join(
            f"{name}={value!r}" for name, value in zip(self.variables, key, strict=False)
        )
        backing = self.rows.get(key, 0)
        over = self.overstatement
        gap = f" ({over:.0f}x the reported count)" if over is not None and over >= 1.5 else ""
        if self.holds:
            return (
                f"P0_{self.mechanism} rests on {self.effective_n:.0f} effective row(s) "
                f"of {self.n_rows}{gap}"
            )
        return (
            f"P0_{self.mechanism} puts {mass:.3g} of its mass on {cells}, which "
            f"{backing} row(s) back; the answer rests on {self.effective_n:.0f} effective "
            f"row(s), not the {self.n_rows} reported above{gap}"
        )


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
    policy: tuple[PolicySupport, ...] = ()
    """What each deletion policy's answer actually rests on, given its weights.

    Reported beside the positivity certificates rather than folded into them: an estimand
    can have every conditioning cell populated -- positivity genuinely holding -- while the
    policy leans the whole answer on the thinnest of them.
    """
    plan: EliminationPlan | None = None
    """What evaluating this estimand cost, and what enumerating it would have.

    Reported rather than kept internal because the cost is a property of the *query*, not
    of the machine: it says which questions this graph can answer, and a user who can see
    that a nearby query jumped from width 2 to width 14 can ask a different one.
    """
    method: str = "eliminate"

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
        if self.plan is not None:
            lines.append(f"  cost ({self.method}): {self.plan.summary()}")
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
        for report in self.policy:
            lines.append(f"    Policy support: {'PASS' if report.holds else 'FAIL'}")
            lines.append(f"    {report.summary()}")
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

    def __init__(self, inner: Model, expectations: Dataset) -> None:
        self._inner = inner
        # Conditional expectations are group means over rows, not functions of the
        # discretized joint, so they are served by the dataset itself. That is what keeps
        # a continuous readout out of `domains` entirely.
        self._expectations = expectations
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
        self,
        mechanism: str,
        variables: Sequence[str],
        assignment: Assignment,
        marginalized: Sequence[str] = (),
    ) -> float:
        return self._inner.fallback(mechanism, variables, assignment, marginalized)

    def conditional_expectation(
        self, target: str, given: Sequence[str], assignment: Assignment
    ) -> float:
        if given:
            self._record(given, assignment)
        return self._expectations.conditional_expectation(target, given, assignment)

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


def _evaluator(method: str, max_entries: int) -> Evaluator:
    if method == "eliminate":
        return lambda expression, model, assignment: eliminate(
            expression, model, assignment, max_entries=max_entries
        )
    if method == "enumerate":
        return evaluate
    raise ValueError(
        f"Unknown evaluation method {method!r}. Expected one of {list(METHODS)}: "
        "'eliminate' pays the estimand's treewidth, 'enumerate' pays its whole footprint "
        "and exists as the reference the fast path is checked against."
    )


def _evaluate_over_scope(
    identified: Identified,
    model: Model,
    variables: tuple[str, ...],
    evaluator: Evaluator,
) -> tuple[dict[Point, float], dict[tuple[str, Point], SupportFailure]]:
    """Evaluate at every point of the estimand's scope, collecting undefined strata."""
    values: dict[Point, float] = {}
    failures: dict[tuple[str, Point], SupportFailure] = {}
    domains = [model.domains[name] for name in variables]

    for combination in itertools.product(*domains):
        assignment = dict(zip(variables, combination, strict=True))
        try:
            values[combination] = evaluator(identified.expression, model, assignment)
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
    model: EmpiricalModel, strata: set[tuple[tuple[str, ...], Point]]
) -> tuple[int | None, Mapping[str, Any] | None]:
    """The smallest row count among the conditioning cells the estimand actually used.

    An estimand can be perfectly well defined and still be resting on a handful of rows.
    Reporting only pass/fail on positivity would hide that, so the thinnest cell is part
    of the report: it is the number that says how much the quotient can be trusted.

    Reads the counts back off the model rather than re-tallying: these are the very cells
    the model was asked for, and on a wide query a second pass per conditioning set is the
    same order of work as the estimate itself.
    """
    smallest: int | None = None
    where: Mapping[str, Any] | None = None
    for names, point in sorted(strata):
        count = model.counts(names).get(point, 0)
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


def _check_policy_support(
    identified: Identified,
    fallbacks: Mapping[str, Mapping[Point, float]] | None,
    domains: Mapping[str, tuple[Any, ...]],
) -> None:
    """Refuse a deletion policy whose mass sits on levels these data cannot represent.

    `_evaluate_over_scope` walks the product of the *observed* domains, so an output level
    absent from the data is never visited. A policy putting mass there would have that mass
    silently dropped: the returned law sums to less than one while every positivity
    certificate reads PASS, because positivity is a property of the conditioning cells the
    evaluator touched and this mass is at a point it never touched. That is a wrong number
    carrying a clean report, which is the one outcome this module exists to prevent.

    The policy is a *declaration* -- it says what the intervention installs -- so a mismatch
    with the data's support is the caller's to resolve, not the estimator's to absorb. The
    dual direction is already refused: `EmpiricalModel.fallback` raises `MissingKernel` for
    a domain level the table has no entry for. This closes the other side.

    Only positive mass is refused. Spelling the unreachable keys out as `0.0` is how the
    coupled policies in this repo are already written, and drops nothing.

    Marginalized outputs are skipped by construction rather than by a flag: they are summed
    inside the node and no domain is required of anyone, so they are exactly the key
    positions whose name `domains` does not carry.
    """
    if not fallbacks:
        return
    for kernel in identified.expression.kernels():
        if kernel.kind != "fallback":
            continue
        mechanism = kernel.label.removeprefix("P0_")
        table = fallbacks.get(mechanism)
        if table is None:
            continue
        checkable = [
            (position, name)
            for position, name in enumerate(kernel.variables)
            if name in domains
        ]
        for key, mass in table.items():
            if mass <= 0.0:
                continue
            for position, name in checkable:
                if position < len(key) and key[position] not in domains[name]:
                    raise UnsupportedEstimand(
                        f"Policy P0_{mechanism} puts mass {mass} on {name}={key[position]!r}, "
                        f"which these data never show: the observed domain of {name!r} is "
                        f"{list(domains[name])}. Evaluation ranges over the observed domains, "
                        f"so that mass would be dropped and the estimated law would silently "
                        f"fail to sum to one. Either pass `domains=` to the Dataset naming "
                        f"every level the policy uses, or declare a policy these data support."
                    )


def _policy_support(
    identified: Identified,
    data: Dataset,
    fallbacks: Mapping[str, Mapping[Point, float]] | None,
    floor: int,
) -> tuple[PolicySupport, ...]:
    """Weigh each deletion policy against the rows that back the levels it leans on.

    The denominator per level is the row count over `out(m)` itself, not over the
    conditioning cell of any one downstream kernel: `out(m)` is what the policy indexes and
    what the sum ranges over, so it is the axis along which the weights redistribute the
    data. Reading the count off a particular kernel's cell would report a different number
    for each factor and none of them for the estimand.

    A policy whose outputs no data-reading kernel conditions on gets `effective_n=None`
    rather than a number -- the marginalized and policy-only shapes have no leverage to
    weigh, and reporting a small figure for them would read as a warning about nothing.
    """
    if not fallbacks:
        return ()
    kernels = tuple(identified.expression.kernels())
    reads_data = {
        name
        for kernel in kernels
        if kernel.kind != "fallback"
        for name in kernel.given
    }
    reports: list[PolicySupport] = []
    for kernel in kernels:
        if kernel.kind != "fallback":
            continue
        mechanism = kernel.label.removeprefix("P0_")
        table = fallbacks.get(mechanism)
        if table is None:
            continue
        weights = {key: mass for key, mass in table.items() if mass > 0.0}
        countable = all(name in data.variables for name in kernel.variables)
        if not weights or not countable or not (set(kernel.variables) & reads_data):
            effective: float | None = None
            rows: Mapping[Point, int] = {}
        else:
            rows = data.counts(kernel.variables)
            burden = 0.0
            for key, mass in weights.items():
                backing = rows.get(key, 0)
                if backing == 0:
                    # Already refused by `_check_policy_support`; be total anyway.
                    burden = float("inf")
                    break
                burden += mass * mass / backing
            effective = 0.0 if burden == float("inf") else 1.0 / burden
        reports.append(
            PolicySupport(
                mechanism=mechanism,
                variables=kernel.variables,
                weights=weights,
                rows=rows,
                n_rows=data.n_rows,
                floor=floor,
                effective_n=effective,
            )
        )
    return tuple(reports)


def estimate(
    result: IdentificationResult | Identified,
    data: Dataset,
    *,
    fallbacks: Mapping[str, Mapping[Point, float]] | None = None,
    replacements: Mapping[str, Mapping[tuple[Point, Point], float]] | None = None,
    bootstrap: int = 0,
    level: float = 0.95,
    seed: int = 0,
    method: str = "eliminate",
    max_entries: int = DEFAULT_MAX_ENTRIES,
    policy_floor: int = DEFAULT_POLICY_FLOOR,
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
    method:
        `"eliminate"` (default) evaluates by variable elimination and costs the estimand's
        treewidth; `"enumerate"` walks the whole footprint and is the reference the fast
        path is verified against. They return the same numbers.
    max_entries:
        Largest intermediate table elimination may build. Exceeding it raises
        `IntractableQuery` naming the variables that met, rather than exhausting memory.
    policy_floor:
        Effective rows below which a policy's leverage is reported as a failure. The
        effective count is computed and printed whatever this is set to, so lowering it
        hides a verdict, never the number behind it.

    Returns
    -------
    An `Estimate` whose `values` cover exactly the points where the estimand is defined in
    these data. Points lost to an empty conditioning stratum are absent from `values` and
    described in `support.failures`.
    """
    evaluator = _evaluator(method, max_entries)
    identified = _as_identified(result)
    # Two different sets, and conflating them is a bug the marginal-query work exposed.
    # `scope` is what the answer is indexed by -- the points that come back in `values`.
    # `footprint` additionally covers variables the estimand *sums over*, which the model
    # must still supply domains for even though no caller ever binds them.
    variables = tuple(sorted(identified.expression.scope()))
    # A copied variable is a second name for one that is in the data, so the column check
    # and the model are built over what it copies. Without this an identifying formula that
    # needed a fresh name -- the `x'` of the front-door estimand -- would be refused as
    # referencing a column no dataset has, which is a formula the library itself emitted.
    aliases = identified.aliases
    footprint = tuple(
        sorted({aliases.get(name, name) for name in identified.expression.footprint()})
    )

    missing = [name for name in footprint if name not in data.variables]
    if missing:
        raise UnsupportedEstimand(
            f"The estimand references {missing}, which the dataset does not contain. "
            f"Dataset variables: {list(data.variables)}."
        )

    try:
        inner = EmpiricalModel(
            data, footprint, fallbacks=fallbacks, replacements=replacements
        )
    except (DatasetError, SemanticsError) as error:  # pragma: no cover - defensive
        raise UnsupportedEstimand(str(error)) from error

    # Computed even when enumerating: the plan is what tells a caller that the query they
    # just paid an exponential for had a width of two.
    # Refused before the plan is costed: a policy the data cannot represent makes every
    # number downstream wrong, so there is nothing to price.
    domains = with_aliases(inner, aliases).domains
    _check_policy_support(identified, fallbacks, domains)

    plan = plan_elimination(
        identified.expression,
        domains,
        bound=variables,
    )

    # Aliases resolve outermost, so everything below records base names: the audit that
    # discharges positivity certificates counts strata of real columns, never of copies.
    auditing = _AuditingModel(inner, data)
    model = with_aliases(auditing, aliases)
    values, failures = _evaluate_over_scope(identified, model, variables, evaluator)
    min_count, thinnest = _thinnest_stratum(inner, auditing.strata)

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

    policy = _policy_support(identified, data, fallbacks, policy_floor)

    interval: dict[Point, tuple[float, float] | None] = {}
    replicate_failures = 0
    if bootstrap > 0:
        interval, replicate_failures = _bootstrap_interval(
            identified,
            data,
            variables,
            footprint,
            fallbacks=fallbacks,
            replacements=replacements,
            replicates=bootstrap,
            level=level,
            seed=seed,
            points=tuple(values),
            evaluator=evaluator,
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
        policy=policy,
        plan=plan,
        method=method,
    )


def _bootstrap_interval(
    identified: Identified,
    data: Dataset,
    variables: tuple[str, ...],
    footprint: tuple[str, ...],
    *,
    fallbacks: Mapping[str, Mapping[Point, float]] | None,
    replacements: Mapping[str, Mapping[tuple[Point, Point], float]] | None,
    replicates: int,
    level: float,
    seed: int,
    points: tuple[Point, ...],
    evaluator: Evaluator,
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
        inner = EmpiricalModel(
            replicate, footprint, fallbacks=fallbacks, replacements=replacements
        )
        model = with_aliases(_AuditingModel(inner, replicate), identified.aliases)
        for point in points:
            assignment = dict(zip(variables, point, strict=True))
            try:
                draws[point].append(evaluator(identified.expression, model, assignment))
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
