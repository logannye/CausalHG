"""Checkers that compare the compiler's symbolic answers against exact ground truth.

Each checker is a pure function over data, and the separation checker takes its oracle
by injection. That is deliberate: it lets `test_conformance_harness.py` feed a
deliberately wrong estimand or a deliberately unsound oracle and confirm the checker
reports the failure, which is the only evidence that a green sweep means anything.
"""
from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from causal_hypergraphs.expression import Expression
from causal_hypergraphs.graph import MechanismGraph
from causal_hypergraphs.semantics import DiscreteModel, UndefinedEstimand, evaluate
from causal_hypergraphs.separation import d_separated

Point = tuple[int, ...]

# Laws are products of at most a handful of factors bounded by 1, so absolute
# tolerances are meaningful and no relative scaling is needed.
ESTIMAND_TOLERANCE = 1e-12
# Kept tight on purpose. A loose tolerance here would let a genuine dependence read as
# independence, which would *mask* an unsound separation verdict rather than surface it.
# Products of sums over a few dozen terms carry ~1e-16 of floating-point noise, so this
# sits four orders of magnitude above the noise floor and far below any real effect.
INDEPENDENCE_TOLERANCE = 1e-12


# --- identifier conformance ---------------------------------------------------


@dataclass(frozen=True)
class EstimandReport:
    """Outcome of comparing one estimand against one interventional law."""

    points_checked: int
    mismatches: tuple[tuple[Point, float, float], ...]
    undefined_at_positive_mass: tuple[Point, ...]
    nonzero_where_truth_zero: tuple[tuple[Point, float], ...]

    @property
    def conforms(self) -> bool:
        """True iff the estimand is correct everywhere the interventional law has mass."""
        return not (
            self.mismatches or self.undefined_at_positive_mass or self.nonzero_where_truth_zero
        )

    @property
    def undefined_somewhere(self) -> bool:
        return bool(self.undefined_at_positive_mass)

    def summary(self) -> str:
        parts = []
        if self.mismatches:
            point, expected, actual = self.mismatches[0]
            parts.append(
                f"{len(self.mismatches)} mismatched point(s); first {point}: "
                f"expected {expected!r}, got {actual!r}"
            )
        if self.undefined_at_positive_mass:
            parts.append(
                f"{len(self.undefined_at_positive_mass)} point(s) of positive interventional "
                f"mass where the estimand is undefined; first "
                f"{self.undefined_at_positive_mass[0]}"
            )
        if self.nonzero_where_truth_zero:
            point, actual = self.nonzero_where_truth_zero[0]
            parts.append(f"estimand is {actual!r} at {point}, where the truth is 0")
        return "; ".join(parts) or "conforms"


def check_estimand(
    expression: Expression,
    model: DiscreteModel,
    truth: Mapping[Point, float],
    variables: Sequence[str],
    tolerance: float = ESTIMAND_TOLERANCE,
) -> EstimandReport:
    """Evaluate `expression` at every point and compare it against `truth`.

    The property checked is: the estimand equals the interventional law at every point
    the law gives positive mass, and is zero wherever the law is zero. Being *undefined*
    where the law is zero is tolerated -- a 0/0 on a null set carries no information and
    is not a defect. Being undefined where the law has mass is a defect, because such an
    estimand cannot answer the query it claims to identify.
    """
    mismatches: list[tuple[Point, float, float]] = []
    undefined: list[Point] = []
    spurious: list[tuple[Point, float]] = []

    points = list(itertools.product(*(model.domains[v] for v in variables)))
    for point in points:
        assignment = dict(zip(variables, point, strict=True))
        expected = truth[point]
        try:
            actual = evaluate(expression, model, assignment)
        except UndefinedEstimand:
            if expected > 0.0:
                undefined.append(point)
            continue
        if expected > 0.0:
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
                mismatches.append((point, expected, actual))
        elif abs(actual) > tolerance:
            spurious.append((point, actual))

    return EstimandReport(
        points_checked=len(points),
        mismatches=tuple(mismatches),
        undefined_at_positive_mass=tuple(undefined),
        nonzero_where_truth_zero=tuple(spurious),
    )


# --- conditional independence -------------------------------------------------


def _sub_marginal(
    joint: Mapping[Point, float], variables: Sequence[str], subset: Sequence[str]
) -> dict[Point, float]:
    index = {name: position for position, name in enumerate(variables)}
    positions = [index[v] for v in subset]
    result: dict[Point, float] = {}
    for key, probability in joint.items():
        reduced = tuple(key[p] for p in positions)
        result[reduced] = result.get(reduced, 0.0) + probability
    return result


def conditional_independence_holds(
    joint: Mapping[Point, float],
    variables: Sequence[str],
    x: Iterable[str],
    y: Iterable[str],
    z: Iterable[str],
    tolerance: float = INDEPENDENCE_TOLERANCE,
) -> bool:
    """Exact test of X indep Y | Z in `joint`.

    Uses the division-free form P(x,y,z) * P(z) == P(x,z) * P(y,z), which is equivalent
    to the conditional factorization wherever P(z) > 0 and is vacuously satisfied where
    P(z) = 0. Avoiding the division keeps the test well behaved on the singular kernels
    the generator deliberately produces.
    """
    xs, ys, zs = tuple(sorted(x)), tuple(sorted(y)), tuple(sorted(z))
    m_xyz = _sub_marginal(joint, variables, xs + ys + zs)
    m_xz = _sub_marginal(joint, variables, xs + zs)
    m_yz = _sub_marginal(joint, variables, ys + zs)
    m_z = _sub_marginal(joint, variables, zs)

    domains = {v: (0, 1) for v in variables}
    for x_values in itertools.product(*(domains[v] for v in xs)):
        for y_values in itertools.product(*(domains[v] for v in ys)):
            for z_values in itertools.product(*(domains[v] for v in zs)):
                lhs = m_xyz.get(x_values + y_values + z_values, 0.0) * m_z.get(z_values, 0.0)
                rhs = m_xz.get(x_values + z_values, 0.0) * m_yz.get(y_values + z_values, 0.0)
                if not math.isclose(lhs, rhs, rel_tol=0.0, abs_tol=tolerance):
                    return False
    return True


# --- separation conformance ---------------------------------------------------

Triple = tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]
Oracle = Callable[[MechanismGraph, tuple[str, ...], tuple[str, ...], tuple[str, ...]], bool]


def separation_triples(
    variables: Sequence[str], limit: int = 60, seed: int = 0
) -> tuple[Triple, ...]:
    """Deterministically chosen disjoint (X, Y, Z) triples over `variables`."""
    candidates: list[Triple] = []
    names = sorted(variables)
    for x_size in (1, 2):
        for x in itertools.combinations(names, x_size):
            remaining_y = [v for v in names if v not in x]
            for y_size in (1, 2):
                for y in itertools.combinations(remaining_y, y_size):
                    rest = [v for v in remaining_y if v not in y]
                    for z_size in (0, 1, 2):
                        for z in itertools.combinations(rest, z_size):
                            candidates.append((x, y, z))
    if len(candidates) <= limit:
        return tuple(candidates)
    return tuple(random.Random(seed).sample(candidates, limit))


@dataclass(frozen=True)
class SeparationReport:
    """Outcome of comparing separation verdicts against exact conditional independence."""

    triples_checked: int
    unsound: tuple[Triple, ...]  # claimed separated, but dependent in the law
    incomplete: tuple[Triple, ...]  # independent in the law, but not claimed separated

    def summary(self) -> str:
        if not self.unsound:
            return f"sound over {self.triples_checked} triples"
        x, y, z = self.unsound[0]
        return (
            f"{len(self.unsound)}/{self.triples_checked} unsound verdict(s); "
            f"first: {list(x)} vs {list(y)} given {list(z)} claimed separated but dependent"
        )


def library_oracle(
    graph: MechanismGraph, x: tuple[str, ...], y: tuple[str, ...], z: tuple[str, ...]
) -> bool:
    return d_separated(graph, set(x), set(y), given=set(z))


def check_separation_claims(
    graph: MechanismGraph,
    joint: Mapping[Point, float],
    variables: Sequence[str],
    triples: Sequence[Triple],
    oracle: Oracle = library_oracle,
) -> SeparationReport:
    """Compare an oracle's separation verdicts against exact independence in `joint`.

    Unsoundness -- a claimed separation that does not hold in the law -- is the failure
    that matters: it licenses an independence downstream. Incompleteness is recorded but
    is not a defect on its own, since undeclared functional determination legitimately
    produces independences the graphical criterion cannot see.
    """
    unsound: list[Triple] = []
    incomplete: list[Triple] = []
    for x, y, z in triples:
        claimed = oracle(graph, x, y, z)
        actual = conditional_independence_holds(joint, variables, x, y, z)
        if claimed and not actual:
            unsound.append((x, y, z))
        elif actual and not claimed:
            incomplete.append((x, y, z))
    return SeparationReport(
        triples_checked=len(triples),
        unsound=tuple(unsound),
        incomplete=tuple(incomplete),
    )
