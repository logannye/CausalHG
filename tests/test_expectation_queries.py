"""`E[Y | do]` for a continuous readout, without ever enumerating Y.

The marginal reduction shrank a query to the outcome's ancestry, but the outcome itself
was still a discrete variable whose domain had to be enumerated. That is the wrong shape
for biology: a readout is an expression level, a growth rate, a viability score -- a
number, not a category -- and binning it is a modelling choice that can manufacture or
destroy the very data support the estimator checks.

Asking for an expectation removes the need. In the truncated factorization the outcome
appears in exactly one factor, the one for its producing mechanism, so

    E[Y | do] = sum over the ancestry of  (other factors) * E[Y | in(m_Y)]

and Y enters only through a conditional mean -- a regression, estimable with Y continuous.
Y's domain is never enumerated because Y is never a free or bound variable of the
estimand.

That the co-outputs of Y's mechanism also drop out is not an assumption but a consequence
of C1: a co-output that were an ancestor of Y would make Y's own mechanism its own
ancestor, which is the cycle C1 forbids. So the joint factor can be marginalized to
`E[Y | in(m_Y)]` with nothing left over.
"""
from __future__ import annotations

import itertools
import random

import pytest

from causal_hypergraphs import (
    DeleteMechanism,
    Identified,
    MechanismGraph,
    ReplaceMechanism,
    identify,
    identify_expectation,
)
from causal_hypergraphs.estimation import Dataset, estimate
from causal_hypergraphs.semantics import DiscreteModel, evaluate

BINARY = (0, 1)


def _chain_graph() -> MechanismGraph:
    """A -> m1 -> B -> m2 -> Y."""
    return MechanismGraph(
        variables={"A", "B", "Y"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("B",)},
            "m2": {"inputs": ("B",), "outputs": ("Y",)},
        },
    )


def _identified(graph: MechanismGraph, query: object, outcome: str) -> Identified:
    result = identify_expectation(graph, query, outcome)  # type: ignore[arg-type]
    assert isinstance(result, Identified), result
    return result


# --- the shape --------------------------------------------------------------------


def test_the_outcome_is_absent_from_the_footprint() -> None:
    """The whole point: Y is integrated out at compile time, never enumerated."""
    result = _identified(_chain_graph(), DeleteMechanism("m1"), "Y")

    assert "Y" not in result.expression.footprint(), result.expression
    assert result.expression.scope() == frozenset()
    assert "E[Y | B]" in str(result.expression), result.expression


def test_the_expectation_matches_the_density_form_it_replaces() -> None:
    """Correctness: E[Y | do] must equal sum_y y * P(y | do) computed the long way."""
    graph = _chain_graph()
    p_a = {0: 0.3, 1: 0.7}
    p_b = {(0, 0): 0.8, (1, 0): 0.2, (0, 1): 0.35, (1, 1): 0.65}  # (b, a)
    p_y = {(0, 0): 0.6, (1, 0): 0.4, (0, 1): 0.25, (1, 1): 0.75}  # (y, b)
    joint = {
        (a, b, y): p_a[a] * p_b[(b, a)] * p_y[(y, b)]
        for a, b, y in itertools.product(BINARY, BINARY, BINARY)
    }
    policy = {(0,): 0.45, (1,): 0.55}
    model = DiscreteModel(
        domains={"A": BINARY, "B": BINARY, "Y": BINARY}, joint=joint, fallbacks={"m1": policy}
    )
    model.validate()

    density = identify(graph, DeleteMechanism("m1", outcomes={"Y"}))
    assert isinstance(density, Identified)
    long_way = sum(
        value * evaluate(density.expression, model, {"Y": value}) for value in BINARY
    )

    functional = _identified(graph, DeleteMechanism("m1"), "Y")
    assert evaluate(functional.expression, model, {}) == pytest.approx(long_way, abs=1e-12)


def test_the_expectation_matches_the_density_form_across_generated_models() -> None:
    """The same identity, swept rather than fixtured.

    For every generated model, every mechanism and every outcome the functional supports,
    `E[Y | do]` must equal `sum_y y * P(y | do)` from the density form. One fixture can
    agree by construction; a sweep over models the author did not choose cannot.
    """
    from tests.conformance.generation import generate_model

    checked = 0
    for seed in range(30):
        model = generate_model(seed, allow_hidden=False, shapes=("positive",))
        graph = model.graph()
        discrete = DiscreteModel(
            domains=model.domains, joint=model.joint(), fallbacks=dict(model.fallbacks)
        )
        for spec in model.mechanisms:
            for outcome in model.variables:
                if outcome in spec.outputs:
                    continue  # answered by the supplied policy; refused by design
                density = identify(graph, DeleteMechanism(spec.name, outcomes={outcome}))
                assert isinstance(density, Identified)
                long_way = sum(
                    value * evaluate(density.expression, discrete, {outcome: value})
                    for value in BINARY
                )
                functional = _identified(graph, DeleteMechanism(spec.name), outcome)
                assert "Y" not in functional.expression.footprint()
                assert outcome not in functional.expression.footprint()
                assert evaluate(functional.expression, discrete, {}) == pytest.approx(
                    long_way, abs=1e-12
                ), f"seed {seed} / delete({spec.name}) / E[{outcome}]"
                checked += 1

    assert checked > 200, f"only {checked} expectation(s) checked"


def test_an_exogenous_outcome_is_just_its_mean() -> None:
    """No producing mechanism means no factor to replace; the answer is E[Y]."""
    graph = MechanismGraph(
        variables={"A", "B"}, mechanisms={"m1": {"inputs": ("A",), "outputs": ("B",)}}
    )
    result = _identified(graph, DeleteMechanism("m1"), "A")

    assert str(result.expression) == "E[A]"
    assert result.expression.footprint() == frozenset()


def test_replacement_queries_support_expectations_too() -> None:
    result = _identified(_chain_graph(), ReplaceMechanism("m1", "m1_prime"), "Y")

    assert "Y" not in result.expression.footprint()
    assert "P_m1_prime" in str(result.expression), result.expression


# --- the refusal ------------------------------------------------------------------


def test_an_outcome_produced_by_the_target_is_refused_not_guessed() -> None:
    """You already supplied that answer; estimating it would be theatre.

    If Y is an output of the mechanism being intervened on, its post-intervention law *is*
    the policy the caller passed in. There is nothing to learn from data, so returning a
    number dressed as an estimate would misrepresent where it came from.
    """
    with pytest.raises(ValueError, match="policy"):
        identify_expectation(_chain_graph(), DeleteMechanism("m2"), "Y")


def test_an_unknown_outcome_is_rejected() -> None:
    with pytest.raises(ValueError, match="ghost"):
        identify_expectation(_chain_graph(), DeleteMechanism("m1"), "ghost")


# --- continuous readouts ----------------------------------------------------------


def test_a_continuous_readout_needs_no_binning() -> None:
    """The motivating case: Y is a float column and is never discretized.

    Ground truth is exact here -- E[Y | do] = sum_b P0(b) * E[Y | B=b], and both
    conditional means are computed directly from the rows -- so any disagreement is the
    estimator, not sampling error.
    """
    rng = random.Random(4)
    rows = []
    for _ in range(4_000):
        a = 1 if rng.random() < 0.4 else 0
        b = 1 if rng.random() < (0.65 if a else 0.2) else 0
        rows.append(
            {"A": a, "B": b, "Y": rng.gauss(3.0 + 5.0 * b, 0.75), "run": f"r{rng.randrange(40)}"}
        )

    data = Dataset.from_records(rows, unit="run", measures=("Y",))
    assert "Y" not in data.variables  # not a modelled discrete variable
    assert "Y" in data.measures

    policy = {(0,): 0.45, (1,): 0.55}
    result = _identified(_chain_graph(), DeleteMechanism("m1"), "Y")
    est = estimate(result, data, fallbacks={"m1": policy})

    means = {}
    for value in BINARY:
        matching = [row["Y"] for row in rows if row["B"] == value]
        means[value] = sum(matching) / len(matching)
    expected = sum(policy[(value,)] * means[value] for value in BINARY)

    assert est.values[()] == pytest.approx(expected, rel=1e-9)


def test_a_continuous_readout_gets_an_interval() -> None:
    rng = random.Random(5)
    rows = [
        {
            "A": (a := 1 if rng.random() < 0.4 else 0),
            "B": (b := 1 if rng.random() < (0.65 if a else 0.2) else 0),
            "Y": rng.gauss(3.0 + 5.0 * b, 0.75),
            "run": f"r{rng.randrange(40)}",
        }
        for _ in range(4_000)
    ]
    data = Dataset.from_records(rows, unit="run", measures=("Y",))
    result = _identified(_chain_graph(), DeleteMechanism("m1"), "Y")

    est = estimate(result, data, fallbacks={"m1": {(0,): 0.45, (1,): 0.55}}, bootstrap=200, seed=2)

    bounds = est.interval[()]
    assert bounds is not None
    low, high = bounds
    assert low < est.values[()] < high
    assert high - low < 1.0  # 4,000 rows over 40 runs should localize the mean


def test_an_empty_regression_cell_is_named_not_averaged_over_nothing() -> None:
    """`E[Y | B=b]` with no rows at `B=b` is undefined, and must say so.

    Same failure mode as an empty conditioning cell in the density form, and it must
    surface the same way: a named stratum, not a mean over an empty list and not a nan.
    """
    rng = random.Random(6)
    rows = [
        {"A": 0, "B": 0, "Y": rng.gauss(3.0, 0.5), "run": f"r{rng.randrange(20)}"}
        for _ in range(500)
    ]
    data = Dataset.from_records(
        rows, unit="run", measures=("Y",), domains={"A": BINARY, "B": BINARY}
    )
    result = _identified(_chain_graph(), DeleteMechanism("m1"), "Y")

    est = estimate(result, data, fallbacks={"m1": {(0,): 0.45, (1,): 0.55}})

    assert not est.support.holds
    assert any("B" in failure.stratum for failure in est.support.failures), est.support.failures
    assert est.values == {}
