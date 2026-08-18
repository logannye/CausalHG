"""The data-facing path: an identified estimand evaluated against an actual dataset.

These tests pin the contract at the boundary, on datasets whose empirical law is exact by
construction so that no sampling noise is involved and every expected number can be
written down in closed form. Convergence from finite samples is a separate concern and
lives in `test_estimation_convergence.py`.

The contract has three parts, and the third is the one that makes this a tool rather than
a calculator:

1. an `Identified` estimand plus a dataset yields a number at every point of its scope;
2. anything else -- a refusal from `identify`, a variable the data lack -- raises rather
   than returning a plausible number;
3. the positivity certificates the estimand carries are *discharged against the data*, and
   the strata where they fail are named. A quotient over an empty cell is the dominant
   practical failure mode in applied causal work, and the usual outcome is a confident
   number produced by dividing by a near-zero denominator.
"""
from __future__ import annotations

import itertools

import pytest

from causal_hypergraphs import (
    DeleteMechanism,
    Identified,
    MechanismGraph,
    ReplaceMechanism,
    identify,
)
from causal_hypergraphs.estimation import (
    DISCHARGEABLE_CODES,
    Dataset,
    NotIdentified,
    UnsupportedEstimand,
    estimate,
)
from tests.conformance.generation import generate_model

BINARY = (0, 1)

# An exact empirical law: counts, not probabilities, so the fixture has no rounding.
#   A -> m1 -> {C, D}
COUNTS: dict[tuple[int, int, int], int] = {
    (0, 0, 0): 120,
    (0, 0, 1): 30,
    (0, 1, 0): 20,
    (0, 1, 1): 30,
    (1, 0, 0): 40,
    (1, 0, 1): 60,
    (1, 1, 0): 50,
    (1, 1, 1): 150,
}
TOTAL = sum(COUNTS.values())  # 500

# A coupled deletion policy: C and D stay equal after m1 is removed.
P0_M1 = {(0, 0): 0.5, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.5}


def _graph() -> MechanismGraph:
    return MechanismGraph(
        variables={"A", "C", "D"},
        mechanisms={"m1": {"inputs": ("A",), "outputs": ("C", "D")}},
    )


def _records() -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for (a, c, d), count in COUNTS.items():
        rows.extend({"A": a, "C": c, "D": d} for _ in range(count))
    return rows


def _dataset() -> Dataset:
    return Dataset.from_records(_records())


def _identified() -> Identified:
    result = identify(_graph(), DeleteMechanism("m1"))
    assert isinstance(result, Identified), result
    return result


def _empirical_p_a(a: int) -> float:
    return sum(count for (av, _, _), count in COUNTS.items() if av == a) / TOTAL


# --- the point estimate -----------------------------------------------------------


def test_estimate_returns_the_interventional_law_at_every_point_of_scope() -> None:
    """P(A, C, D | delete(m1)) = P(A) * P0_m1(C, D), read off the data exactly."""
    est = estimate(_identified(), _dataset(), fallbacks={"m1": P0_M1})

    assert est.variables == ("A", "C", "D")
    for a, c, d in itertools.product(BINARY, BINARY, BINARY):
        expected = _empirical_p_a(a) * P0_M1[(c, d)]
        assert est.values[(a, c, d)] == pytest.approx(expected, abs=1e-12)


def test_the_estimated_law_is_a_distribution() -> None:
    """A post-intervention law that does not sum to one is a bug, not a rounding artifact."""
    est = estimate(_identified(), _dataset(), fallbacks={"m1": P0_M1})

    assert sum(est.values.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(value >= 0.0 for value in est.values.values())


def test_the_estimate_carries_the_sample_size_and_the_unit_of_independence() -> None:
    """Both are needed to read the interval, so neither may be implicit."""
    est = estimate(_identified(), _dataset(), fallbacks={"m1": P0_M1})

    assert est.n_rows == TOTAL
    assert est.n_units == TOTAL  # no unit column declared: each row is its own unit
    assert "row" in est.unit


def test_a_declared_unit_column_groups_rows() -> None:
    """Rows sharing a unit are resampled together, so the count must reflect the grouping."""
    records = [dict(row, donor=f"d{index % 25}") for index, row in enumerate(_records())]
    data = Dataset.from_records(records, unit="donor")

    assert data.n_rows == TOTAL
    assert data.n_units == 25
    # The unit column must not become a modelled variable.
    assert "donor" not in data.variables


# --- refusals ---------------------------------------------------------------------


def test_estimating_a_refusal_raises_and_names_what_is_missing() -> None:
    """`Unknown` carries the reason identification failed; it must not be discarded."""
    graph = MechanismGraph(
        variables={"A", "C", "D"},
        mechanisms={"m1": {"inputs": ("A",), "outputs": ("C", "D")}},
        fallback_variables={"A"},  # no policy covering C or D
    )
    refusal = identify(graph, DeleteMechanism("m1"))

    with pytest.raises(NotIdentified) as caught:
        estimate(refusal, _dataset(), fallbacks={"m1": P0_M1})  # type: ignore[arg-type]

    assert "fallback" in str(caught.value).lower()


def test_a_dataset_missing_a_variable_in_scope_raises() -> None:
    """Silently marginalizing an absent variable would answer a different question."""
    records = [{"A": row["A"], "C": row["C"]} for row in _records()]

    with pytest.raises(UnsupportedEstimand) as caught:
        estimate(_identified(), Dataset.from_records(records), fallbacks={"m1": P0_M1})

    assert "D" in str(caught.value)


# --- certificate discharge --------------------------------------------------------


def test_a_satisfied_positivity_certificate_is_reported_as_checked() -> None:
    """The certificate is not just carried through; the estimate says it was verified."""
    est = estimate(_identified(), _dataset(), fallbacks={"m1": P0_M1})

    assert est.support.holds
    assert not est.support.failures
    assert "Downstream positivity" in est.support.checked
    # This estimand is P(A) * P0_m1(C,D): it conditions on nothing, so there is no
    # stratum that could be empty. That is not the same as a stratum with zero rows, and
    # reporting it as `0` would read as the opposite of what is true.
    assert est.support.min_stratum_count is None
    assert "conditions on nothing" in est.support.summary()


def test_a_surviving_conditional_reports_its_thinnest_stratum() -> None:
    """A defined estimand can still be resting on very few rows; say how few.

    Pass/fail on positivity is not enough to judge a quotient. `P(F | C,E)` survives here
    because every cell has support, but one cell has an order of magnitude fewer rows than
    the others, and that is the number that bounds how much the estimate can be trusted.
    """
    graph = MechanismGraph(
        variables={"A", "C", "E", "F"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("C",)},
            "m2": {"inputs": ("C", "E"), "outputs": ("F",)},
        },
    )
    result = identify(graph, DeleteMechanism("m1"))
    assert isinstance(result, Identified)

    records: list[dict[str, int]] = []
    for a, c, e, f in itertools.product(BINARY, BINARY, BINARY, BINARY):
        repeats = 3 if (c == 1 and e == 1) else 40
        records.extend({"A": a, "C": c, "E": e, "F": f} for _ in range(repeats))

    est = estimate(result, Dataset.from_records(records), fallbacks={"m1": {(0,): 0.5, (1,): 0.5}})

    assert est.support.holds
    # `P(F | C,E)` is a ratio of two counts, and the numerator cell {C=1, E=1, F=f} is the
    # smaller one: 2 values of A x 3 repeats. Reporting the conditioning cell {C=1, E=1}
    # instead would claim 12 rows of support for a quantity resting on 6.
    assert est.support.min_stratum_count == 6
    assert est.support.thinnest_stratum is not None
    assert {"C": 1, "E": 1}.items() <= est.support.thinnest_stratum.items()


def test_an_empty_stratum_is_named_rather_than_divided_through() -> None:
    """The dominant practical failure: a conditioning cell with no observations.

    Deleting `m2` leaves `P(F | C,E)` in the estimand. If no row has `C=1, E=1` then that
    conditional is undefined there, and the honest output is a named stratum -- not a
    number produced by dividing by zero, and not a silent `nan`.
    """
    graph = MechanismGraph(
        variables={"A", "C", "E", "F"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("C",)},
            "m2": {"inputs": ("C", "E"), "outputs": ("F",)},
        },
    )
    result = identify(graph, DeleteMechanism("m1"))
    assert isinstance(result, Identified)

    # Every combination except C=1, E=1 -- that conditioning cell is empty.
    records = [
        {"A": a, "C": c, "E": e, "F": f}
        for a, c, e, f in itertools.product(BINARY, BINARY, BINARY, BINARY)
        if not (c == 1 and e == 1)
    ]
    est = estimate(
        result,
        Dataset.from_records(records, domains={v: BINARY for v in ("A", "C", "E", "F")}),
        fallbacks={"m1": {(0,): 0.5, (1,): 0.5}},
    )

    assert not est.support.holds
    assert est.support.failures
    strata = [failure.stratum for failure in est.support.failures]
    assert {"C": 1, "E": 1} in strata, strata
    # The points it could not reach are absent from `values`, never present as nan.
    assert all(value == value for value in est.values.values())  # noqa: PLR0124
    assert len(est.values) < 2 ** 4
    # Every point of scope is accounted for: reached, or attributed to a named stratum.
    assert est.support.points_total == 2 ** 4
    assert len(est.values) + est.support.points_undefined == 2 ** 4


def test_a_summed_variable_must_be_supplied_to_the_model_even_though_no_caller_binds_it() -> None:
    """Scope and footprint are different sets, and treating them as one is a real bug.

    A marginal estimand is indexed by its outcome alone, but evaluating it ranges over the
    whole ancestral closure. Building the model from the scope leaves the summed variables
    without domains, and the failure surfaces as a `KeyError` deep inside the evaluator
    rather than as anything a caller could act on. Named here so a refactor that collapses
    the two is caught with a message instead of a traceback.
    """
    graph = MechanismGraph(
        variables={"a", "b", "c"},
        mechanisms={
            "m1": {"inputs": ("a",), "outputs": ("b",)},
            "m2": {"inputs": ("b",), "outputs": ("c",)},
        },
    )
    result = identify(graph, DeleteMechanism("m1", outcomes={"c"}))
    assert isinstance(result, Identified)
    assert result.expression.scope() == frozenset({"c"})
    assert result.expression.footprint() == frozenset({"a", "b", "c"})

    records = [
        {"a": a, "b": b, "c": c}
        for a, b, c in itertools.product(BINARY, BINARY, BINARY)
        for _ in range(10)
    ]
    est = estimate(
        result, Dataset.from_records(records), fallbacks={"m1": {(0,): 0.5, (1,): 0.5}}
    )

    assert set(est.values) == {(0,), (1,)}
    assert sum(est.values.values()) == pytest.approx(1.0, abs=1e-12)


def test_discharged_codes_are_codes_the_compiler_actually_emits() -> None:
    """`DISCHARGEABLE_CODES` is a literal; the set it must match is computed.

    A code listed here is announced to the user as verified against their data. A typo, or
    a code renamed in the compiler, would leave the label attached to nothing -- the
    estimate would report a certificate as checked while checking a certificate that no
    longer exists. So the list is pinned against the codes the compiler emits, collected by
    running it rather than by writing them down a second time.
    """
    emitted: set[str] = set()
    for seed in range(60):
        model = generate_model(seed)
        graph = model.graph()
        for spec in model.mechanisms:
            for query in (
                DeleteMechanism(spec.name),
                ReplaceMechanism(spec.name, f"{spec.name}_prime"),
            ):
                result = identify(graph, query)
                if isinstance(result, Identified):
                    emitted.update(a.code for a in result.assumptions)

    unknown = DISCHARGEABLE_CODES - emitted
    assert not unknown, (
        f"{sorted(unknown)} are listed as dischargeable but no identified result emits "
        "them, so the estimator would report a certificate that does not exist"
    )


def test_every_positivity_certificate_the_compiler_emits_is_dischargeable() -> None:
    """The other direction: a new positivity assumption must not go silently unchecked.

    Default-to-unchecked is the safe behaviour, but silently safe is still silent. If the
    compiler grows a certificate about support that this module does not discharge, that
    should surface here rather than as a summary quietly listing it under 'not checked'.
    """
    emitted: set[str] = set()
    for seed in range(60):
        model = generate_model(seed)
        graph = model.graph()
        for spec in model.mechanisms:
            for query in (
                DeleteMechanism(spec.name),
                ReplaceMechanism(spec.name, f"{spec.name}_prime"),
            ):
                result = identify(graph, query)
                if isinstance(result, Identified):
                    emitted.update(a.code for a in result.assumptions)

    positivity_like = {code for code in emitted if "positivity" in code.lower()}
    assert positivity_like, "no positivity certificate was emitted; this gate is vacuous"
    assert positivity_like <= DISCHARGEABLE_CODES, (
        f"{sorted(positivity_like - DISCHARGEABLE_CODES)} look like support certificates "
        "but are not discharged against the data"
    )


def test_the_summary_leads_with_what_the_data_cannot_check() -> None:
    """C2 is the assumption most likely to be false and least likely to be noticed.

    Once this path returns numbers, the numbers get believed. `identify` records C1-C4 in
    a list nobody reads; the estimate must surface the unverifiable ones at the point of
    use, separated from the ones actually discharged against the data.
    """
    summary = estimate(_identified(), _dataset(), fallbacks={"m1": P0_M1}).summary()

    not_checked, _, checked = summary.partition("Checked against the data")
    assert "Not checked" in not_checked
    assert "C2" in not_checked, summary
    assert "independent exogenous noise" in not_checked.lower()
    assert "Downstream positivity" in checked, summary
    # The unverifiable block comes first: it is the caveat, not the footnote.
    assert summary.index("Not checked") < summary.index("Checked against the data")
