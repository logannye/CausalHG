"""Dataset ingestion preserves sample meaning instead of coercing invalid values into it."""

from __future__ import annotations

import random
from decimal import Decimal
from fractions import Fraction
from typing import Any, cast

import numpy as np
import pytest

from causal_hypergraphs.estimation import Dataset, DatasetError


@pytest.mark.parametrize(
    "missing", [None, float("nan"), float("inf"), -float("inf"), np.datetime64("NaT", "ns")]
)
def test_missing_or_nonfinite_categories_are_rejected_not_treated_as_extra_levels(
    missing: Any,
) -> None:
    with pytest.raises(DatasetError, match="missing|finite"):
        Dataset.from_records([{"X": 0}, {"X": missing}])
    with pytest.raises(DatasetError, match="missing|finite"):
        Dataset.from_records([{"X": 0}], domains={"X": (0, missing)})


@pytest.mark.parametrize("unit", [None, float("nan"), float("inf"), ("batch", None), [1]])
def test_invalid_independence_units_never_reach_bootstrap_grouping(unit: Any) -> None:
    with pytest.raises(DatasetError):
        Dataset.from_records([{"X": 0, "unit": unit}], unit="unit")


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "3.5", complex(1, 2)])
def test_measure_values_require_finite_real_numbers(value: Any) -> None:
    with pytest.raises(DatasetError, match="numeric measure"):
        Dataset.from_records([{"X": 0, "Y": value}], measures=("Y",))


@pytest.mark.parametrize(
    "second",
    [
        {"X": 0, "Y": 2},
        {"X": 0, "unit": "b"},
        {"Y": 2, "unit": "b"},
        {"X": 0, "Y": 2, "unit": "b", "extra": 3},
    ],
)
def test_ragged_records_name_schema_error_including_missing_units_and_measures(
    second: dict,
) -> None:
    with pytest.raises(DatasetError, match="Record 1.*schema"):
        Dataset.from_records([{"X": 0, "Y": 1, "unit": "a"}, second], measures=("Y",), unit="unit")


@pytest.mark.parametrize(
    "domains",
    [
        {},
        {"X": ()},
        {"X": (0, 0)},
        {"X": (False, 0)},
        {"X": "01"},
        {"X": (0,), "unused": (0,)},
        {"X": (1,)},
        {"X": ({"unhashable": 1},)},
    ],
)
def test_declared_domains_are_exact_nonempty_unique_and_cover_record_values(domains: dict) -> None:
    with pytest.raises(DatasetError):
        Dataset.from_records([{"X": 0}], domains=domains)


@pytest.mark.parametrize("count", [-1, 1.5, 1.0, True, float("nan"), None, "2"])
def test_contingency_counts_must_be_nonnegative_integers(count: Any) -> None:
    with pytest.raises(DatasetError, match="nonnegative integer"):
        Dataset.from_counts({(0,): count}, ("X",))


def test_counts_reject_bad_coordinates_and_declared_support_even_on_zero_cells() -> None:
    for counts in ({0: 1}, {(0, 1): 1}, {(None,): 1}):
        with pytest.raises(DatasetError):
            Dataset.from_counts(counts, ("X",))  # type: ignore[arg-type]
    with pytest.raises(DatasetError, match="outside"):
        Dataset.from_counts({(0,): 4, (2,): 0}, ("X",), domains={"X": (0, 1)})
    with pytest.raises(DatasetError, match="domains"):
        Dataset.from_counts({(0,): 4}, ("X",), domains={})
    with pytest.raises(DatasetError, match="all-zero"):
        Dataset.from_counts({(0,): 0}, ("X",))


def test_contingency_axis_order_and_numpy_integer_counts_preserve_exact_empirical_law() -> None:
    counts = {(0, 1): np.int64(3), (1, 0): np.int64(1)}
    data = Dataset.from_counts(cast(Any, counts), ("B", "A"), domains={"A": (0, 1, 2), "B": (0, 1)})
    assert data.variables == ("A", "B")
    law = data.empirical_joint(("A", "B"))
    assert law == {(0, 0): 0, (0, 1): 0.25, (1, 0): 0.75, (1, 1): 0, (2, 0): 0, (2, 1): 0}
    assert data.n_rows == data.n_units == 4


def test_direct_construction_snapshots_containers_and_reorders_rows_with_their_axes() -> None:
    domains = {"B": [0, 1], "A": [0, 1]}
    rows = [[1, 0], [0, 1]]
    units = ["first", "second"]
    measurements = {"score": [0.1, 0.9]}
    data = Dataset(
        variables=("B", "A"),
        domains=cast(Any, domains),
        rows=cast(Any, rows),
        units=cast(Any, units),
        measures=("score",),
        measurements=cast(Any, measurements),
    )
    rows[0][0] = 0
    units[0] = "changed"
    domains["A"].append(2)
    measurements["score"][0] = 99
    assert data.variables == ("A", "B")
    assert data.rows == ((0, 1), (1, 0))
    assert data.units == ("first", "second")
    assert data.domains["A"] == (0, 1)
    assert data.conditional_expectation("score", ("A",), {"A": 0}) == 0.1
    with pytest.raises(TypeError):
        data.domains["A"] = (0,)  # type: ignore[index]
    with pytest.raises(TypeError):
        data.measurements["score"] = (0, 0)  # type: ignore[index]


@pytest.mark.parametrize(
    "change",
    [
        {"variables": ("X", "X")},
        {"variables": (1,)},
        {"rows": ()},
        {"rows": ((0, 1),)},
        {"rows": ((None,),)},
        {"units": ()},
        {"units": (None,)},
        {"domains": {"X": ()}},
        {"measures": ("Y",)},
        {"measurements": {"Y": (1,)}},
        {"measures": ("X",), "measurements": {"X": (1,)}},
        {"unit_column": "X"},
    ],
)
def test_direct_constructor_cannot_bypass_ingestion_invariants(change: dict) -> None:
    arguments = {"variables": ("X",), "domains": {"X": (0, 1)}, "rows": ((0,),), "units": (0,)}
    with pytest.raises(DatasetError):
        Dataset(**{**arguments, **change})


def test_repeated_direct_unit_ids_require_a_group_label_in_the_report() -> None:
    with pytest.raises(DatasetError, match="named unit_column"):
        Dataset(("X",), {"X": (0, 1)}, ((0,), (1,)), ("same", "same"))


def test_measure_only_data_and_cluster_resampling_remain_valid() -> None:
    only_measure = Dataset.from_records(
        [{"score": Decimal("1.25")}, {"score": Fraction(7, 4)}], measures=("score",)
    )
    assert only_measure.variables == ()
    assert only_measure.conditional_expectation("score", (), {}) == 1.5
    data = Dataset.from_records(
        [
            {"X": 0, "score": 1, "batch": ("a", 1)},
            {"X": 0, "score": 3, "batch": ("a", 1)},
            {"X": 1, "score": 5, "batch": ("b", 2)},
        ],
        unit="batch",
        measures=("score",),
        domains={"X": (0, 1, 2)},
    )
    replicate = data.resample(random.Random(5))
    assert replicate.domains == data.domains
    assert replicate.n_units == 2
    assert len(replicate.measurements["score"]) == replicate.n_rows
    for label in set(replicate.units):
        points = [
            row for row, unit in zip(replicate.rows, replicate.units, strict=True) if unit == label
        ]
        assert len(set(points)) == 1  # whole clusters were selected together


def test_malformed_schema_and_container_inputs_raise_dataset_errors() -> None:
    for records in (None, [1], [{"": 0}], [{1: 0}], [{"X": []}]):
        with pytest.raises(DatasetError):
            Dataset.from_records(records)  # type: ignore[arg-type]
    with pytest.raises(DatasetError, match="duplicate"):
        Dataset.from_records([{"Y": 1}], measures=("Y", "Y"))


def test_estimation_diagnostics_respect_declared_order_of_incomparable_labels() -> None:
    from causal_hypergraphs import DeleteMechanism, MechanismGraph, estimate, identify

    graph = MechanismGraph(
        ("X", "Y"),
        {"target": {"outputs": ("X",)}, "response": {"inputs": ("X",), "outputs": ("Y",)}},
    )
    data = Dataset.from_counts(
        {(0, 0): 3, (0, 1): 3, ("high", 0): 2, ("high", 1): 2},
        ("X", "Y"),
        domains={"X": (0, "high"), "Y": (1, 0)},
    )
    result = identify(graph, DeleteMechanism("target", outcomes=("Y",)))
    estimated = estimate(result, data, fallbacks={"target": {(0,): 0.5, ("high",): 0.5}})
    assert estimated.values == {(0,): 0.5, (1,): 0.5}
    assert estimated.support.min_stratum_count == 2
    assert estimated.support.thinnest_stratum == {"X": "high", "Y": 1}
