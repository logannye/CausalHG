"""Uncertainty requests must not manufacture an interval from invalid controls."""

from typing import Any, cast

import pytest

from causal_hypergraphs import (
    CausalQuery,
    Dataset,
    HardIntervention,
    MechanismGraph,
    compile_query,
    estimate_query,
)
from causal_hypergraphs.estimation import DatasetError


def _query():
    graph = MechanismGraph(
        ("X", "Y"), {"mechanism": {"inputs": ("X",), "outputs": ("Y",)}}
    )
    return compile_query(graph, CausalQuery(("Y",), HardIntervention({"X": 1})))


def _data():
    return Dataset.from_records([{"X": 0, "Y": 0}, {"X": 1, "Y": 1}])


@pytest.mark.parametrize("bootstrap", [-1, 1.5, True, "20", None])
def test_invalid_replicate_counts_are_rejected(bootstrap):
    with pytest.raises(ValueError, match="bootstrap must be a nonnegative integer"):
        estimate_query(_query(), _data(), bootstrap=cast(Any, bootstrap))


@pytest.mark.parametrize("level", [0, 1, -0.1, 1.1, float("nan"), float("inf"), "95%", None])
def test_invalid_confidence_levels_are_rejected_even_without_resampling(level):
    with pytest.raises(ValueError, match="level must be strictly between"):
        estimate_query(_query(), _data(), level=cast(Any, level))


def test_one_successful_draw_does_not_define_an_interval():
    # Seed 4 draws both strata on the first replicate, so it is fully evaluable.
    result = estimate_query(_query(), _data(), bootstrap=1, seed=4)
    assert result.replicate_failures == 0
    assert result.interval == {(0,): None, (1,): None}
    assert "fewer than two successful bootstrap draws" in result.summary()


def test_one_surviving_draw_after_empty_strata_is_reported_without_interval():
    # Seed 1 draws X=(0,0), then X=(1,0). The first replicate has no X=1 stratum.
    result = estimate_query(_query(), _data(), bootstrap=2, seed=1)
    assert result.replicate_failures == 2  # both outcome points in the first replicate
    assert result.interval == {(0,): None, (1,): None}
    assert "2 replicate-point(s) undefined" in result.summary()


def test_one_sampling_unit_cannot_supply_bootstrap_uncertainty():
    data = Dataset.from_records(
        [{"X": 0, "Y": 0, "unit": "only"}, {"X": 1, "Y": 1, "unit": "only"}],
        unit="unit",
    )
    assert estimate_query(_query(), data).values == {(0,): 0.0, (1,): 1.0}
    with pytest.raises(DatasetError, match="two independent sampling units"):
        estimate_query(_query(), data, bootstrap=20)
