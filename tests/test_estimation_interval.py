"""The interval, and whether the unit of independence is doing anything.

A point estimate with no interval is not decision-usable, so the interval is part of the
deliverable rather than a later refinement. Two things about it need pinning.

*Calibration.* A 95% interval that covers the truth 60% of the time is worse than no
interval, because it invites confidence rather than withholding it. Coverage is therefore
measured against exact interventional laws and reported, not assumed from the fact that a
percentile bootstrap was used.

*The unit of independence.* `Dataset` makes the caller name it, and the estimate prints
it. That is only worth doing if getting it wrong changes the answer. The test at the
bottom builds data whose rows are perfectly correlated within unit and measures how much
narrower the interval becomes if you resample rows instead -- which is the mistake the
parameter exists to prevent.
"""
from __future__ import annotations

import random

import pytest

from causal_hypergraphs import DeleteMechanism, Identified, MechanismGraph, identify
from causal_hypergraphs.estimation import Dataset, estimate
from tests.conformance.generation import generate_model

BINARY = (0, 1)
LEVEL = 0.95

# Coverage is estimated from a finite number of trials, so the gate has to admit its own
# Monte Carlo error. 40 models x 8 points is enough to separate a calibrated interval from
# a badly miscalibrated one without flagging ordinary percentile-bootstrap conservatism.
COVERAGE_MODELS = 40
MIN_COVERAGE = 0.90
MAX_COVERAGE = 0.98
"""Two-sided, because calibration fails in both directions.

A floor alone passes an interval that is merely wide: covering 100% of the time means the
interval is uninformative, not that it is good. The measured value is 94.3% against a
nominal 95%, and both bounds sit close enough to catch a real regression -- the old 0.85
floor would have accepted a drop to 86% without a word.
"""


def _graph() -> MechanismGraph:
    return MechanismGraph(
        variables={"A", "C", "D"},
        mechanisms={"m1": {"inputs": ("A",), "outputs": ("C", "D")}},
    )


def _identified() -> Identified:
    result = identify(_graph(), DeleteMechanism("m1"))
    assert isinstance(result, Identified), result
    return result


P0_M1 = {(0, 0): 0.5, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.5}


def _records(rng: random.Random, n_rows: int) -> list[dict[str, int]]:
    return [
        {
            "A": 1 if rng.random() < 0.4 else 0,
            "C": 1 if rng.random() < 0.5 else 0,
            "D": 1 if rng.random() < 0.5 else 0,
        }
        for _ in range(n_rows)
    ]


def _width(estimate_result, point) -> float:
    bounds = estimate_result.interval[point]
    assert bounds is not None, point
    low, high = bounds
    return high - low


# --- basic shape ------------------------------------------------------------------


def test_no_bootstrap_means_no_interval_rather_than_a_fake_one() -> None:
    """Zero replicates must yield no interval at all, not a zero-width one."""
    data = Dataset.from_records(_records(random.Random(1), 400))
    est = estimate(_identified(), data, fallbacks={"m1": P0_M1})

    assert est.bootstrap == 0
    assert not est.interval


def test_the_interval_brackets_the_point_estimate() -> None:
    data = Dataset.from_records(_records(random.Random(2), 400))
    est = estimate(_identified(), data, fallbacks={"m1": P0_M1}, bootstrap=300, seed=11)

    for point, value in est.values.items():
        bounds = est.interval[point]
        assert bounds is not None
        low, high = bounds
        assert low <= value <= high, (point, value, bounds)


def test_the_interval_narrows_with_more_data() -> None:
    """The whole content of an interval is that it responds to sample size."""
    rng = random.Random(3)
    narrow = estimate(
        _identified(),
        Dataset.from_records(_records(rng, 8_000)),
        fallbacks={"m1": P0_M1},
        bootstrap=300,
        seed=5,
    )
    wide = estimate(
        _identified(),
        Dataset.from_records(_records(rng, 250)),
        fallbacks={"m1": P0_M1},
        bootstrap=300,
        seed=5,
    )

    point = (1, 1, 1)
    assert _width(narrow, point) < 0.5 * _width(wide, point)


# --- calibration ------------------------------------------------------------------


def test_bootstrap_coverage_is_near_nominal() -> None:
    """A 95% interval must actually contain the truth about 95% of the time.

    Measured against exact interventional laws from generated models, at every point of
    the estimand's scope. An interval that covered far less than nominal would be worse
    than none at all: it would license confidence the data do not support.
    """
    rng = random.Random(20260818)
    covered = 0
    total = 0

    for seed in range(COVERAGE_MODELS):
        model = generate_model(seed, allow_hidden=False, shapes=("positive",))
        spec = model.mechanisms[0]
        result = identify(model.graph(), DeleteMechanism(spec.name))
        assert isinstance(result, Identified)
        truth = model.interventional_delete(spec.name)

        counts = model.sample_counts(model.joint(), 1_500, rng)
        data = Dataset.from_counts(
            counts, model.variables, domains={v: BINARY for v in model.variables}
        )
        est = estimate(
            result,
            data,
            fallbacks={spec.name: model.fallbacks[spec.name]},
            bootstrap=200,
            level=LEVEL,
            seed=seed,
        )

        for point, bounds in est.interval.items():
            if bounds is None:
                continue
            low, high = bounds
            total += 1
            if low <= truth[point] <= high:
                covered += 1

    assert total > 200, f"only {total} interval(s) checked; the gate is nearly vacuous"
    coverage = covered / total
    assert MIN_COVERAGE <= coverage <= MAX_COVERAGE, (
        f"95% intervals covered the truth {coverage:.1%} of the time over {total} points, "
        f"outside [{MIN_COVERAGE:.0%}, {MAX_COVERAGE:.0%}]. Under-coverage licenses "
        f"confidence the data do not support; over-coverage means the interval is wide "
        f"enough to be uninformative. Both are calibration failures."
    )


# --- the unit of independence -----------------------------------------------------


def test_declaring_the_unit_of_independence_widens_the_interval() -> None:
    """The parameter is load-bearing: getting it wrong understates uncertainty badly.

    Here every row within a donor is identical, so 20 donors carry exactly 20 independent
    observations even though the table has 1,000 rows. Resampling rows would treat it as
    1,000, and the resulting interval is narrower by roughly the square root of the rows
    per donor. If both spellings gave the same answer, `unit=` would be decoration and the
    default (each row independent) would be harmless -- it is not.
    """
    rng = random.Random(7)
    rows_per_donor = 50
    donors = 20

    records: list[dict[str, object]] = []
    for donor in range(donors):
        shared: dict[str, object] = {
            "A": 1 if rng.random() < 0.4 else 0,
            "C": 1 if rng.random() < 0.5 else 0,
            "D": 1 if rng.random() < 0.5 else 0,
        }
        records.extend(dict(shared, donor=f"d{donor}") for _ in range(rows_per_donor))

    by_unit = Dataset.from_records(records, unit="donor")
    by_row = Dataset.from_records(records)  # the mistake: rows treated as independent

    assert by_unit.n_units == donors
    assert by_row.n_units == donors * rows_per_donor

    honest = estimate(_identified(), by_unit, fallbacks={"m1": P0_M1}, bootstrap=400, seed=3)
    overconfident = estimate(
        _identified(), by_row, fallbacks={"m1": P0_M1}, bootstrap=400, seed=3
    )

    point = max(honest.values, key=lambda key: honest.values[key])
    honest_width = _width(honest, point)
    overconfident_width = _width(overconfident, point)

    # Two-sided, and centred on the quantity theory predicts rather than on "bigger".
    # Every row within a donor is identical here, so 50 rows carry one donor's worth of
    # information and the widths should differ by about sqrt(50) = 7.07. A one-sided
    # `> 3.0x` gate would pass a ratio of 3 -- half the predicted value -- and call it
    # confirmation. Measured: 6.56x.
    ratio = honest_width / overconfident_width
    assert 4.5 < ratio < 9.5, (
        f"unit bootstrap width {honest_width:.4f} vs row bootstrap "
        f"{overconfident_width:.4f} is a ratio of {ratio:.2f}, away from the "
        f"sqrt({rows_per_donor}) = {rows_per_donor ** 0.5:.2f} the within-donor "
        f"correlation implies"
    )
    assert "donor" in honest.unit
    assert "row" in overconfident.unit


def test_a_replicate_that_empties_a_stratum_is_counted_not_dropped() -> None:
    """Thin strata vanish under resampling; that rate is a finding, not an inconvenience.

    Silently computing the interval from whichever replicates happened to survive would
    report a narrow interval precisely where the estimate is least trustworthy.
    """
    records: list[dict[str, int]] = []
    for a in BINARY:
        for c in BINARY:
            for e in BINARY:
                for f in BINARY:
                    repeats = 1 if (c == 1 and e == 1) else 30
                    records.extend({"A": a, "C": c, "E": e, "F": f} for _ in range(repeats))

    graph = MechanismGraph(
        variables={"A", "C", "E", "F"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("C",)},
            "m2": {"inputs": ("C", "E"), "outputs": ("F",)},
        },
    )
    result = identify(graph, DeleteMechanism("m1"))
    assert isinstance(result, Identified)

    est = estimate(
        result,
        Dataset.from_records(records),
        fallbacks={"m1": {(0,): 0.5, (1,): 0.5}},
        bootstrap=200,
        seed=17,
    )

    assert est.support.holds  # defined in the observed sample...
    assert est.replicate_failures > 0  # ...but not in every resample of it
    assert "replicate-point(s) undefined" in est.summary()


@pytest.mark.parametrize("level", [0.5, 0.8, 0.99])
def test_the_requested_level_changes_the_width(level: float) -> None:
    """A `level` argument that did not reach the percentile computation would be silent."""
    data = Dataset.from_records(_records(random.Random(4), 600))
    est = estimate(
        _identified(), data, fallbacks={"m1": P0_M1}, bootstrap=300, level=level, seed=9
    )
    baseline = estimate(
        _identified(), data, fallbacks={"m1": P0_M1}, bootstrap=300, level=0.95, seed=9
    )

    point = (1, 1, 1)
    if level < 0.95:
        assert _width(est, point) < _width(baseline, point)
    else:
        assert _width(est, point) > _width(baseline, point)
