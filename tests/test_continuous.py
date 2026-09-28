from decimal import Decimal
from typing import Any

import numpy as np
import pytest

from causal_hypergraphs.estimation.continuous import (
    ContinuousBackend,
    ContinuousDataError,
    LinearGaussianKernel,
    RankDeficientFit,
    UnsupportedContinuousQuery,
    estimate_continuous,
    fit_linear_gaussian,
)
from causal_hypergraphs.graph import MechanismGraph
from causal_hypergraphs.queries import (
    CausalQuery,
    Composite,
    Delete,
    EffectContrast,
    HardIntervention,
    JointPolicy,
    Replace,
)


def _mean(outcome: str, intervention: Any, given: tuple[str, ...] = ()) -> CausalQuery:
    return CausalQuery((outcome,), intervention, "expectation", given)


def _simple_graph() -> MechanismGraph:
    return MechanismGraph(("X", "Y"), {"response": {"inputs": "X", "outputs": "Y"}})


def test_linear_mean_bootstrap_and_analytic_variance() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=800)
    y = 1 + 3 * x + rng.normal(scale=0.5, size=800)
    fitted = fit_linear_gaussian(
        _simple_graph(), ({"X": a, "Y": b} for a, b in zip(x, y, strict=True))
    )
    assert isinstance(fitted, ContinuousBackend)
    backend: ContinuousBackend = fitted
    estimate = estimate_continuous(
        backend, _mean("Y", HardIntervention({"X": 2})), bootstrap=99, seed=42
    )
    assert estimate.value == pytest.approx(7, abs=0.12)
    assert estimate.model_variance == pytest.approx(0.25, abs=0.04)
    assert estimate.interval is not None
    assert estimate.interval[0] <= 7 <= estimate.interval[1]
    assert estimate.bootstrap_failures == 0
    assert estimate.monte_carlo_error == 0
    assert estimate.numerical_error is None
    assert "Linear additive kernels" in {item.code for item in estimate.assumptions}


def test_partial_output_intervention_preserves_sibling_noise_and_mean() -> None:
    rng = np.random.default_rng(7)
    x, u, v = rng.normal(size=(3, 3000))
    records = [
        {"X": a, "C": 2 * a + b, "D": -a + b + 0.5 * c} for a, b, c in zip(x, u, v, strict=True)
    ]
    graph = MechanismGraph(("X", "C", "D"), {"joint": {"inputs": "X", "outputs": ("C", "D")}})
    fitted = fit_linear_gaussian(graph, records)
    observational = fitted.estimate(_mean("D", HardIntervention({})))
    intervened = fitted.estimate(_mean("D", HardIntervention({"C": 10})))
    assert intervened.value == pytest.approx(observational.value)
    assert intervened.model_variance == pytest.approx(observational.model_variance)
    assert fitted.kernels["joint"].covariance[0][1] == pytest.approx(1, abs=0.08)
    assert any("Extrapolation" in message for message in intervened.diagnostics)


def test_singular_joint_output_noise_is_allowed() -> None:
    records = [
        {"X": float(x), "A": 2.0 * x + u, "B": 2.0 * x + u}
        for x in range(-3, 4)
        for u in (-1.0, 1.0)
    ]
    graph = MechanismGraph(("X", "A", "B"), {"joint": {"inputs": "X", "outputs": ("A", "B")}})
    fitted = fit_linear_gaussian(graph, records)
    assert np.linalg.matrix_rank(fitted.kernels["joint"].covariance) == 1
    result = fitted.estimate(_mean("A", HardIntervention({"X": 1.5})))
    assert result.value == pytest.approx(3)
    assert result.model_variance is not None and result.model_variance > 0


def test_joint_policy_covariance_and_axis_order_are_preserved() -> None:
    rng = np.random.default_rng(11)
    a, b, u = rng.normal(size=(3, 2000))
    rows = [{"A": x, "B": y, "Y": x + y + 0.1 * error} for x, y, error in zip(a, b, u, strict=True)]
    graph = MechanismGraph(
        ("A", "B", "Y"),
        {"joint": {"outputs": ("A", "B")}, "sum": {"inputs": ("A", "B"), "outputs": "Y"}},
    )
    fitted = fit_linear_gaussian(graph, rows)
    coupled = JointPolicy(("B", "A"), {(-1, -1): 0.5, (1, 1): 0.5})
    opposed = JointPolicy(("B", "A"), {(-1, 1): 0.5, (1, -1): 0.5})
    first = fitted.estimate(_mean("Y", Delete("joint", coupled)))
    second = fitted.estimate(_mean("Y", Delete("joint", opposed)))
    assert first.model_variance is not None and second.model_variance is not None
    assert first.model_variance - second.model_variance == pytest.approx(4, abs=0.06)
    asymmetric = JointPolicy(("B", "A"), {(2, -1): 0.5, (4, 3): 0.5})
    assert fitted.estimate(_mean("Y", asymmetric)).value == pytest.approx(4, abs=0.04)


def test_replacement_can_bind_runtime_kernel_with_different_axis_order() -> None:
    from causal_hypergraphs.simulation import LinearGaussianMechanism

    rows = [
        {"X": float(x), "A": x + u, "B": 2 * x + v}
        for x in range(-3, 4)
        for u in (-1, 1)
        for v in (-1, 1)
    ]
    graph = MechanismGraph(("X", "A", "B"), {"joint": {"inputs": "X", "outputs": ("A", "B")}})
    replacement = LinearGaussianMechanism(
        inputs=("X",),
        outputs=("B", "A"),
        coefficients=((10,), (2,)),
        intercept=(1, 3),
        covariance=((9, 1), (1, 4)),
    )
    query = _mean("A", Composite((HardIntervention({"X": 1}), Replace("joint", "new"))))
    result = fit_linear_gaussian(graph, rows).estimate(query, replacements={"new": replacement})
    assert result.value == pytest.approx(5)
    assert result.model_variance == pytest.approx(4)


def test_gaussian_baseline_conditioning_uses_conditional_not_marginal_mean() -> None:
    rng = np.random.default_rng(91)
    x, z, u, v = rng.normal(size=(4, 10000))
    rows = [
        {"X": a, "Z": b, "M": b + c, "Y": 2 * a + 3 * b + d}
        for a, b, c, d in zip(x, z, u, v, strict=True)
    ]
    graph = MechanismGraph(
        ("X", "Z", "M", "Y"),
        {
            "baseline": {"inputs": "Z", "outputs": "M"},
            "response": {"inputs": ("X", "Z"), "outputs": "Y"},
        },
    )
    fitted = fit_linear_gaussian(graph, rows)
    query = _mean("Y", HardIntervention({"X": 1}), given=("M",))
    conditioned = fitted.estimate(query, given={"M": 1})
    assert conditioned.value == pytest.approx(3.5, abs=0.12)
    assert not any(
        "positive mass" in assumption.description for assumption in conditioned.assumptions
    )
    policy_query = _mean("Y", JointPolicy(("X",), {(0,): 0.5, (1,): 0.5}), given=("M",))
    with pytest.raises(UnsupportedContinuousQuery, match="mixtures"):
        fitted.estimate(policy_query, given={"M": 1})


def test_mixed_numeric_discrete_root_has_linear_mean_contract() -> None:
    rng = np.random.default_rng(55)
    rows = [
        {"X": x, "Y": 1 + 3 * x + error}
        for x, error in zip(rng.integers(0, 2, 1200), rng.normal(size=1200), strict=True)
    ]
    fitted = fit_linear_gaussian(_simple_graph(), rows, discrete=("X",))
    contrast = EffectContrast(
        _mean("Y", HardIntervention({"X": 1})), _mean("Y", HardIntervention({"X": 0}))
    )
    result = fitted.estimate(contrast)
    assert result.value == pytest.approx(3, abs=0.15)
    assert result.model_variance is None
    assert any("discrete roots" in item for item in result.diagnostics)
    with pytest.raises(UnsupportedContinuousQuery, match="observed support"):
        fitted.estimate(_mean("Y", HardIntervention({"X": 2})))


def test_paired_contrast_bootstrap_does_not_add_independent_arm_variances() -> None:
    rng = np.random.default_rng(2)
    rows = [
        {"X": x, "Y": 1 + 2 * x + error}
        for x, error in zip(rng.normal(size=100), rng.normal(size=100), strict=True)
    ]
    query = _mean("Y", HardIntervention({"X": 1}))
    result = fit_linear_gaussian(_simple_graph(), rows).estimate(
        EffectContrast(query, query), bootstrap=30, seed=4
    )
    assert result.value == 0
    assert result.interval == (0, 0)
    assert result.standard_error == 0


def test_cluster_bootstrap_retains_dependent_rows() -> None:
    rng = np.random.default_rng(6)
    rows = []
    for group in range(30):
        x, cluster_error = rng.normal(size=2)
        for _ in range(20):
            rows.append({"unit": group, "X": x, "Y": 2 * x + cluster_error + rng.normal(scale=0.1)})
    query = _mean("Y", HardIntervention({"X": 1}))
    clustered = fit_linear_gaussian(_simple_graph(), rows, unit="unit").estimate(
        query, bootstrap=70
    )
    independent = fit_linear_gaussian(_simple_graph(), rows).estimate(query, bootstrap=70)
    assert clustered.standard_error is not None and independent.standard_error is not None
    assert clustered.standard_error > 3 * independent.standard_error
    assert clustered.n_units == 30


def test_failed_bootstrap_refits_are_reported_without_conditional_success_interval() -> None:
    rows = [{"unit": x, "X": x, "Y": 2 * x + error} for x in (0, 1) for error in (-1, 1)]
    result = fit_linear_gaussian(_simple_graph(), rows, unit="unit").estimate(
        _mean("Y", HardIntervention({"X": 1})), bootstrap=20, seed=0
    )
    assert result.bootstrap_failures > 0
    assert result.interval is None
    assert result.bootstrap_failures + result.successful_replicates == 20


@pytest.mark.parametrize(
    "bad",
    [
        None,
        float("nan"),
        float("inf"),
        Decimal("NaN"),
        Decimal("Infinity"),
        np.datetime64("NaT", "ns"),
        (None,),
        frozenset({None}),
        [],
    ],
)
def test_missing_or_unstable_unit_identifiers_are_rejected(bad: Any) -> None:
    rows = [{"unit": bad, "X": float(x), "Y": 2.0 * x} for x in range(4)]
    with pytest.raises(ContinuousDataError, match="sampling-unit identifier"):
        fit_linear_gaussian(_simple_graph(), rows, unit="unit")


def test_pandas_missing_unit_is_rejected_when_pandas_is_available() -> None:
    pd = pytest.importorskip("pandas")
    rows = [{"unit": pd.NA, "X": float(x), "Y": 2.0 * x} for x in range(4)]
    with pytest.raises(ContinuousDataError, match="sampling-unit identifier"):
        fit_linear_gaussian(_simple_graph(), rows, unit="unit")


@pytest.mark.parametrize("bad", [None, "1.0", float("nan"), float("inf")])
def test_missing_and_nonfinite_numeric_data_are_rejected(bad: Any) -> None:
    with pytest.raises(ContinuousDataError):
        fit_linear_gaussian(_simple_graph(), [{"X": 1, "Y": bad}, {"X": 2, "Y": 3}])
    with pytest.raises(ContinuousDataError, match="missing variables"):
        fit_linear_gaussian(_simple_graph(), [{"X": 1}, {"X": 2, "Y": 3}])


def test_rank_deficient_predictors_are_not_fitted_by_silent_pseudoinverse() -> None:
    graph = MechanismGraph(("X", "Z", "Y"), {"response": {"inputs": ("X", "Z"), "outputs": "Y"}})
    with pytest.raises(RankDeficientFit):
        fit_linear_gaussian(graph, [{"X": x, "Z": 2 * x, "Y": x} for x in range(10)])


def test_unsupported_causal_models_and_post_treatment_conditions_are_refused() -> None:
    hidden = MechanismGraph(
        ("X", "Y"), {"response": {"inputs": "X", "outputs": "Y"}}, observed_variables="Y"
    )
    with pytest.raises(UnsupportedContinuousQuery, match="observed"):
        fit_linear_gaussian(hidden, [])
    graph = MechanismGraph(
        ("X", "M", "Y"),
        {"a": {"inputs": "X", "outputs": "M"}, "b": {"inputs": "M", "outputs": "Y"}},
    )
    rows = [
        {"X": float(x), "M": x + u, "Y": x + u + v}
        for x in range(-2, 3)
        for u in (-1, 1)
        for v in (-1, 1)
    ]
    fitted = fit_linear_gaussian(graph, rows)
    with pytest.raises(UnsupportedContinuousQuery, match="Baseline conditioning"):
        fitted.estimate(_mean("Y", HardIntervention({"X": 1}), given=("M",)), given={"M": 0})
    with pytest.raises(UnsupportedContinuousQuery, match="expectations"):
        fitted.estimate(CausalQuery(("Y",), HardIntervention({"X": 1})))


def test_nonlinear_misspecification_is_a_demonstrated_limit_not_an_identification_success() -> None:
    rng = np.random.default_rng(79)
    x = rng.uniform(-2, 2, 4000)
    rows = [
        {"X": value, "Y": value**2 + error}
        for value, error in zip(x, rng.normal(scale=0.1, size=len(x)), strict=True)
    ]
    result = fit_linear_gaussian(_simple_graph(), rows).estimate(
        _mean("Y", HardIntervention({"X": 1.8}))
    )
    assert abs(result.value - 1.8**2) > 1.5
    assert "Linear additive kernels" in {assumption.code for assumption in result.assumptions}


def test_kernel_rejects_invalid_covariance_and_keeps_declared_axis_order() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        LinearGaussianKernel((), ("Y",), ((),), (0,), ((-1,),))
    kernel = LinearGaussianKernel(("Z", "X"), ("Y",), ((2, 3),), (1,), ((0,),))
    assert kernel.inputs == ("Z", "X")


@pytest.mark.parametrize("unordered", [{"A", "B"}, frozenset({"A", "B"}), {"A": 1, "B": 2}])
def test_kernel_axes_must_have_explicit_order(unordered: Any) -> None:
    with pytest.raises(TypeError, match="ordered sequence"):
        LinearGaussianKernel(unordered, ("Y",), ((2, 3),), (0,), ((1,),))


def test_declared_output_equality_must_match_the_fitted_data() -> None:
    graph = MechanismGraph(
        ("X", "A", "B"),
        {"joint": {"inputs": "X", "outputs": ("A", "B"), "output_equalities": (("A", "B"),)}},
    )
    with pytest.raises(ContinuousDataError, match="declared output equality"):
        fit_linear_gaussian(graph, [{"X": 0, "A": 1, "B": 2}, {"X": 1, "A": 2, "B": 2}])


def test_singular_baseline_conditioning_is_refused_even_when_output_noise_is_valid() -> None:
    graph = MechanismGraph(
        ("X", "A", "B", "Y"),
        {"baseline": {"outputs": ("A", "B")}, "response": {"inputs": ("A", "X"), "outputs": "Y"}},
    )
    rows = [
        {"X": x, "A": a, "B": a, "Y": 2 * x + a + u}
        for x in (-2.0, 0.0, 2.0)
        for a in (-1.0, 1.0)
        for u in (-1.0, 1.0)
    ]
    fitted = fit_linear_gaussian(graph, rows)
    with pytest.raises(RankDeficientFit, match="Baseline covariance is singular"):
        fitted.estimate(_mean("Y", HardIntervention({"X": 1}), ("A", "B")), given={"A": 0, "B": 0})
