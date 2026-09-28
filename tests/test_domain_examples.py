"""Small synthetic end-to-end contracts formerly exercised by the dataset demo.

The examples have independent numeric ground truth. The remaining regressions keep
the generic feedback, covariate, clustered-unit, support, and policy diagnostics
without downloading or pinning a particular scientific dataset.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from fractions import Fraction
from itertools import product
from pathlib import Path

import pytest

from causal_hypergraphs import (
    Dataset,
    DeleteMechanism,
    Identified,
    MechanismGraph,
    Unknown,
    check_covariates,
    estimate,
    identify,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize(
    ("script", "expected", "theorem", "factors"),
    [
        ("software_pipeline.py", Fraction(11, 40), "T2", 2),
        ("manufacturing.py", Fraction(67, 128), "T3", 3),
        ("policy.py", Fraction(19, 32), "T2", 2),
    ],
)
def test_examples_use_installed_imports_and_match_known_laws(
    script: str, expected: Fraction, theorem: str, factors: int, tmp_path: Path,
) -> None:
    # Outside the repository and without PYTHONPATH: this must import an installed
    # package, not quietly find src/ through a checkout-specific path manipulation.
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / script)], cwd=tmp_path, env=environment,
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["probability"] == pytest.approx(float(expected), abs=1e-12)
    assert report["expected_fraction"] == str(expected)
    assert report["total_probability"] == pytest.approx(1.0)
    assert report["support_holds"]
    assert report["theorem"] == theorem
    assert report["observational_factors"] == factors
    assert report["max_entries"] <= 16


def test_time_expansion_preserves_feedback_edges_and_yields_an_exact_multistep_query() -> None:
    edges = (("x", "y"), ("y", "x"), ("y", "readout"))
    simultaneous = MechanismGraph(
        variables={"x", "y", "readout"},
        mechanisms={
            target: {"inputs": (source,), "outputs": (target,)} for source, target in edges
        },
    )
    refusal = identify(simultaneous, DeleteMechanism("x", outcomes=("readout",)))
    assert isinstance(refusal, Unknown)

    mechanisms = {
        f"{target}_{step}": {
            "inputs": (f"{source}_{step - 1}",), "outputs": (f"{target}_{step}",),
        }
        for step in range(1, 4) for source, target in edges
    }
    expanded = MechanismGraph(
        variables={f"{name}_{step}" for name in simultaneous.variable_set for step in range(4)},
        mechanisms=mechanisms,
    )
    assert len(expanded.mechanisms) == len(edges) * 3
    assert expanded.is_mechanism_acyclic()
    for step in range(1, 4):
        for source, target in edges:
            mechanism = expanded.get_mechanism(f"{target}_{step}")
            assert mechanism.inputs == (f"{source}_{step - 1}",)
            assert mechanism.outputs == (f"{target}_{step}",)

    result = identify(expanded, DeleteMechanism("x_1", outcomes=("readout_3",)))
    assert isinstance(result, Identified)
    assert result.expression.footprint() == frozenset({"x_1", "y_2", "readout_3"})
    assert sum(kernel.kind == "probability" for kernel in result.expression.kernels()) == 2

    rows = []
    for x, y, readout in product((0, 1), repeat=3):
        p_y = Fraction(1, 4) + Fraction(1, 2) * x
        p_readout = Fraction(1, 10) + Fraction(4, 5) * y
        mass = Fraction(1, 2) * (p_y if y else 1 - p_y)
        mass *= p_readout if readout else 1 - p_readout
        rows.extend({"x_1": x, "y_2": y, "readout_3": readout} for _ in range(int(mass * 80)))
    estimated = estimate(
        result, Dataset.from_records(rows), fallbacks={"x_1": {(0,): 0.25, (1,): 0.75}},
    )
    # The assumed lagged process gives E[y_2]=1/4+(1/2)(3/4)=5/8,
    # then P(readout_3=1)=1/10+(4/5)(5/8)=3/5.
    assert estimated.values[(1,)] == pytest.approx(0.6)
    assert estimated.support.holds
    assert estimated.plan is not None and estimated.plan.max_entries <= 8


def _diagnostic_graph() -> MechanismGraph:
    return MechanismGraph(
        variables={"context", "treatment", "mediator", "outcome"},
        mechanisms={
            "target": {"inputs": ("context",), "outputs": ("treatment",)},
            "middle": {"inputs": ("treatment",), "outputs": ("mediator",)},
            "last": {"inputs": ("mediator",), "outputs": ("outcome",)},
        },
    )


def _diagnostic_rows() -> list[dict[str, int]]:
    rows = []
    for treatment, count in ((0, 72), (1, 8)):
        for index in range(count):
            rows.append({
                "unit": index % 4, "context": index % 2, "treatment": treatment,
                "mediator": index % 2, "outcome": (index // 2) % 2,
            })
    return rows


def test_joint_diagnostics_keep_positive_support_distinct_from_sparse_policy_support() -> None:
    graph = _diagnostic_graph()
    query = DeleteMechanism("target", outcomes=("outcome",))
    result = identify(graph, query)
    assert isinstance(result, Identified)
    estimated = estimate(
        result, Dataset.from_records(_diagnostic_rows(), unit="unit"),
        fallbacks={"target": {(0,): 0.25, (1,): 0.75}},
    )
    assert estimated.n_rows == 80 and estimated.n_units == 4
    assert estimated.support.holds
    assert estimated.support.min_stratum_count == 4
    (policy,) = estimated.policy
    assert policy.effective_n == pytest.approx(576 / 41)
    assert not policy.holds
    assert estimated.values[(1,)] == pytest.approx(0.5)
    covariates = check_covariates(graph, query, "outcome", ("context", "mediator"))
    assert covariates.admissible == ("context",)
    assert covariates.post_treatment == ("mediator",)


def test_declared_but_unobserved_conditioning_level_names_a_support_failure() -> None:
    graph = _diagnostic_graph()
    result = identify(graph, DeleteMechanism("target", outcomes=("outcome",)))
    rows = [dict(row, treatment=2 * row["treatment"]) for row in _diagnostic_rows()]
    data = Dataset.from_records(
        rows, unit="unit", domains={
            "context": (0, 1), "treatment": (0, 1, 2), "mediator": (0, 1), "outcome": (0, 1),
        },
    )
    estimated = estimate(
        result, data, fallbacks={"target": {(0,): 0.25, (1,): 0.0, (2,): 0.75}},
    )
    assert not estimated.support.holds
    assert estimated.support.points_undefined == estimated.support.points_total == 2
    assert not estimated.values
    assert any(failure.stratum == {"treatment": 1} for failure in estimated.support.failures)
