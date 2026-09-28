"""Predefined numerical validation populations for the scoped continuous backend.

Run from an installed checkout: python benchmarks/continuous_validation.py
The defaults (100 datasets, 99 bootstrap refits each) and thresholds below were
declared before the first run. They are regression criteria, not universal
statistical guarantees. Aggregates count every dataset and interval failure, with
a nonlinear coefficient sweep that deliberately violates the model assumption.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np

from causal_hypergraphs.estimation.continuous import fit_linear_gaussian
from causal_hypergraphs.graph import MechanismGraph
from causal_hypergraphs.queries import CausalQuery, EffectContrast, HardIntervention

THRESHOLDS = {
    "absolute_bias_max": 0.15,
    "coverage_min": 0.85,
    "coverage_max": 1.0,
    "failed_intervals_max": 0,
    "nonlinear_absolute_bias_min": 1.3,
}
POPULATIONS = ("gaussian", "mixed", "clustered", "null")


def _population(name: str, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    slope = 0.0 if name == "null" else 2.0
    groups, per_group = (40, 10) if name == "clustered" else (400, 1)
    for group in range(groups):
        x = float(rng.binomial(1, 0.45)) if name == "mixed" else float(rng.normal())
        z, shared = rng.normal(size=2)
        for _ in range(per_group):
            residual = shared + rng.normal(scale=0.2)
            y = 1 + slope * x + 0.7 * z + residual
            sibling = -0.5 * x + 0.3 * z + 0.6 * residual + rng.normal(scale=0.8)
            rows.append({"X": x, "Z": z, "Y": y, "S": sibling, "unit": group})
    return rows


def run(replicates: int = 100, bootstrap: int = 99, seed: int = 20260928) -> dict[str, Any]:
    if replicates < 20 or bootstrap < 20:
        raise ValueError("Validation requires at least 20 datasets and 20 bootstrap refits.")
    graph = MechanismGraph(
        ("X", "Z", "Y", "S"),
        {"joint": {"inputs": ("X", "Z"), "outputs": ("Y", "S")}},
    )
    query = EffectContrast(
        CausalQuery(("Y",), HardIntervention({"X": 1.0}), "expectation"),
        CausalQuery(("Y",), HardIntervention({"X": 0.0}), "expectation"),
    )
    populations = {}
    for index, name in enumerate(POPULATIONS):
        truth = 0.0 if name == "null" else 2.0
        values, covered = [], 0
        failed_intervals, failed_refits = 0, 0
        for replicate in range(replicates):
            case_seed = seed + index * 10000 + replicate
            fitted = fit_linear_gaussian(
                graph,
                _population(name, case_seed),
                unit="unit" if name == "clustered" else None,
                discrete=("X",) if name == "mixed" else (),
            )
            result = fitted.estimate(query, bootstrap=bootstrap, seed=case_seed + 700000)
            values.append(result.value)
            failed_refits += result.bootstrap_failures
            if result.interval is None:
                failed_intervals += 1
            else:
                covered += result.interval[0] <= truth <= result.interval[1]
        bias = float(np.mean(values)) - truth
        coverage = covered / replicates
        populations[name] = {
            "truth": truth,
            "bias": bias,
            "coverage": coverage,
            "coverage_monte_carlo_se": float(np.sqrt(coverage * (1 - coverage) / replicates)),
            "failed_intervals": failed_intervals,
            "failed_bootstrap_refits": failed_refits,
            "passes": abs(bias) <= THRESHOLDS["absolute_bias_max"]
            and THRESHOLDS["coverage_min"] <= coverage <= THRESHOLDS["coverage_max"]
            and failed_intervals <= THRESHOLDS["failed_intervals_max"],
        }
    # Declared-assumption sensitivity, with known synthetic truth. This sweep is
    # a diagnostic benchmark, not an estimator that repairs nonlinearity.
    simple = MechanismGraph(("X", "Y"), {"response": {"inputs": "X", "outputs": "Y"}})
    nonlinear_query = CausalQuery(("Y",), HardIntervention({"X": 1.8}), "expectation")
    sweep = []
    for coefficient in (0.0, 0.25, 0.5, 1.0):
        errors = []
        truth = 1 + 2 * 1.8 + coefficient * 1.8**2
        for replicate in range(replicates):
            rng = np.random.default_rng(seed + 900000 + replicate)
            x = rng.uniform(-2, 2, 600)
            noise = rng.normal(scale=0.5, size=len(x))
            rows = [
                {"X": value, "Y": 1 + 2 * value + coefficient * value**2 + error}
                for value, error in zip(x, noise, strict=True)
            ]
            result = fit_linear_gaussian(simple, rows).estimate(nonlinear_query)
            errors.append(result.value - truth)
        sweep.append(
            {"quadratic_coefficient": coefficient, "truth": truth, "bias": float(np.mean(errors))}
        )
    negative_pass = abs(sweep[-1]["bias"]) >= THRESHOLDS["nonlinear_absolute_bias_min"]
    return {
        "seed": seed,
        "replicates": replicates,
        "bootstrap": bootstrap,
        "nominal_coverage": 0.95,
        "thresholds_declared_before_run": THRESHOLDS,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "populations": populations,
        "nonlinear_assumption_sensitivity": sweep,
        "nonlinear_negative_control_passes": negative_pass,
        "passes": all(value["passes"] for value in populations.values()) and negative_pass,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=99)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.replicates, args.bootstrap, args.seed)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
    raise SystemExit(0 if report["passes"] else 1)
