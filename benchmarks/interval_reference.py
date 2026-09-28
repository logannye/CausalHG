"""Independent interval references for the predefined validation populations.

This diagnostic uses NumPy and SciPy, never CausalHG's fit, bootstrap, or inference
implementations. Install SciPy separately to reproduce it; ordinary library use
does not require SciPy. Populations deliberately match continuous_validation.py
and discrete_validation.py. The clustered population has Gaussian homogeneous
unit errors and identical predictors within each unit, so unit-mean OLS with a
Student t interval is an exact conditional reference for that population only.
The categorical influence-function Wald interval remains asymptotic.

Example: python benchmarks/interval_reference.py --seed 73192026 --replicates 300
No pass/fail threshold is fitted to these results. Coverage and its Monte Carlo
uncertainty diagnose the percentile-bootstrap results, not certify calibration.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.stats import binomtest, norm, t

POPULATIONS = ("gaussian", "mixed", "clustered", "null")


def _linear_population(name: str, seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Reproduce the benchmark generator without invoking the library's model."""
    rng = np.random.default_rng(seed)
    slope = 0.0 if name == "null" else 2.0
    groups, per_group = (40, 10) if name == "clustered" else (400, 1)
    predictors, responses = [], []
    for _ in range(groups):
        x = float(rng.binomial(1, 0.45)) if name == "mixed" else float(rng.normal())
        z, shared = rng.normal(size=2)
        outcomes = []
        for _ in range(per_group):
            residual = shared + rng.normal(scale=0.2)
            outcomes.append(1 + slope * x + 0.7 * z + residual)
            # Match the joint-output generator's draw for its second output.
            rng.normal(scale=0.8)
        predictors.append([1, x, z])
        responses.append(float(np.mean(outcomes)))
    return np.asarray(predictors), np.asarray(responses), per_group


def _summary(
    estimates: list[float], errors: list[float], covered: int, truth: float, method: str
) -> dict[str, Any]:
    count = len(estimates)
    coverage = covered / count
    test = binomtest(covered, count, 0.95, alternative="less")
    interval = binomtest(covered, count).proportion_ci(confidence_level=0.95, method="exact")
    return {
        "truth": truth,
        "coverage": coverage,
        "covered": covered,
        "misses": count - covered,
        "coverage_monte_carlo_se": float(np.sqrt(coverage * (1 - coverage) / count)),
        "coverage_exact_binomial_95_interval": [float(interval.low), float(interval.high)],
        "one_sided_p_for_undercoverage_of_095": float(test.pvalue),
        "bias": float(np.mean(estimates)) - truth,
        "empirical_standard_deviation": float(np.std(estimates, ddof=1)),
        "mean_reference_standard_error": float(np.mean(errors)),
        "method": method,
    }


def run(seed: int = 73192026, replicates: int = 300) -> dict[str, Any]:
    if replicates < 20:
        raise ValueError("The reference diagnostic requires at least 20 datasets.")
    populations = {}
    for index, name in enumerate(POPULATIONS):
        estimates, errors, covered = [], [], 0
        truth = 0.0 if name == "null" else 2.0
        for replicate in range(replicates):
            design, outcome, _ = _linear_population(name, seed + index * 10000 + replicate)
            gram = design.T @ design
            coefficients = np.linalg.solve(gram, design.T @ outcome)
            residual = outcome - design @ coefficients
            degrees = len(outcome) - design.shape[1]
            covariance = np.linalg.inv(gram) * float(residual @ residual / degrees)
            error = float(np.sqrt(covariance[1, 1]))
            critical = float(t.ppf(0.975, degrees))
            value = float(coefficients[1])
            covered += int(value - critical * error <= truth <= value + critical * error)
            estimates.append(value)
            errors.append(error)
        populations[name] = _summary(
            estimates,
            errors,
            covered,
            truth,
            "independent-unit homoskedastic OLS with exact Student t interval "
            "under this population",
        )

    estimates, errors, covered = [], [], 0
    for replicate in range(replicates):
        rng = np.random.default_rng(seed + replicate)
        z = rng.binomial(1, 0.5, 400)
        treatment = rng.binomial(1, 0.2 + 0.6 * z)
        outcome = rng.binomial(1, 0.1 + 0.3 * treatment + 0.4 * z)
        means = {
            (a, c): float(outcome[(treatment == a) & (z == c)].mean())
            for a in (0, 1)
            for c in (0, 1)
        }
        propensities = {c: float(treatment[z == c].mean()) for c in (0, 1)}
        effects = np.asarray([means[1, c] - means[0, c] for c in z])
        value = float(effects.mean())
        influence = (
            effects
            - value
            + np.asarray(
                [
                    (y - means[1, c]) / propensities[c]
                    if a
                    else -(y - means[0, c]) / (1 - propensities[c])
                    for a, c, y in zip(treatment, z, outcome, strict=True)
                ]
            )
        )
        error = float(np.std(influence, ddof=1) / np.sqrt(len(z)))
        critical = float(norm.ppf(0.975))
        covered += int(value - critical * error <= 0.3 <= value + critical * error)
        estimates.append(value)
        errors.append(error)
    populations["categorical"] = _summary(
        estimates, errors, covered, 0.3, "saturated influence-function asymptotic Wald interval"
    )
    return {
        "seed": seed,
        "replicates": replicates,
        "nominal_coverage": 0.95,
        "rows_per_dataset": 400,
        "cluster_units_per_dataset": 40,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "populations": populations,
        "interpretation": "Independent comparison; not a calibrated-coverage release certificate.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=73192026)
    parser.add_argument("--replicates", type=int, default=300)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.seed, args.replicates)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)
