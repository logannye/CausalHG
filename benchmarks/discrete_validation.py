"""Finite categorical estimation coverage and sparse-support negative control.

The population has an observed binary confounder Z, treatment T, and outcome Y.
P(Z=1)=0.5; P(T=1|Z)=0.2+0.6Z; P(Y=1|T,Z)=0.1+0.3T+0.4Z.
The intervention mean contrast is 0.3. Defaults and thresholds are fixed before
execution; every missing interval counts against unconditional coverage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal_hypergraphs import Dataset
from causal_hypergraphs.graph import MechanismGraph
from causal_hypergraphs.inference import compile_query, estimate_query
from causal_hypergraphs.queries import CausalQuery, EffectContrast, HardIntervention

THRESHOLDS = {"absolute_bias_max": 0.05, "coverage_min": 0.85, "failed_intervals_max": 0}


def run(replicates: int = 100, bootstrap: int = 99, seed: int = 20260928) -> dict[str, Any]:
    if replicates < 20 or bootstrap < 20:
        raise ValueError("Validation requires at least 20 datasets and 20 bootstrap refits.")
    graph = MechanismGraph(
        ("Z", "T", "Y"),
        {
            "treatment": {"inputs": "Z", "outputs": "T"},
            "outcome": {"inputs": ("T", "Z"), "outputs": "Y"},
        },
    )
    query = EffectContrast(
        CausalQuery(("Y",), HardIntervention({"T": 1}), "expectation"),
        CausalQuery(("Y",), HardIntervention({"T": 0}), "expectation"),
    )
    compiled = compile_query(graph, query)
    values = []
    covered, failed_intervals, failed_refits = 0, 0, 0
    for replicate in range(replicates):
        rng = np.random.default_rng(seed + replicate)
        z = rng.binomial(1, 0.5, 400)
        treatment = rng.binomial(1, 0.2 + 0.6 * z)
        outcome = rng.binomial(1, 0.1 + 0.3 * treatment + 0.4 * z)
        rows = [
            {"Z": int(a), "T": int(b), "Y": int(c)}
            for a, b, c in zip(z, treatment, outcome, strict=True)
        ]
        result = estimate_query(
            compiled, Dataset.from_records(rows), bootstrap=bootstrap, seed=seed + 10000 + replicate
        )
        if () not in result.values:
            raise AssertionError("Positive-support population produced an undefined contrast.")
        values.append(result.values[()])
        failed_refits += result.replicate_failures
        interval = result.interval.get(())
        if interval is None:
            failed_intervals += 1
        else:
            covered += interval[0] <= 0.3 <= interval[1]
    sparse = Dataset.from_records(
        [{"Z": z, "T": z, "Y": y} for _ in range(20) for z in (0, 1) for y in (0, 1)]
    )
    negative = estimate_query(compiled, sparse)
    sparse_passes = not negative.support.holds and () not in negative.values
    bias = float(np.mean(values)) - 0.3
    coverage = covered / replicates
    return {
        "seed": seed,
        "replicates": replicates,
        "bootstrap": bootstrap,
        "thresholds_declared_before_run": THRESHOLDS,
        "truth": 0.3,
        "bias": bias,
        "coverage": coverage,
        "coverage_monte_carlo_se": float(np.sqrt(coverage * (1 - coverage) / replicates)),
        "failed_intervals": failed_intervals,
        "failed_bootstrap_refits": failed_refits,
        "sparse_support_negative_control_passes": sparse_passes,
        "passes": abs(bias) <= THRESHOLDS["absolute_bias_max"]
        and coverage >= THRESHOLDS["coverage_min"]
        and failed_intervals <= THRESHOLDS["failed_intervals_max"]
        and sparse_passes,
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
