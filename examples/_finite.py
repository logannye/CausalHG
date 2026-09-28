"""Shared execution for the exact, synthetic examples; no data downloads or path edits."""

from __future__ import annotations

import json
from fractions import Fraction
from math import isclose, lcm

from causal_hypergraphs import (
    Dataset,
    DeleteMechanism,
    Identified,
    MechanismGraph,
    ReplaceMechanism,
    estimate,
    identify,
)


def bernoulli(value: int, probability: Fraction) -> Fraction:
    return probability if value else 1 - probability


def run_example(
    *,
    name: str,
    graph: MechanismGraph,
    query: DeleteMechanism | ReplaceMechanism,
    joint: dict[tuple[int, ...], Fraction],
    expected: Fraction,
    fallbacks: dict | None = None,
    replacements: dict | None = None,
) -> dict:
    """Estimate an exact empirical law and compare with independently supplied arithmetic.

    The caller builds its observational joint directly from the synthetic generating
    model. Integer frequencies represent that law exactly, so this demonstration has
    no Monte Carlo error. Rows are independent units in these deliberately artificial
    datasets; a real dataset must declare its actual unit of independence.
    """
    if sum(joint.values()) != 1:
        raise ValueError("The synthetic generating law must sum to one.")
    total = lcm(*(probability.denominator for probability in joint.values()))
    variables = tuple(sorted(graph.variable_set))
    rows = []
    for values, probability in joint.items():
        record = dict(zip(variables, values, strict=True))
        rows.extend(dict(record) for _ in range(int(probability * total)))
    data = Dataset.from_records(rows, domains={variable: (0, 1) for variable in variables})
    identified = identify(graph, query)
    if not isinstance(identified, Identified):
        raise RuntimeError(f"Expected a supported example, received {identified!r}.")
    result = estimate(identified, data, fallbacks=fallbacks, replacements=replacements)
    reference = estimate(
        identified, data, fallbacks=fallbacks, replacements=replacements, method="enumerate"
    )
    value = result.values[(1,)]
    if not isclose(value, float(expected), abs_tol=1e-12, rel_tol=0):
        raise AssertionError(f"{name}: estimated {value}, expected {expected}.")
    if not isclose(value, reference.values[(1,)], abs_tol=1e-12, rel_tol=0):
        raise AssertionError("Elimination and enumeration disagree.")
    return {
        "example": name,
        "outcome": result.variables[0],
        "probability": value,
        "expected_fraction": str(expected),
        "expression": identified.expression.render(),
        "theorem": identified.theorem,
        "observational_factors": sum(
            kernel.kind == "probability" for kernel in identified.expression.kernels()
        ),
        "rows": result.n_rows,
        "support_holds": result.support.holds,
        "total_probability": sum(result.values.values()),
        "max_entries": result.plan.max_entries if result.plan else None,
    }


def print_report(report: dict) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))
