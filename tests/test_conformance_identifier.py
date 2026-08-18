"""Randomized conformance: every identified estimand must equal the interventional law.

The property, stated precisely:

    For a model satisfying C1-C4, if `identify` returns `Identified`, then evaluating its
    estimand against the observational law reproduces the interventional law at every
    point that law gives positive mass -- unless a positivity assumption the result
    *itself records* fails in this model, in which case the query is skipped and counted.

The escape clause is not a loophole, it is the contract: positivity is semantic and not
checkable from incidence, so the compiler records it as a certificate for the caller to
discharge. What is not permitted is an estimand that cannot be evaluated while recording
no such certificate -- that is an undisclosed requirement, and it is exactly the defect
this harness exists to prevent from recurring.
"""
from __future__ import annotations

import pytest

from causal_hypergraphs import (
    DeleteMechanism,
    Identified,
    ReplaceMechanism,
    Unknown,
    identify,
)
from causal_hypergraphs.semantics import DiscreteModel
from tests.conformance.checks import check_estimand
from tests.conformance.generation import RandomModel, generate_model

MODEL_COUNT = 220

# Assumptions whose failure legitimately makes an estimand unevaluable.
POSITIVITY_CODES = frozenset({"Target positivity", "Downstream positivity"})


def _discrete_model(model: RandomModel, target: str, *, with_replacement: bool) -> DiscreteModel:
    """Build the evaluation model over the *observed* variables only."""
    joint = model.marginalize_to_observed(model.joint())
    replacements = (
        {f"{target}_prime": model.replacement_table(target)} if with_replacement else {}
    )
    return DiscreteModel(
        domains=model.observed_domains,
        joint=joint,
        fallbacks=model.fallbacks,
        replacements=replacements,
    )


def _records_positivity(result: Identified) -> bool:
    return any(item.code in POSITIVITY_CODES for item in result.assumptions)


@pytest.fixture(scope="module")
def sweep() -> dict[str, int]:
    """Run the sweep once and let several tests assert different things about it."""
    tally = {
        "queries": 0,
        "identified": 0,
        "verified": 0,
        "skipped_positivity": 0,
        "refused": 0,
        "hidden_variable_models": 0,
        "non_factorizing_deletions": 0,
    }
    failures: list[str] = []

    for seed in range(MODEL_COUNT):
        model = generate_model(seed)
        graph = model.graph()
        if len(model.observed) < len(model.variables):
            tally["hidden_variable_models"] += 1
        tally["non_factorizing_deletions"] += len(model.non_factorizing_fallbacks())

        for spec in model.mechanisms:
            cases = (
                (DeleteMechanism(spec.name), model.interventional_delete(spec.name), False),
                (
                    ReplaceMechanism(spec.name, f"{spec.name}_prime"),
                    model.interventional_replace(spec.name),
                    True,
                ),
            )
            for query, full_truth, with_replacement in cases:
                tally["queries"] += 1
                result = identify(graph, query)

                if isinstance(result, Unknown):
                    tally["refused"] += 1
                    continue
                assert isinstance(result, Identified), f"seed {seed}: unexpected {result!r}"
                tally["identified"] += 1

                discrete = _discrete_model(model, spec.name, with_replacement=with_replacement)
                truth = model.marginalize_to_observed(full_truth)
                report = check_estimand(result.expression, discrete, truth, model.observed)

                where = f"seed {seed} / {spec.name} ({spec.shape}) / {result.theorem}"
                if report.mismatches or report.nonzero_where_truth_zero:
                    failures.append(f"{where}: {report.summary()}")
                elif report.undefined_somewhere:
                    if not _records_positivity(result):
                        failures.append(
                            f"{where}: estimand unevaluable but no positivity assumption "
                            f"recorded -- {report.summary()}"
                        )
                    else:
                        tally["skipped_positivity"] += 1
                else:
                    tally["verified"] += 1

    tally["failures"] = len(failures)  # type: ignore[assignment]
    if failures:
        pytest.fail(
            f"{len(failures)} nonconforming estimand(s) out of {tally['identified']}:\n  "
            + "\n  ".join(failures[:10])
        )
    return tally


def test_every_identified_estimand_reproduces_its_interventional_law(sweep) -> None:
    """The sweep itself fails on the first nonconforming estimand; this pins the counts."""
    assert sweep["identified"] > 0
    assert sweep["verified"] + sweep["skipped_positivity"] == sweep["identified"]


def test_the_sweep_is_not_vacuous(sweep) -> None:
    """Guard against a green run that verified almost nothing.

    Without this, tightening a refusal condition or breaking evaluation could silently
    turn the sweep into a no-op that still reports success.
    """
    assert sweep["queries"] >= 400, sweep
    assert sweep["verified"] >= 300, sweep
    assert sweep["verified"] >= 0.5 * sweep["identified"], sweep


def test_the_sweep_exercises_the_hard_paths(sweep) -> None:
    """Coverage of the regimes where the two known defects actually lived."""
    assert sweep["hidden_variable_models"] >= 20, sweep
    assert sweep["refused"] >= 1, sweep


def test_the_sweep_exercises_non_factorizing_deletion_policies(sweep) -> None:
    """`P0` is joint, and the sweep must be able to tell.

    A deletion policy that happens to factorize is reproduced just as well by a product of
    per-variable fallbacks, so a sweep containing only those would pass against the type
    this generalization replaced -- it could not detect a regression. These are policies no
    product form can express, so they make the joint type load-bearing here.
    """
    assert sweep["non_factorizing_deletions"] >= 50, sweep
