"""Randomized conformance for T1: is the d*-separation oracle sound?

`d_separated` returning True licenses a conditional independence downstream, so a false
True is the only failure that can corrupt a result. This sweep enumerates disjoint
(X, Y, Z) triples over generated models and checks every claimed separation against the
exact conditional independence in the model's own joint law.

This matters more than the usual property test. `Fact 4b` in `THEOREM_T1.md` -- the step
that is supposed to reconcile the determination closure of the augmented blowup with the
closure in the hypergraph -- is not established by the argument given there: its
parent-closure iteration cannot derive `D` from `{C}` when `C` and `D` are siblings
produced by one mechanism, which is precisely the configuration the augmentation exists
to handle. The theorem may well hold; the proof does not show it. Until the proof is
repaired, this sweep is the evidence that the oracle behaves as T1 claims.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from tests.conformance.checks import check_separation_claims, separation_triples
from tests.conformance.generation import generate_model

MODEL_COUNT = 120
TRIPLES_PER_MODEL = 45


@dataclass(frozen=True)
class SweepResult:
    unsound: tuple[str, ...]
    triples_checked: int
    incomplete: int
    coupled_mechanisms: int

    def failure_text(self) -> str:
        return "\n  ".join(self.unsound[:10])


def _sweep(
    *,
    declare_equalities: bool = True,
    shapes: Sequence[str] | None = None,
    model_count: int = MODEL_COUNT,
) -> SweepResult:
    """Check every separation verdict over `model_count` models against the exact law."""
    unsound: list[str] = []
    triples_checked = 0
    incomplete = 0
    coupled = 0

    for seed in range(model_count):
        model = generate_model(seed, declare_equalities=declare_equalities, shapes=shapes)
        coupled += sum(
            1 for spec in model.mechanisms if spec.shape == "coupled" and len(spec.outputs) > 1
        )
        triples = separation_triples(model.variables, limit=TRIPLES_PER_MODEL, seed=seed)
        report = check_separation_claims(
            model.graph(), model.joint(), model.variables, triples
        )
        triples_checked += report.triples_checked
        incomplete += len(report.incomplete)
        if report.unsound:
            unsound.append(f"seed {seed}: {report.summary()}")

    return SweepResult(
        unsound=tuple(unsound),
        triples_checked=triples_checked,
        incomplete=incomplete,
        coupled_mechanisms=coupled,
    )


@pytest.fixture(scope="module")
def declared() -> SweepResult:
    return _sweep(declare_equalities=True)


@pytest.fixture(scope="module")
def undeclared() -> SweepResult:
    return _sweep(declare_equalities=False)


@pytest.fixture(scope="module")
def positive_only() -> SweepResult:
    return _sweep(shapes=("positive",))


def test_no_claimed_separation_is_false(declared: SweepResult) -> None:
    """T1 soundness: every True verdict is an actual conditional independence."""
    assert not declared.unsound, declared.failure_text()


def test_the_separation_sweep_is_not_vacuous(declared: SweepResult) -> None:
    """A sweep over zero triples, or over models with no determination, proves nothing."""
    assert declared.triples_checked >= 2000, declared
    assert declared.coupled_mechanisms >= 20, declared


def test_separation_is_complete_on_generically_faithful_models(
    positive_only: SweepResult,
) -> None:
    """T1 completeness, gated where its faithfulness hypothesis is actually met.

    With strictly positive kernels there is no deterministic structure, and unfaithful
    parameterizations are measure zero, so every conditional independence in the law
    should be visible to the graphical criterion. Any shortfall here would be a genuine
    completeness defect rather than an artifact of the model class.
    """
    assert positive_only.triples_checked >= 2000, positive_only
    assert not positive_only.unsound, positive_only.failure_text()
    assert positive_only.incomplete == 0, (
        f"{positive_only.incomplete} independence(s) missed on models with no "
        "deterministic structure, where the criterion is supposed to be complete"
    )


def test_undeclared_determination_costs_completeness_not_soundness(
    undeclared: SweepResult,
) -> None:
    """The targeted case the generator will not produce by accident.

    When a mechanism's outputs are functionally equal but that equality is *not*
    declared, the closure cannot see it, so the oracle misses independences it would
    otherwise report. That is incompleteness, and it is the safe direction: the oracle
    declines to claim a separation rather than claiming a false one. This pins that the
    failure mode really is one-sided.
    """
    assert not undeclared.unsound, undeclared.failure_text()
    assert undeclared.coupled_mechanisms >= 20, undeclared


def test_declaring_determination_recovers_independences(
    declared: SweepResult, undeclared: SweepResult
) -> None:
    """Declaring the equality strictly increases what the oracle can prove.

    If it did not, `output_equalities` and the determination closure would be inert --
    a feature that costs nothing and buys nothing.
    """
    assert undeclared.incomplete > declared.incomplete, (
        f"declared={declared.incomplete} undeclared={undeclared.incomplete}: declaring "
        "output equalities recovered no independence, so the determination closure is "
        "doing nothing"
    )
