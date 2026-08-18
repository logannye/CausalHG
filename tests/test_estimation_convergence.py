"""Does `estimate` converge to the interventional law it claims to estimate?

The identifier conformance sweep checks that a compiled estimand evaluates to the truth
against the *exact* law. That leaves the data-facing path untested: an estimator can
implement a correct formula and still be wrong about how a finite sample maps onto it --
by normalizing over the wrong set, by dropping empty cells, by conditioning on the sample
rather than the population.

So this sweep samples from a generated model, runs the estimator on the sample, and
measures total-variation distance to the exact interventional law the sample came from.
The property is convergence, and it is measured rather than asserted at one sample size:
error must fall roughly as `n**-0.5`, so a sixteen-fold increase in `n` should roughly
halve it twice.

`test_the_convergence_gate_rejects_an_estimator_that_ignores_the_intervention` runs the
identical gate against a deliberately wrong quantity. A convergence test that a wrong
estimator also passes measures sampling noise, not correctness -- and this one is
especially exposed to that, because the observational and interventional laws agree on
most of the sample space and differ only where the deleted mechanism had influence.
"""
from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from causal_hypergraphs import DeleteMechanism, Identified, identify
from causal_hypergraphs.estimation import Dataset, estimate
from tests.conformance.generation import Point, RandomModel, generate_model

MODEL_COUNT = 30
SMALL = 500
LARGE = 8_000

# Sampling error on a total-variation distance falls like n**-0.5, so a 16x sample should
# shrink it about 4x. Demanding 2x leaves generous headroom for the noise in a 30-model
# mean while still failing decisively for an estimator that converges to the wrong law.
MIN_SHRINK = 2.0


@dataclass(frozen=True)
class ConvergenceResult:
    small: float
    large: float
    models: int

    @property
    def shrink(self) -> float:
        return self.small / self.large if self.large > 0 else float("inf")

    def __str__(self) -> str:
        return (
            f"mean TV over {self.models} models: {self.small:.5f} at n={SMALL}, "
            f"{self.large:.5f} at n={LARGE} (shrink {self.shrink:.2f}x)"
        )


def total_variation(left: Mapping[Point, float], right: Mapping[Point, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def _clean_model(seed: int) -> RandomModel:
    """A model with no hidden variables and strictly positive kernels.

    Convergence is the property under test, so the model class deliberately excludes the
    regimes that make an estimand *refuse*: a hidden boundary produces `Unknown`, and a
    structurally sparse kernel produces genuine positivity failures. Those are covered by
    the identifier sweep and by `test_estimation_api.py`; mixing them in here would
    confound a convergence signal with a support signal.
    """
    return generate_model(seed, allow_hidden=False, shapes=("positive",))


def _target(model: RandomModel) -> tuple[Identified, str]:
    spec = model.mechanisms[0]
    result = identify(model.graph(), DeleteMechanism(spec.name))
    assert isinstance(result, Identified), f"seed {model.seed}: {result!r}"
    return result, spec.name


def _estimated_law(
    model: RandomModel, identified: Identified, target: str, n_rows: int, rng: random.Random
) -> dict[Point, float]:
    counts = model.sample_counts(model.joint(), n_rows, rng)
    data = Dataset.from_counts(
        counts, model.variables, domains={v: (0, 1) for v in model.variables}
    )
    result = estimate(identified, data, fallbacks={target: model.fallbacks[target]})
    return dict(result.values)


def _sweep(n_rows: int, *, ignore_intervention: bool = False) -> float:
    """Mean total-variation error of the estimator against the exact interventional law.

    With `ignore_intervention`, the "estimate" is the empirical *observational* law -- a
    plausible-looking wrong answer that a real bug could produce, for instance by failing
    to swap the target factor. It is the control for this gate.
    """
    rng = random.Random(20260818)
    errors: list[float] = []

    for seed in range(MODEL_COUNT):
        model = _clean_model(seed)
        identified, target = _target(model)
        truth = model.interventional_delete(target)

        if ignore_intervention:
            counts = model.sample_counts(model.joint(), n_rows, rng)
            total = float(sum(counts.values()))
            estimated = {cell: count / total for cell, count in counts.items()}
        else:
            estimated = _estimated_law(model, identified, target, n_rows, rng)

        errors.append(total_variation(estimated, truth))

    return sum(errors) / len(errors)


def _convergence(*, ignore_intervention: bool = False) -> ConvergenceResult:
    return ConvergenceResult(
        small=_sweep(SMALL, ignore_intervention=ignore_intervention),
        large=_sweep(LARGE, ignore_intervention=ignore_intervention),
        models=MODEL_COUNT,
    )


@pytest.fixture(scope="module")
def convergence() -> ConvergenceResult:
    return _convergence()


def test_the_estimator_converges_to_the_interventional_law(
    convergence: ConvergenceResult,
) -> None:
    """More data, less error -- against a law the estimator never sees."""
    assert convergence.shrink >= MIN_SHRINK, str(convergence)


def test_the_estimate_is_already_close_at_the_larger_sample(
    convergence: ConvergenceResult,
) -> None:
    """Convergence toward the *wrong* limit would still shrink; pin the limit too."""
    assert convergence.large < 0.02, str(convergence)


def test_the_convergence_gate_rejects_an_estimator_that_ignores_the_intervention() -> None:
    """The control. If this passed, the gate above would be measuring nothing.

    The observational law is what an estimator returns when the target factor is never
    swapped out. It converges -- to the wrong limit -- so a gate that only checked
    "error decreases with n" would accept it. Both halves of the gate must reject it.
    """
    control = _convergence(ignore_intervention=True)

    assert control.large >= 0.02, (
        f"the observational law came within the correctness threshold of the "
        f"interventional law, so these fixtures cannot tell them apart: {control}"
    )
    assert control.shrink < MIN_SHRINK, (
        f"the shrink gate accepted an estimator that ignores the intervention entirely, "
        f"so it is measuring sampling noise rather than correctness: {control}"
    )
