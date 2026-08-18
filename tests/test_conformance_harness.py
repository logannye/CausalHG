"""Fail-proofs for the conformance harness.

A conformance sweep that nothing can trip is worse than no sweep, because it reads as
coverage. Before trusting the harness in `test_conformance_identifier.py` and
`test_conformance_separation.py`, show that each of its checkers rejects a deliberately
wrong input.
"""
from __future__ import annotations

from causal_hypergraphs import DeleteMechanism, Identified, identify
from causal_hypergraphs.expression import Fallback, Probability, Product
from causal_hypergraphs.semantics import DiscreteModel
from causal_hypergraphs.separation import d_separated, deterministic_closure
from tests.conformance.checks import (
    check_estimand,
    check_separation_claims,
    conditional_independence_holds,
    separation_triples,
)
from tests.conformance.generation import generate_model


def _full_observation_model(start: int = 0):
    """The first generated model with no hidden variable, so the T2/T4 branch is used."""
    for seed in range(start, start + 200):
        model = generate_model(seed)
        if len(model.observed) == len(model.variables) and model.mechanisms:
            return model
    raise AssertionError("no fully observed model found")


def _delete_case(model):
    target = model.mechanisms[0].name
    result = identify(model.graph(), DeleteMechanism(target))
    assert isinstance(result, Identified)
    discrete = DiscreteModel(
        domains=model.domains,
        joint=model.joint(),
        fallbacks=model.fallbacks,
        replacements={},
    )
    return result, discrete, model.interventional_delete(target)


def test_checker_accepts_the_compiled_estimand() -> None:
    """Baseline: the real estimand conforms. Without this, rejection proves nothing."""
    model = _full_observation_model()
    result, discrete, truth = _delete_case(model)

    report = check_estimand(result.expression, discrete, truth, model.variables)

    assert report.conforms, report.summary()
    assert report.points_checked == 2 ** len(model.variables)


def test_checker_rejects_a_wrong_estimand() -> None:
    """Drop a fallback factor from the compiled estimand; the checker must notice."""
    model = _full_observation_model()
    result, discrete, truth = _delete_case(model)

    factors = list(result.expression.factors)  # type: ignore[attr-defined]
    mutated = Product([f for f in factors if not isinstance(f, Fallback)][: len(factors) - 1])

    report = check_estimand(mutated, discrete, truth, model.variables)

    assert not report.conforms, "checker accepted an estimand missing its fallback factors"


def test_checker_rejects_an_estimand_that_is_merely_the_observational_law() -> None:
    """P(V) itself is the most seductive wrong answer: right shape, no intervention."""
    model = _full_observation_model()
    _, discrete, truth = _delete_case(model)

    observational = Probability(model.variables)

    report = check_estimand(observational, discrete, truth, model.variables)

    assert not report.conforms


def test_conditional_independence_detects_real_dependence() -> None:
    """A variable is not independent of the mechanism output it feeds, given nothing."""
    model = _full_observation_model()
    spec = model.mechanisms[0]
    joint = model.joint()

    dependent = conditional_independence_holds(
        joint, model.variables, {spec.inputs[0]}, {spec.outputs[0]}, set()
    )

    assert dependent is False


def test_conditional_independence_accepts_a_true_independence() -> None:
    """Distinct exogenous variables are independent by construction."""
    model = _full_observation_model()
    exogenous = model.exogenous
    if len(exogenous) < 2:
        model = generate_model(next(s for s in range(500) if len(generate_model(s).exogenous) >= 2))
        exogenous = model.exogenous

    holds = conditional_independence_holds(
        model.joint(), model.variables, {exogenous[0]}, {exogenous[1]}, set()
    )

    assert holds is True


def test_separation_checker_rejects_an_oracle_that_separates_everything() -> None:
    """Inject an unsound oracle; the harness must report unsoundness rather than pass."""
    model = _full_observation_model()
    triples = separation_triples(model.variables, limit=40)

    report = check_separation_claims(
        model.graph(), model.joint(), model.variables, triples, oracle=lambda *_: True
    )

    assert report.unsound, "harness accepted an oracle that declares every triple separated"


def test_separation_checker_accepts_a_sound_oracle() -> None:
    """An oracle that never claims separation is sound, if uninformative."""
    model = _full_observation_model()
    triples = separation_triples(model.variables, limit=40)

    report = check_separation_claims(
        model.graph(), model.joint(), model.variables, triples, oracle=lambda *_: False
    )

    assert not report.unsound
    assert report.triples_checked == len(triples)


def test_harness_detects_the_historical_separation_bug() -> None:
    """The decisive fail-proof: reintroduce the real bug and confirm the sweep catches it.

    Before the reachability rewrite, `d_separated` returned True as soon as *any* element
    of X or Y fell in the determination closure of Z. Rebuild exactly that early return
    on top of the current oracle and run it through the same checker used by
    `test_conformance_separation.py`. If the sweep cannot see this, it cannot see the
    class of defect it exists to prevent.
    """
    def legacy_oracle(graph, x, y, z) -> bool:
        conditioned = deterministic_closure(graph, set(z))
        if set(x) & conditioned or set(y) & conditioned:
            return True  # the bug: partial determination treated as separation
        return d_separated(graph, set(x), set(y), given=set(z))

    caught = 0
    for seed in range(60):
        model = generate_model(seed)
        triples = separation_triples(model.variables, limit=45, seed=seed)
        report = check_separation_claims(
            model.graph(), model.joint(), model.variables, triples, oracle=legacy_oracle
        )
        caught += len(report.unsound)

    assert caught > 0, (
        "the conformance sweep did not flag the historical partial-determination bug; "
        "it cannot be trusted to catch a regression of it"
    )
