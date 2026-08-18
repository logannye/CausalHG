"""Asking about a few readouts must cost a few readouts, not the whole system.

`DeleteMechanism` and `ReplaceMechanism` have always accepted an `outcomes` argument, but
the compiler ignored it: every estimand was the full joint `P(V | do)`. Evaluating that
means enumerating the product of every variable's domain, which is fine for a six-variable
example and impossible for anything of realistic width -- 2**20000 for a transcriptome.

Nobody wants the joint law of the system. They want `P(Y | do)` for a handful of readouts.

The reduction is exact, not an approximation, and it is the ordinary ancestral argument.
In the truncated factorization every factor is a conditional `P(out(m) | in(m))`, which
sums to one over `out(m)` at any fixed `in(m)`. So summing the product over variables
outside the ancestral closure of `Y` collapses those factors to one and deletes them. What
is left is the sub-product over the mechanisms that can actually reach `Y`.

Two consequences are worth stating as tests rather than as comments:

- the cost of a query scales with the *ancestry of the outcome*, not with the size of the
  system -- which is the whole point;
- a mechanism that cannot reach `Y` leaves the estimand entirely, so its deletion policy
  is structurally absent rather than numerically cancelling. An answer that cannot depend
  on a policy is a stronger statement than one that happens not to.
"""
from __future__ import annotations

import itertools

import pytest

from causal_hypergraphs import (
    DeleteMechanism,
    Fallback,
    Identified,
    MechanismGraph,
    ReplaceMechanism,
    identify,
)
from causal_hypergraphs.semantics import DiscreteModel, evaluate
from tests.conformance.generation import generate_model

BINARY = (0, 1)


def _chain(length: int) -> MechanismGraph:
    """v0 -> m0 -> v1 -> m1 -> v2 -> ... a pure chain of `length` mechanisms."""
    return MechanismGraph(
        variables={f"v{i}" for i in range(length + 1)},
        mechanisms={
            f"m{i}": {"inputs": (f"v{i}",), "outputs": (f"v{i + 1}",)}
            for i in range(length)
        },
    )


def _identified(graph: MechanismGraph, query: object) -> Identified:
    result = identify(graph, query)  # type: ignore[arg-type]
    assert isinstance(result, Identified), result
    return result


def _fallback_nodes(expression: object) -> list[Fallback]:
    from causal_hypergraphs import Product, SumOut

    if isinstance(expression, Fallback):
        return [expression]
    if isinstance(expression, Product):
        return [n for factor in expression.factors for n in _fallback_nodes(factor)]
    if isinstance(expression, SumOut):
        return _fallback_nodes(expression.expression)
    return []


# --- the reduction ----------------------------------------------------------------


def test_the_footprint_of_a_query_scales_with_ancestry_not_system_size() -> None:
    """The headline property: a long chain does not make a nearby question expensive.

    `footprint()` counts every variable evaluating the estimand must range over, bound
    ones included -- it is the exponent in the enumeration cost. Asking about `v2` in a
    thirty-link chain must not cost thirty variables.
    """
    graph = _chain(30)

    whole_system = _identified(graph, DeleteMechanism("m0"))
    nearby = _identified(graph, DeleteMechanism("m0", outcomes={"v2"}))

    assert len(whole_system.expression.footprint()) == 31
    # Two, not three: the reduction closes over the *post-deletion* graph, so `v0` -- the
    # input `m0` used to read -- is not required either. A deletion replaces that factor
    # with a policy, so nothing above it can reach `v2`.
    assert len(nearby.expression.footprint()) == 2, nearby.expression
    assert nearby.expression.scope() == frozenset({"v2"})


def test_the_reduction_is_exact_against_the_full_joint() -> None:
    """The marginal estimand must equal the full-joint estimand, summed. No approximation.

    Checked pointwise against exact arithmetic over generated models, comparing the
    reduced expression to a brute-force marginalization of the unreduced one. If the
    ancestral argument dropped a factor it should not have, this is what catches it.
    """
    for seed in range(40):
        model = generate_model(seed, allow_hidden=False, shapes=("positive",))
        graph = model.graph()
        discrete = DiscreteModel(
            domains=model.domains,
            joint=model.joint(),
            fallbacks=dict(model.fallbacks),
        )
        for spec in model.mechanisms:
            full = _identified(graph, DeleteMechanism(spec.name))
            for outcome in model.variables:
                reduced = _identified(graph, DeleteMechanism(spec.name, outcomes={outcome}))
                others = [v for v in model.variables if v != outcome]

                for value in BINARY:
                    brute = 0.0
                    for combination in itertools.product(*(BINARY for _ in others)):
                        assignment = dict(zip(others, combination, strict=True))
                        assignment[outcome] = value
                        brute += evaluate(full.expression, discrete, assignment)
                    got = evaluate(reduced.expression, discrete, {outcome: value})
                    assert got == pytest.approx(brute, abs=1e-12), (
                        f"seed {seed} / {spec.name} / {outcome}={value}: "
                        f"reduced {got} vs marginalized {brute}"
                    )


def test_an_unreachable_mechanism_leaves_the_estimand_entirely() -> None:
    """If `m` cannot reach `Y`, `Y`'s answer must not mention `m`'s policy at all.

    Numerical cancellation would give the same number, but a structurally absent policy is
    a stronger claim: it says the answer *cannot* depend on what the intervention installs,
    which is checkable by looking rather than by evaluating at every parameter value.
    """
    graph = MechanismGraph(
        variables={"a", "b", "x", "y"},
        mechanisms={
            "upstream": {"inputs": ("a",), "outputs": ("x",)},
            "unrelated": {"inputs": ("b",), "outputs": ("y",)},
        },
    )

    reachable = _identified(graph, DeleteMechanism("upstream", outcomes={"x"}))
    unreachable = _identified(graph, DeleteMechanism("unrelated", outcomes={"x"}))

    assert _fallback_nodes(reachable.expression), reachable.expression
    assert not _fallback_nodes(unreachable.expression), unreachable.expression
    # With no intervention factor left, the surviving factors are the ancestral
    # factorization of the observational law, which sums to exactly P(x). Emitting that
    # directly costs one variable instead of the whole ancestry -- and it is the strongest
    # form of the statement, since P(x) mentions neither the mechanism nor its policy.
    assert unreachable.expression.footprint() == frozenset({"x"})
    assert str(unreachable.expression) == "P(x)"


def test_an_unreachable_intervention_returns_the_observational_marginal() -> None:
    """The numerical counterpart: deleting something Y cannot see leaves P(Y) alone."""
    graph = MechanismGraph(
        variables={"a", "b", "x", "y"},
        mechanisms={
            "upstream": {"inputs": ("a",), "outputs": ("x",)},
            "unrelated": {"inputs": ("b",), "outputs": ("y",)},
        },
    )
    joint = {
        (a, b, x, y): p
        for (a, b, x, y), p in {
            (0, 0, 0, 0): 0.10, (0, 0, 0, 1): 0.05, (0, 0, 1, 0): 0.04, (0, 0, 1, 1): 0.06,
            (0, 1, 0, 0): 0.08, (0, 1, 0, 1): 0.07, (0, 1, 1, 0): 0.05, (0, 1, 1, 1): 0.05,
            (1, 0, 0, 0): 0.03, (1, 0, 0, 1): 0.07, (1, 0, 1, 0): 0.09, (1, 0, 1, 1): 0.06,
            (1, 1, 0, 0): 0.04, (1, 1, 0, 1): 0.06, (1, 1, 1, 0): 0.08, (1, 1, 1, 1): 0.07,
        }.items()
    }
    model = DiscreteModel(
        domains={v: BINARY for v in ("a", "b", "x", "y")},
        joint=joint,
        fallbacks={"unrelated": {(0,): 0.9, (1,): 0.1}},
    )
    model.validate()
    result = _identified(graph, DeleteMechanism("unrelated", outcomes={"x"}))

    for value in BINARY:
        observational = sum(p for key, p in joint.items() if key[2] == value)
        assert evaluate(result.expression, model, {"x": value}) == pytest.approx(
            observational, abs=1e-12
        )


def test_replacement_queries_reduce_the_same_way() -> None:
    """`replace` swaps a factor rather than dropping it; the ancestral argument is identical."""
    graph = _chain(10)

    whole = _identified(graph, ReplaceMechanism("m0", "m0_prime"))
    assert len(whole.expression.footprint()) == 11

    narrow = _identified(graph, ReplaceMechanism("m0", "m0_prime", outcomes={"v2"}))
    assert len(narrow.expression.footprint()) == 3
    assert narrow.expression.scope() == frozenset({"v2"})


# --- contract ---------------------------------------------------------------------


def test_no_outcomes_still_returns_the_full_joint() -> None:
    """Backward compatibility: the default query is unchanged."""
    graph = _chain(4)
    result = _identified(graph, DeleteMechanism("m0"))

    assert result.expression.scope() == graph.variable_set


def test_an_outcome_outside_the_graph_is_rejected() -> None:
    """A typo in a readout name must not silently become a full-joint query."""
    graph = _chain(3)

    with pytest.raises(ValueError, match="ghost"):
        identify(graph, DeleteMechanism("m0", outcomes={"ghost"}))


def test_marginal_queries_estimate_correctly_end_to_end() -> None:
    """Through the estimator, against exact ground truth summed to the outcome.

    The reduction is applied at compile time, so a mistake in it would surface as a wrong
    *number* here rather than as a wrong expression. Sampling is exact -- the dataset is
    built from the model's own law scaled to integer counts -- so any disagreement is the
    reduction, not Monte Carlo error.
    """
    from causal_hypergraphs.estimation import Dataset, estimate

    checked = 0
    for seed in range(25):
        model = generate_model(seed, allow_hidden=False, shapes=("positive",))
        graph = model.graph()
        scale = 100_000
        counts = {
            key: round(probability * scale)
            for key, probability in model.joint().items()
            if round(probability * scale) > 0
        }
        data = Dataset.from_counts(
            counts, model.variables, domains={v: BINARY for v in model.variables}
        )
        empirical_total = sum(counts.values())

        for spec in model.mechanisms:
            truth = model.interventional_delete(spec.name)
            for outcome in model.variables:
                result = _identified(graph, DeleteMechanism(spec.name, outcomes={outcome}))
                est = estimate(result, data, fallbacks={spec.name: model.fallbacks[spec.name]})
                position = model.variables.index(outcome)
                for value in BINARY:
                    exact = sum(p for key, p in truth.items() if key[position] == value)
                    assert est.values[(value,)] == pytest.approx(exact, abs=2e-4), (
                        f"seed {seed} / delete({spec.name}) / {outcome}={value}"
                    )
                    checked += 1
        assert empirical_total > 0

    assert checked > 500, f"only {checked} marginal point(s) checked"


def test_the_derivation_records_the_reduction() -> None:
    """A dropped factor is a step in the argument, so it belongs in the proof."""
    graph = _chain(6)
    result = _identified(graph, DeleteMechanism("m0", outcomes={"v2"}))

    labels = [step.label for step in result.derivation]
    assert "Restrict to ancestry" in labels, labels
    detail = next(s.detail for s in result.derivation if s.label == "Restrict to ancestry")
    assert "v2" in detail
