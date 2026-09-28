"""Unified queries tested against explicit finite structural laws, including shared outputs."""

from __future__ import annotations

from fractions import Fraction as F
from itertools import product

import pytest

from causal_hypergraphs import ADMG, Dataset, Identified, MechanismGraph, Unidentified, Unknown
from causal_hypergraphs.examples import frontdoor_hidden_boundary_graph
from causal_hypergraphs.inference import compile_query, estimate_query, evaluate_query
from causal_hypergraphs.queries import (
    CausalQuery,
    Composite,
    Delete,
    EffectContrast,
    HardIntervention,
    JointPolicy,
    Replace,
)
from causal_hypergraphs.semantics import DiscreteModel, IntractableQuery, SemanticsError
from tests.idcorpus import random_scm


def _fixture() -> tuple[MechanismGraph, DiscreteModel, Dataset]:
    graph = MechanismGraph(
        variables={"C", "T", "U", "Y"},
        mechanisms={
            "pair": {"inputs": ("C",), "outputs": ("T", "U")},
            "outcome": {"inputs": ("C", "T", "U"), "outputs": ("Y",)},
        },
    )
    pair = {
        0: {(0, 0): F(5, 10), (0, 1): F(1, 10), (1, 0): F(1, 10), (1, 1): F(3, 10)},
        1: {(0, 0): F(1, 10), (0, 1): F(1, 10), (1, 0): F(2, 10), (1, 1): F(6, 10)},
    }
    joint = {}
    rows = []
    for c, t, u, y in product((0, 1), repeat=4):
        p_y = F(1, 10) + F(3, 10) * t + F(1, 5) * u + F(1, 5) * c
        probability = (F(2, 5) if c else F(3, 5)) * pair[c][t, u]
        probability *= p_y if y else 1 - p_y
        joint[c, t, u, y] = float(probability)
        count = probability * 500
        assert count.denominator == 1
        rows.extend({"C": c, "T": t, "U": u, "Y": y} for _ in range(int(count)))
    model = DiscreteModel({name: (0, 1) for name in graph.variable_set}, joint)
    return graph, model, Dataset.from_records(rows)


def test_hard_intervention_retains_marginal_shared_output_not_conditional() -> None:
    graph, model, data = _fixture()
    compiled = compile_query(graph, CausalQuery(("Y",), HardIntervention({"T": 1})))
    assert isinstance(compiled.result, Identified)
    # E[U]=.6*.4+.4*.7=.52 under do(T=1), not E[U|T=1]=.75.
    # E[Y]=.1+.3+.2*.52+.2*.4=.584; conditioning on the forced T gives .63 instead.
    assert evaluate_query(compiled, model)[(1,)] == pytest.approx(0.584)
    assert estimate_query(compiled, data).values[(1,)] == pytest.approx(0.584)
    primitive = [
        kernel for kernel in compiled.result.expression.kernels() if kernel.variables == ("U",)
    ]
    assert len(primitive) == 1 and primitive[0].given == ("C",)


def test_joint_policy_preserves_non_alphabetical_axes_and_coupling() -> None:
    graph, model, data = _fixture()
    policy = JointPolicy(("U", "T"), {(1, 0): 0.25, (0, 1): 0.75})
    compiled = compile_query(graph, CausalQuery(("Y", "U", "T"), policy))
    values = evaluate_query(compiled, model)
    assert compiled.variables == ("Y", "U", "T")
    assert sum(values.values()) == pytest.approx(1)
    assert sum(value for (y, _, _), value in values.items() if y == 1) == pytest.approx(0.455)
    assert sum(value for (_, u, t), value in values.items() if u == t) == 0
    assert estimate_query(compiled, data).values == pytest.approx(values)
    with pytest.raises(TypeError):
        compiled.policies["override"] = policy  # type: ignore[index]


def test_delete_uses_declared_joint_policy_without_an_independence_assumption() -> None:
    graph, model, data = _fixture()
    policy = JointPolicy(("U", "T"), {(0, 0): 0.5, (1, 1): 0.5})
    compiled = compile_query(graph, CausalQuery(("U", "T"), Delete("pair", policy)))
    expected = {(0, 0): 0.5, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.5}
    assert evaluate_query(compiled, model) == expected
    assert estimate_query(compiled, data).values == expected


def test_replacement_kernel_is_bound_by_name_and_keeps_input_boundary() -> None:
    graph, model, data = _fixture()
    table = {((t, u), (c,)): float(t == u == 1) for c, t, u in product((0, 1), repeat=3)}
    bound_model = DiscreteModel(model.domains, model.joint, replacements={"new_pair": table})
    compiled = compile_query(graph, CausalQuery(("Y",), Replace("pair", "new_pair")))
    assert evaluate_query(compiled, bound_model)[(1,)] == pytest.approx(0.68)
    estimated = estimate_query(compiled, data, replacements={"new_pair": table})
    assert estimated.values[(1,)] == pytest.approx(0.68)


def test_unnormalized_replacement_is_rejected_by_both_finite_backends() -> None:
    graph, model, data = _fixture()
    table = {((t, u), (c,)): 0.5 for c, t, u in product((0, 1), repeat=3)}
    compiled = compile_query(graph, CausalQuery(("Y",), Replace("pair", "bad")))
    with pytest.raises(SemanticsError, match="normalize"):
        evaluate_query(
            compiled, DiscreteModel(model.domains, model.joint, replacements={"bad": table})
        )
    with pytest.raises(SemanticsError, match="normalize"):
        estimate_query(compiled, data, replacements={"bad": table})


def test_compatible_composite_has_simultaneous_order_independent_semantics() -> None:
    graph, model, _ = _fixture()
    operations = (HardIntervention({"T": 1}), HardIntervention({"C": 0}))
    for order in (operations, tuple(reversed(operations))):
        compiled = compile_query(graph, CausalQuery(("Y",), Composite(order)))
        assert evaluate_query(compiled, model)[(1,)] == pytest.approx(0.48)


def test_graph_dependent_intervention_conflicts_fail_before_identification() -> None:
    graph, _, _ = _fixture()
    query = CausalQuery(("Y",), Composite((Replace("pair", "new"), HardIntervention({"T": 1}))))
    with pytest.raises(ValueError, match="Conflicting"):
        compile_query(graph, query)


def test_observational_query_and_intervened_outcome_use_declared_axes() -> None:
    graph, model, _ = _fixture()
    observational = compile_query(graph, CausalQuery(("Y",), Composite(())))
    truth = sum(mass for (_, _, _, y), mass in model.joint.items() if y == 1)
    assert evaluate_query(observational, model)[(1,)] == pytest.approx(truth)
    assigned = compile_query(graph, CausalQuery(("T",), HardIntervention({"T": 1})))
    assert evaluate_query(assigned, model) == {(0,): 0.0, (1,): 1.0}


def test_baseline_distribution_and_finite_expectation_are_conditioned_correctly() -> None:
    graph, model, data = _fixture()
    intervention = HardIntervention({"T": 1})
    conditional = compile_query(graph, CausalQuery(("Y",), intervention, given=("C",)))
    law = evaluate_query(conditional, model)
    assert law[(1, 0)] == pytest.approx(0.48)
    assert law[(1, 1)] == pytest.approx(0.74)
    assert sum(law[y, 0] for y in (0, 1)) == pytest.approx(1)
    assert estimate_query(conditional, data).values == pytest.approx(law)
    mean = compile_query(graph, CausalQuery(("Y",), intervention, "expectation", given=("C",)))
    expected = {(0,): 0.48, (1,): 0.74}
    assert evaluate_query(mean, model) == pytest.approx(expected)
    assert estimate_query(mean, data).values == pytest.approx(expected)


def test_descendant_or_intervened_baselines_get_an_explicit_refusal() -> None:
    graph, _, _ = _fixture()
    descendant = compile_query(graph, CausalQuery(("U",), HardIntervention({"T": 1}), given=("Y",)))
    assigned = compile_query(graph, CausalQuery(("Y",), HardIntervention({"T": 1}), given=("T",)))
    assert isinstance(descendant.result, Unknown)
    assert isinstance(assigned.result, Unknown)
    assert "Baseline conditioning" in descendant.result.reason


def test_effect_contrast_estimates_the_functional_on_paired_bootstrap_replicates() -> None:
    graph, model, data = _fixture()
    left = CausalQuery(("Y",), HardIntervention({"T": 1}), "expectation")
    right = CausalQuery(("Y",), HardIntervention({"T": 0}), "expectation")
    compiled = compile_query(graph, EffectContrast(left, right))
    assert evaluate_query(compiled, model) == pytest.approx({(): 0.3})
    assert estimate_query(compiled, data, bootstrap=20).values == pytest.approx({(): 0.3})
    identical = compile_query(graph, EffectContrast(left, left))
    estimated = estimate_query(identical, data, bootstrap=25, seed=83)
    assert estimated.values == {(): 0.0}
    assert estimated.interval == {(): (0.0, 0.0)}
    assert estimated.plan is None  # an opaque difference must not claim a width-zero plan
    assert estimated.summary().startswith(
        "Estimate of effect contrast (left - right) for mean of Y"
    )
    assert (
        estimate_query(compile_query(graph, left), data)
        .summary()
        .startswith("Estimate of mean of Y")
    )


def _frontdoor() -> ADMG:
    return ADMG(("X", "Y", "Z"), (("X", "Z"), ("Z", "Y")), (("X", "Y"),))


@pytest.mark.parametrize("mechanism_graph", [False, True])
def test_frontdoor_queries_preserve_copies_and_match_independent_structural_oracle(
    mechanism_graph: bool,
) -> None:
    admg = _frontdoor()
    graph = frontdoor_hidden_boundary_graph() if mechanism_graph else admg
    query = CausalQuery(("Y",), HardIntervention({"X": 1}))
    compiled = compile_query(graph, query)
    assert isinstance(compiled.result, Identified)
    assert compiled.result.aliases == {"X_prime": "X"}
    for seed in range(8):
        scm = random_scm(admg, seed)
        model = DiscreteModel({name: (0, 1) for name in admg.nodes}, scm.joint())
        assert evaluate_query(compiled, model) == pytest.approx(
            scm.interventional(("Y",), {"X": 1})
        )


def test_projected_stochastic_policy_is_a_joint_mixture_of_identified_variable_effects() -> None:
    admg = _frontdoor()
    policy = JointPolicy(("X",), {(0,): 0.4, (1,): 0.6})
    compiled = compile_query(admg, CausalQuery(("Y",), policy))
    for seed in range(8):
        scm = random_scm(admg, seed)
        model = DiscreteModel({name: (0, 1) for name in admg.nodes}, scm.joint())
        zero, one = scm.interventional(("Y",), {"X": 0}), scm.interventional(("Y",), {"X": 1})
        expected = {point: 0.4 * zero[point] + 0.6 * one[point] for point in zero}
        assert evaluate_query(compiled, model) == pytest.approx(expected)


def test_hedge_is_not_promoted_to_a_proof_against_a_policy_mixture_or_contrast() -> None:
    graph = ADMG(("X", "Y"), (("X", "Y"),), (("X", "Y"),))
    hard = CausalQuery(("Y",), HardIntervention({"X": 1}))
    assert isinstance(compile_query(graph, hard).result, Unidentified)
    policy = CausalQuery(("Y",), JointPolicy(("X",), {(0,): 0.5, (1,): 0.5}))
    assert isinstance(compile_query(graph, policy).result, Unknown)
    assert isinstance(compile_query(graph, EffectContrast(hard, hard)).result, Unknown)
    hidden = MechanismGraph(
        variables=("W", "X", "Y"),
        observed_variables=("X", "Y"),
        mechanisms={
            "cause": {"outputs": ("W",)},
            "treatment": {"inputs": ("W",), "outputs": ("X",)},
            "response": {"inputs": ("W", "X"), "outputs": ("Y",)},
        },
    )
    assert isinstance(compile_query(hidden, hard).result, Unknown)


def test_hidden_replacement_and_projected_overlapping_outcome_are_explicitly_unsupported() -> None:
    hidden = frontdoor_hidden_boundary_graph()
    query = CausalQuery(("Y",), Replace("m_x", "new_x"))
    assert isinstance(compile_query(hidden, query).result, Unknown)
    overlap = CausalQuery(("X",), HardIntervention({"X": 1}))
    assert isinstance(compile_query(_frontdoor(), overlap).result, Unknown)


def test_sparse_policy_does_not_extrapolate_beyond_the_declared_data_domain() -> None:
    graph, _, data = _fixture()
    query = compile_query(graph, CausalQuery(("Y",), HardIntervention({"T": 1})))
    t_index = data.variables.index("T")
    rows = [
        dict(zip(data.variables, point, strict=True)) for point in data.rows if point[t_index] == 0
    ]
    with pytest.raises(SemanticsError, match="outside declared finite domains"):
        estimate_query(query, Dataset.from_records(rows))
    declared = Dataset.from_records(rows, domains=data.domains)
    unsupported = estimate_query(query, declared)
    assert not unsupported.support.holds
    assert not unsupported.values
    with pytest.raises(ValueError, match="cannot be overridden"):
        estimate_query(query, data, fallbacks={})


def test_non_numeric_expectations_and_excessive_reference_work_raise_named_errors() -> None:
    graph, model, data = _fixture()
    mean = compile_query(graph, CausalQuery(("Y",), HardIntervention({"T": 1}), "expectation"))
    with pytest.raises(IntractableQuery):
        evaluate_query(mean, model, max_entries=1)
    contrast = compile_query(graph, EffectContrast(mean.query, mean.query))  # type: ignore[arg-type]
    with pytest.raises(IntractableQuery):
        estimate_query(contrast, data, max_entries=1)
    labels = MechanismGraph(variables=("Y",), mechanisms={})
    compiled = compile_query(labels, CausalQuery(("Y",), Composite(()), "expectation"))
    categorical = DiscreteModel({"Y": ("no", "yes")}, {("no",): 0.5, ("yes",): 0.5})
    with pytest.raises(SemanticsError, match="numeric"):
        evaluate_query(compiled, categorical)


def test_nested_mean_and_conditional_work_cannot_bypass_the_resource_budget() -> None:
    # Flat distribution queries on a binary chain have width one. Wrapping that
    # law in a mean or conditional ratio creates a nested reference sum that the
    # legacy planner cannot optimize; its work must not be advertised as size two.
    chain = tuple(f"X{index}" for index in range(9))
    graph = MechanismGraph(
        variables=(*chain, "B"),
        mechanisms={
            f"m{index}": {"inputs": (chain[index - 1],), "outputs": (chain[index],)}
            for index in range(1, len(chain))
        },
    )
    data = Dataset.from_records([
        {**dict.fromkeys(chain, x), "B": b} for x, b in product((0, 1), repeat=2)
    ])
    intervention = HardIntervention({"X0": 0})
    simple = compile_query(graph, CausalQuery(("X8",), intervention))
    assert estimate_query(simple, data, max_entries=4).values == {(0,): 1.0, (1,): 0.0}
    mean = compile_query(graph, CausalQuery(("X8",), intervention, "expectation"))
    conditional = compile_query(graph, CausalQuery(("X8",), intervention, given=("B",)))
    for compiled in (mean, conditional):
        with pytest.raises(IntractableQuery):
            estimate_query(compiled, data, max_entries=4)
