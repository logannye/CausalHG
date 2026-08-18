"""A hidden output that reaches nothing observed does not obstruct identification.

`delete(m)` is refused whenever `boundary(m)` contains a hidden variable. That test is too
blunt in one direction that turns out to be the common one. Over the conformance
generator's models, 87 of the 143 mechanisms with a hidden boundary are hidden because of
an *output that nothing consumes* -- and such a variable cannot obstruct anything, because
no observable depends on it.

The policy is the reason it works. `delete(m)` installs a joint `P0^m(out(m))`, supplied by
the caller over every output including the hidden one, so its marginal over the observed
outputs is already in hand: sum the hidden coordinate out of the declared table. Nothing
has to be estimated, and nothing has to be assumed -- the domain of a hidden output is part
of the intervention the caller specified, not something the data has to supply.

That distinguishes this case sharply from the one next door. A hidden output that *does*
reach an observation is not identifiable at all (see `test_hidden_output_deletion.py`):
relabelling it is a symmetry of the data and not of the policy. The two differ by exactly
one question -- can this variable move anything anyone measured -- and the answers are at
opposite ends of the range, so the check that separates them has to be right.
"""
from __future__ import annotations

from causal_hypergraphs import (
    DeleteMechanism,
    Identified,
    MechanismGraph,
    Unidentified,
    identify,
)
from causal_hypergraphs.semantics import DiscreteModel

BINARY = (0, 1)


def _graph() -> MechanismGraph:
    """`m1` produces an observed `C` and a hidden `dead`, which nothing consumes."""
    return MechanismGraph(
        variables={"A", "C", "dead", "Y"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("C", "dead")},
            "m2": {"inputs": ("C",), "outputs": ("Y",)},
        },
        observed_variables={"A", "C", "Y"},
    )


# --- it identifies -----------------------------------------------------------------


def test_a_dead_end_hidden_output_no_longer_blocks_identification() -> None:
    result = identify(_graph(), DeleteMechanism("m1", outcomes={"Y"}))

    assert isinstance(result, Identified), result


def test_the_hidden_output_is_summed_out_of_the_policy_not_left_free() -> None:
    """The caller supplies the joint policy, so its marginal is a sum, not an estimate.

    The hidden variable must survive in neither `scope()` nor `footprint()`. Not in scope,
    because a caller cannot bind a value for something never measured; not in the
    footprint either, because the sum runs over the declared table's own keys rather than
    over a domain, so nothing downstream needs to know the variable exists.
    """
    result = identify(_graph(), DeleteMechanism("m1", outcomes={"Y"}))
    assert isinstance(result, Identified)

    assert "dead" not in result.expression.scope()
    # Summed *inside* the factor, the way `ConditionalExpectation` integrates its target,
    # so it is not a coordinate anything enumerates and costs nothing.
    assert "dead" not in result.expression.footprint()
    assert "sum_{dead} P0_m1(C,dead)" in str(result.expression), result.expression


def test_the_full_joint_query_also_binds_the_hidden_output() -> None:
    """Without `outcomes` there is no ancestral reduction, and the answer is unchanged:
    the marginalization lives in the factor, not in a sum the reduction happened to add."""
    result = identify(_graph(), DeleteMechanism("m1"))
    assert isinstance(result, Identified), result

    assert "dead" not in result.expression.scope()
    assert result.expression.scope() == frozenset({"A", "C", "Y"})


# --- it identifies the *right* thing -----------------------------------------------


def test_the_estimand_matches_the_exact_interventional_law() -> None:
    """Checked against ground truth the compiler never sees, on generated models.

    The sweep is the point: this case is 60% of the hidden-boundary population, so a
    fixture would only show that one graph works. Every model here has a mechanism whose
    hidden output is a dead end, and the exact post-deletion *observed* law comes from the
    harness, computed from the model's own kernels by doing the factor swap and summing the
    hidden variables away.

    The comparator is the harness's own, so an estimand that is undefined where the law
    has mass counts as a defect while a 0/0 on a null set does not -- the sparse-kernel
    models genuinely have such points, and tolerating them is not the same as ignoring a
    failure.
    """
    from tests.conformance.checks import check_estimand
    from tests.conformance.generation import generate_model

    checked = 0
    for seed in range(200):
        model = generate_model(seed)
        graph = model.graph()

        for spec in model.mechanisms:
            mechanism = graph.get_mechanism(spec.name)
            hidden_outputs = set(mechanism.outputs) - graph.observed_set
            hidden_inputs = set(mechanism.inputs) - graph.observed_set
            if not hidden_outputs or hidden_inputs:
                continue
            if any(graph.observed_closure((h,)) for h in hidden_outputs):
                continue  # the unidentifiable neighbour

            result = identify(graph, DeleteMechanism(spec.name))
            assert isinstance(result, Identified), (seed, spec.name, result)

            # No wrapper and no extra domain: the estimand mentions only observed
            # variables, which is exactly what marginalizing inside the factor buys.
            assert result.expression.footprint() <= graph.observed_set

            report = check_estimand(
                result.expression,
                DiscreteModel(
                    domains=model.observed_domains,
                    joint=model.marginalize_to_observed(model.joint()),
                    fallbacks=dict(model.fallbacks),
                ),
                model.marginalize_to_observed(model.interventional_delete(spec.name)),
                model.observed,
            )
            assert report.conforms, f"seed {seed} / delete({spec.name}): {report.summary()}"
            checked += 1

    assert checked > 40, f"only {checked} removable-output query/queries checked"


# --- the boundary between the two verdicts -----------------------------------------


def test_one_consumer_of_the_hidden_output_flips_the_verdict() -> None:
    """The same graph, plus one mechanism reading the hidden variable.

    That single edge is the whole difference between "sum it out of the policy" and "no
    formula exists", so it is worth exhibiting rather than trusting.
    """
    identifiable = identify(_graph(), DeleteMechanism("m1", outcomes={"Y"}))
    assert isinstance(identifiable, Identified)

    graph = MechanismGraph(
        variables={"A", "C", "dead", "Y", "Z"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("C", "dead")},
            "m2": {"inputs": ("C",), "outputs": ("Y",)},
            "m3": {"inputs": ("dead",), "outputs": ("Z",)},   # the one new edge
        },
        observed_variables={"A", "C", "Y", "Z"},
    )
    result = identify(graph, DeleteMechanism("m1", outcomes={"Y"}), allow_t7=True)

    assert isinstance(result, Unidentified), result


def test_a_hidden_input_is_still_refused() -> None:
    """Only outputs are removable. A hidden *input* is a real obstruction and stays one."""
    graph = MechanismGraph(
        variables={"h", "C", "Y"},
        mechanisms={
            "m1": {"inputs": ("h",), "outputs": ("C",)},
            "m2": {"inputs": ("C",), "outputs": ("Y",)},
        },
        observed_variables={"C", "Y"},
    )
    result = identify(graph, DeleteMechanism("m1", outcomes={"Y"}))

    assert not isinstance(result, Identified), result


def test_removability_is_answered_against_the_measurement_plan_in_hand() -> None:
    """`observed_closure` and `removable_outputs` take the observed set as an argument, so
    `identify(..., observed_variables=...)` -- "what would this buy me if I could only
    measure these?" -- gets an answer about *that* plan rather than about the graph's
    declared one."""
    graph = MechanismGraph(
        variables={"A", "C", "dead", "Y", "Z"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("C", "dead")},
            "m2": {"inputs": ("C",), "outputs": ("Y",)},
            "m3": {"inputs": ("dead",), "outputs": ("Z",)},
        },
        observed_variables={"A", "C", "Y", "Z"},
    )

    assert graph.removable_outputs("m1") == ()  # `dead` reaches the measured Z
    assert graph.removable_outputs("m1", frozenset({"A", "C", "Y"})) == ("dead",)


def test_no_quotient_estimand_ever_divides_by_a_hidden_output() -> None:
    """The relaxation is sound only where the target factor is *omitted*, not divided by.

    With other hidden variables around, the identifier emits `P(O) / P(out(m) | in(m))`,
    and that denominator is not an observable quantity when any output is hidden -- the
    fact that nothing downstream depends on it does not make it measurable. Relaxing the
    boundary check on that route produced an estimand naming a variable that was never
    recorded, which is why the two routes are separated rather than sharing one check.
    """
    from tests.conformance.generation import generate_model

    swept = 0
    for seed in range(200):
        model = generate_model(seed)
        graph = model.graph()
        for spec in model.mechanisms:
            result = identify(graph, DeleteMechanism(spec.name))
            if not isinstance(result, Identified):
                continue
            hidden = graph.variable_set - graph.observed_set
            assert not (result.expression.footprint() & hidden), (
                f"seed {seed} / delete({spec.name}): {result.expression}"
            )
            swept += 1

    assert swept > 300, swept
