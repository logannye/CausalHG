"""The Pearl-ID backend's contract, stated as behaviour rather than as rendered text.

This file used to pin four literal strings produced by a three-case stub. Three of the four
survive as facts about the answer; the fourth pinned a **wrong refusal** -- the stub
declined `X <-> Y` with no directed edge between them, where `X` cannot affect `Y` at all
and the answer is simply `P(Y)`.

Numeric agreement lives in `test_shpitser_id.py`, checked against exact interventional laws
on concrete models. What is pinned here is the contract: which shapes come back identified,
which come back refused, and what a refusal carries.
"""
from causal_hypergraphs import ADMG, Identified, Unidentified, identify_effect
from causal_hypergraphs.identification.shpitser import Hedge


def test_a_query_with_no_intervention_is_the_observational_marginal() -> None:
    graph = ADMG(nodes={"X", "Y"}, directed_edges={("X", "Y")})

    result = identify_effect(graph, outcomes={"Y"})

    assert isinstance(result, Identified)
    # `sum_X P(X,Y)` and `P(Y)` are the same number; the second is the one worth emitting,
    # because a sum nobody has to perform is a sum nobody has to pay for.
    assert str(result.expression) == "P(Y)"
    assert result.expression.scope() == frozenset({"Y"})


def test_an_unconfounded_effect_is_the_conditional() -> None:
    graph = ADMG(nodes={"X", "Y"}, directed_edges={("X", "Y")})

    result = identify_effect(graph, outcomes={"Y"}, interventions={"X"})

    assert isinstance(result, Identified)
    assert str(result.expression) == "P(Y | X)"


def test_the_frontdoor_effect_is_identified_and_declares_its_copy() -> None:
    graph = ADMG(
        nodes={"X", "Z", "Y"},
        directed_edges={("X", "Z"), ("Z", "Y")},
        bidirected_edges={("X", "Y")},
    )

    result = identify_effect(graph, outcomes={"Y"}, interventions={"X"})

    assert isinstance(result, Identified)
    assert result.expression.scope() == frozenset({"X", "Y"})
    assert result.aliases == {"X_prime": "X"}
    # The copy must not leak into what a caller has to supply.
    assert "X_prime" not in result.expression.scope()


def test_an_effect_that_cannot_reach_its_outcome_is_identified_not_refused() -> None:
    """`X <-> Y` with no directed edge: `X` has no effect, so the answer is `P(Y)`.

    Confounding alone never obstructs identification -- it is confounding *along a causal
    path* that does. Refusing here, as the previous backend did, is a false negative, and a
    false negative from an algorithm advertised as complete is a claim about the graph that
    is not true of it.
    """
    graph = ADMG(nodes={"X", "Y"}, bidirected_edges={("X", "Y")})

    result = identify_effect(graph, outcomes={"Y"}, interventions={"X"})

    assert isinstance(result, Identified), result
    assert str(result.expression) == "P(Y)"


def test_the_bow_arc_is_refused_with_a_hedge_naming_its_two_forests() -> None:
    """Add the directed edge and the same confounding becomes fatal."""
    graph = ADMG(nodes={"X", "Y"}, directed_edges={("X", "Y")}, bidirected_edges={("X", "Y")})

    result = identify_effect(graph, outcomes={"Y"}, interventions={"X"})

    assert isinstance(result, Unidentified)
    assert isinstance(result.witness, Hedge)
    assert result.witness.forest == ("X", "Y")
    assert result.witness.subforest == ("Y",)
    assert "hedge" in result.reason
