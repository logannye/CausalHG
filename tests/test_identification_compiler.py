from causal_hypergraphs import (
    DeleteMechanism,
    Identified,
    MechanismGraph,
    ReplaceMechanism,
    Unknown,
    identify,
)
from causal_hypergraphs.examples import (
    hidden_variable_graph,
    latent_mechanism_graph,
    reaction_graph,
)


def test_delete_full_observation_returns_t2_expression() -> None:
    result = identify(reaction_graph(), DeleteMechanism("m1"))

    assert isinstance(result, Identified)
    assert result.status == "identified"
    assert result.theorem == "T2"
    assert str(result.expression) == "P(A,B,C,D,E,F) / P(C,D | A,B) * P0(C) * P0(D)"
    assert result.expression.to_latex() == (
        r"\frac{P(A,B,C,D,E,F)}{P(C,D \mid A,B)} \cdot P_0(C) \cdot P_0(D)"
    )
    assert any(assumption.code == "C4" for assumption in result.assumptions)
    assert result.derivation[0].label == "Validate graph"


def test_replacement_full_observation_returns_t3_expression() -> None:
    result = identify(reaction_graph(), ReplaceMechanism("m1", replacement="m1_prime"))

    assert isinstance(result, Identified)
    assert result.theorem == "T3"
    assert str(result.expression) == (
        "P(A,B,C,D,E,F) / P(C,D | A,B) * P_m1_prime(C,D | A,B)"
    )
    assert any(assumption.code == "Replacement incidence" for assumption in result.assumptions)


def test_latent_mechanism_uses_t4_when_all_variables_are_observed() -> None:
    result = identify(latent_mechanism_graph(), DeleteMechanism("m_lat"))

    assert isinstance(result, Identified)
    assert result.theorem == "T4"
    assert str(result.expression) == "P(A,B,C,D,E,F) / P(B,E) * P0(B) * P0(E)"


def test_hidden_variable_graph_accepts_observed_boundary_with_t6() -> None:
    result = identify(hidden_variable_graph(), DeleteMechanism("m1"))

    assert isinstance(result, Identified)
    assert result.theorem == "T6"
    assert str(result.expression) == "P(A,B,C,D,E,F) / P(C,D | A,B) * P0(C) * P0(D)"
    assert any(assumption.code == "Observed boundary" for assumption in result.assumptions)


def test_hidden_boundary_returns_unknown_with_t7_guidance() -> None:
    result = identify(hidden_variable_graph(), DeleteMechanism("m_2"))

    assert isinstance(result, Unknown)
    assert result.status == "unknown"
    assert result.reason == "Target mechanism boundary contains hidden variables."
    assert result.next_algorithm == "T7 Pearl-ID reduction"
    assert result.missing_variables == ("W",)
    assert any("Measure boundary variable 'W'." == suggestion for suggestion in result.suggestions)
    assert result.derivation[0].label == "Boundary check"


def test_missing_fallback_policy_returns_unknown() -> None:
    graph = MechanismGraph(
        variables={"A", "B", "C", "D", "E", "F"},
        mechanisms={
            "m1": {"inputs": {"A", "B"}, "outputs": {"C", "D"}},
            "m2": {"inputs": {"C", "E"}, "outputs": {"F"}},
        },
        fallback_variables={"A", "B", "E", "F"},
    )

    result = identify(graph, DeleteMechanism("m1"))

    assert isinstance(result, Unknown)
    assert (
        result.reason
        == "Mechanism deletion would orphan outputs without a declared fallback policy."
    )
    assert result.missing_variables == ("C", "D")
    assert result.next_algorithm is None
