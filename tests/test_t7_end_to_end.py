from causal_hypergraphs import (
    DeleteMechanism,
    Identified,
    Unknown,
    identify,
    latent_project_to_variable_admg,
    reduce_mechanism_query_to_stochastic_intervention,
)
from causal_hypergraphs.examples import frontdoor_hidden_boundary_graph


def test_frontdoor_hidden_boundary_graph_reduces_to_variable_admg() -> None:
    graph = frontdoor_hidden_boundary_graph()
    admg = latent_project_to_variable_admg(graph)

    assert admg.nodes == ("X", "Y", "Z")
    assert admg.directed_edges == (("X", "Z"), ("Z", "Y"))
    assert admg.bidirected_edges == (("X", "Y"),)


def test_delete_mechanism_reduction_records_pearl_query() -> None:
    graph = frontdoor_hidden_boundary_graph()
    reduction = reduce_mechanism_query_to_stochastic_intervention(
        graph,
        DeleteMechanism("m_x", outcomes={"Y"}),
    )

    assert reduction.query_type == "delete"
    assert reduction.target_outputs == ("X",)
    assert reduction.conditioning_inputs == ("W",)
    assert reduction.pearl_interventions == ("X",)
    assert reduction.pearl_outcomes == ("Y",)
    assert reduction.variable_admg is not None


def test_t7_is_opt_in_for_boundary_violating_queries() -> None:
    graph = frontdoor_hidden_boundary_graph()

    result = identify(graph, DeleteMechanism("m_x", outcomes={"Y"}))

    assert isinstance(result, Unknown)
    assert result.next_algorithm == "T7 Pearl-ID reduction"
    assert result.missing_variables == ("W",)


def test_t7_frontdoor_deletion_path_identifies_when_enabled() -> None:
    graph = frontdoor_hidden_boundary_graph()

    result = identify(graph, DeleteMechanism("m_x", outcomes={"Y"}), allow_t7=True)

    assert isinstance(result, Identified)
    assert result.theorem == "T7"
    assert str(result.expression) == (
        "sum_{X} P0_m_x(X) * sum_{X_prime,Z} P(X_prime) * P(Y | X_prime,Z) * P(Z | X)"
    )
    assert result.expression.scope() == frozenset({"Y"})
    assert result.aliases == {"X_prime": "X"}
    assert any(assumption.code == "T7 reduction" for assumption in result.assumptions)
    assert result.derivation[2].label == "Pearl backend"


def test_t7_requires_explicit_outcomes_for_boundary_violating_deletion() -> None:
    graph = frontdoor_hidden_boundary_graph()

    result = identify(graph, DeleteMechanism("m_x"), allow_t7=True)

    assert isinstance(result, Unknown)
    assert result.reason == "T7 deletion queries require an explicit observed outcome set."
    assert result.next_algorithm == "Call DeleteMechanism(target, outcomes={...})."


def test_the_t7_answer_carries_the_core_assumptions_its_refusals_carry() -> None:
    """C1, C2 and C4 are conditions on the model, not on which branch the compiler took.

    All four boundary-violating refusals record `CORE_T7_ASSUMPTIONS`. The single branch
    that returns a formula recorded only the reduction's own assumptions plus the Pearl
    backend's, so the one result a caller can turn into a number was the one whose ledger
    omitted C2 -- the assumption `Estimate.summary()` is explicitly ordered to lead with
    because it is the likeliest to be false.
    """
    graph = frontdoor_hidden_boundary_graph()

    answer = identify(graph, DeleteMechanism("m_x", outcomes={"Y"}), allow_t7=True)
    assert isinstance(answer, Identified)
    answered = {assumption.code for assumption in answer.assumptions}

    refusal = identify(graph, DeleteMechanism("m_x", outcomes={"Y"}))
    assert isinstance(refusal, Unknown)
    refused = {assumption.code for assumption in refusal.assumptions}

    assert {"C1", "C2", "C4"} <= answered
    # The reduction's own assumptions must survive alongside them.
    assert "T7 reduction" in answered
    # Whatever the model-level ledger is, an answer may not carry less of it than a refusal.
    assert refused & {"C1", "C2", "C4"} <= answered
