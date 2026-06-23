import pytest

from causal_hypergraphs import (
    BipartiteADMG,
    DeleteMechanism,
    HedgeWitness,
    HyperHedgeWitness,
    Identified,
    StochasticInterventionReduction,
    T7ReductionPlaceholder,
    build_bipartite_dag,
    identify,
    identify_via_t7,
    latent_project,
    mechanism_node,
    reduce_mechanism_query_to_stochastic_intervention,
    variable_node,
)
from causal_hypergraphs.examples import hidden_variable_graph


def test_t7_placeholder_interfaces_are_importable() -> None:
    admg = BipartiteADMG(
        observed_nodes=("A", "B"),
        hidden_nodes=("U",),
        directed_edges=(("A", "B"),),
        bidirected_edges=(("A", "B"),),
    )
    reduction = StochasticInterventionReduction(
        target_outputs=("B",),
        conditioning_inputs=("A",),
        replacement_kernel="P_repl(B | A)",
    )
    witness = HyperHedgeWitness(
        mechanisms=("m1",),
        pearl_witness=HedgeWitness(districts=(("A", "B"),)),
    )
    placeholder = T7ReductionPlaceholder(admg=admg, stochastic_intervention=reduction)

    assert placeholder.admg == admg
    assert placeholder.stochastic_intervention == reduction
    assert witness.pearl_witness is not None


def test_identify_via_t7_returns_honest_unknown() -> None:
    result = identify_via_t7()

    assert result.status == "unknown"
    assert result.next_algorithm == (
        "Build bipartite ADMG, reduce to stochastic intervention, run Pearl ID."
    )


def test_t7_builds_typed_bipartite_dag() -> None:
    dag = build_bipartite_dag(hidden_variable_graph())

    assert variable_node("W") in dag.hidden_nodes
    assert mechanism_node("m_W") in dag.hidden_nodes
    assert (variable_node("A"), mechanism_node("m1")) in dag.directed_edges
    assert (mechanism_node("m_2"), variable_node("F")) in dag.directed_edges


def test_t7_latent_projection_returns_bipartite_admg() -> None:
    admg = latent_project(hidden_variable_graph())

    assert variable_node("W") in admg.hidden_nodes
    assert mechanism_node("m_2") in admg.observed_nodes
    assert (variable_node("C"), mechanism_node("m_2")) in admg.directed_edges
    assert admg.to_admg().node_set == frozenset(admg.observed_nodes)


def test_t7_reduces_mechanism_query_to_stochastic_intervention_object() -> None:
    graph = hidden_variable_graph()
    reduction = reduce_mechanism_query_to_stochastic_intervention(
        graph,
        DeleteMechanism("m_2"),
    )

    assert reduction.status == "reduced"
    assert reduction.query_type == "delete"
    assert reduction.target_mechanism == "m_2"
    assert reduction.target_outputs == ("F",)
    assert reduction.conditioning_inputs == ("C", "E", "W")
    assert reduction.admg is not None


def test_identify_via_t7_records_reduction_context_when_available() -> None:
    result = identify_via_t7(hidden_variable_graph(), DeleteMechanism("m_2"))

    assert result.status == "unknown"
    assert result.suggestions[0] == (
        "Reduced delete(m_2) to a stochastic intervention object."
    )


@pytest.mark.xfail(reason="T7 Pearl-ID reduction is a future milestone.")
def test_t7_todo_boundary_violating_case_eventually_identifies_when_possible() -> None:
    result = identify(hidden_variable_graph(), DeleteMechanism("m_2"))

    assert isinstance(result, Identified)
