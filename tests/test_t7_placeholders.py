import pytest

from causal_hypergraphs import (
    BipartiteADMG,
    DeleteMechanism,
    HedgeWitness,
    HyperHedgeWitness,
    Identified,
    StochasticInterventionReduction,
    T7ReductionPlaceholder,
    identify,
    identify_via_t7,
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


@pytest.mark.xfail(reason="T7 Pearl-ID reduction is a future milestone.")
def test_t7_todo_boundary_violating_case_eventually_identifies_when_possible() -> None:
    result = identify(hidden_variable_graph(), DeleteMechanism("m_2"))

    assert isinstance(result, Identified)
