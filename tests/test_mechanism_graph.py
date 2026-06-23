import pytest

from causal_hypergraphs import Mechanism, MechanismGraph


def test_c1_cyclic_mechanism_graph_rejects() -> None:
    with pytest.raises(ValueError, match="C1 violation"):
        MechanismGraph(
            variables={"A", "B"},
            mechanisms={
                "m1": {"inputs": {"B"}, "outputs": {"A"}},
                "m2": {"inputs": {"A"}, "outputs": {"B"}},
            },
        )


def test_c4_multi_producer_graph_rejects() -> None:
    with pytest.raises(ValueError, match="C4 violation"):
        MechanismGraph(
            variables={"A", "B", "C"},
            mechanisms={
                "m1": {"inputs": {"A"}, "outputs": {"C"}},
                "m2": {"inputs": {"B"}, "outputs": {"C"}},
            },
        )


def test_mechanism_specs_are_normalized_and_validated() -> None:
    graph = MechanismGraph(
        variables={"C", "B", "A"},
        mechanisms={
            "m1": Mechanism(
                name="other",
                inputs=("B", "A"),
                outputs=("C",),
                output_equalities=(("C",),),
            )
        },
        observed_variables={"C", "A", "B"},
    )

    mechanism = graph.get_mechanism("m1")
    assert mechanism.name == "m1"
    assert mechanism.inputs == ("A", "B")
    assert mechanism.outputs == ("C",)
    assert graph.variable_set == frozenset({"A", "B", "C"})
    assert graph.observed_set == graph.variable_set


def test_hidden_and_fallback_partitions_are_explicit() -> None:
    graph = MechanismGraph(
        variables={"A", "B", "C"},
        mechanisms={"m1": {"inputs": {"A"}, "outputs": {"B"}}},
        observed_variables={"A", "B"},
        fallback_variables={"B"},
    )

    assert graph.hidden_variables == frozenset({"C"})
    assert graph.fallback_set == frozenset({"B"})
    assert graph.exogenous_variables == frozenset({"A", "C"})
    assert graph.missing_boundary_variables("m1") == ()
