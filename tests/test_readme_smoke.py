from causal_hypergraphs import DeleteMechanism, MechanismGraph, identify


def test_readme_primary_snippet_smoke() -> None:
    graph = MechanismGraph(
        variables={"A", "B", "C", "D", "E", "F"},
        mechanisms={
            "m1": {"inputs": {"A", "B"}, "outputs": {"C", "D"}},
            "m2": {"inputs": {"C", "E"}, "outputs": {"F"}},
        },
        observed_variables={"A", "B", "C", "D", "E", "F"},
    )

    result = identify(graph, DeleteMechanism("m1"))

    assert result.status == "identified"
    assert str(result.expression) == "P(A,B,C,D,E,F) / P(C,D | A,B) * P0(C) * P0(D)"
    assert result.theorem == "T2"
