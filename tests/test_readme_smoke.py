from causal_hypergraphs import DeleteMechanism, Identified, MechanismGraph, identify


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

    # Narrowing is the point of the result hierarchy: an estimand is unreachable until the
    # caller has established that the query was identified.
    assert isinstance(result, Identified)
    assert result.status == "identified"
    assert str(result.expression) == "P(A) * P(B) * P(E) * P(F | C,E) * P0(C) * P0(D)"
    assert result.theorem == "T2"
