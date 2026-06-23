from causal_hypergraphs import d_separated, deterministic_closure
from causal_hypergraphs.examples import reaction_graph


def test_equality_closure_preserves_current_deterministic_behavior() -> None:
    graph = reaction_graph()

    assert deterministic_closure(graph, {"C"}) == frozenset({"C", "D"})
    assert deterministic_closure(graph, {"D"}) == frozenset({"C", "D"})
    assert deterministic_closure(graph, {"A"}) == frozenset({"A"})


def test_d_separation_uses_equality_closure() -> None:
    graph = reaction_graph()

    assert d_separated(graph, {"A"}, {"D"}, {"C"})
    assert not d_separated(graph, {"A"}, {"F"})
