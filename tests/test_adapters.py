"""Adapter contracts: preserve incidence, identity, roles, and metadata explicitly."""

import random
from decimal import Decimal
from typing import Any, cast

import pytest

from causal_hypergraphs.graph import DirectedHyperedge, DirectedHypergraph
from causal_hypergraphs.io.adapters import (
    from_hypernetx,
    from_incidence,
    from_networkx,
    from_xgi,
    records_from_array,
    records_from_dataframe,
    to_incidence,
    to_networkx,
    to_xgi,
)


def specimen():
    return DirectedHypergraph(
        ("x", "y", "isolate"),
        {
            "e": DirectedHyperedge("e", ("x",), ("y",), {"weight": 2}),
            "parallel": DirectedHyperedge("parallel", ("x",), ("y",)),
            "empty": DirectedHyperedge("empty"),
        },
        {"source": {"names": ["original"]}},
    )


def test_role_records_preserve_empty_edges_isolates_parallel_and_metadata():
    graph = specimen()
    assert from_incidence(**to_incidence(graph)) == graph
    with pytest.raises(ValueError, match="role"):
        from_incidence([{"edge": "e", "node": "x", "role": "unknown"}])


def test_networkx_native_round_trip_is_exact_without_clique_expansion():
    pytest.importorskip("networkx")
    graph = specimen()
    restored = from_networkx(to_networkx(graph))
    assert restored.graph == graph
    assert restored.node_ids["x"] == ("variable", "x")
    assert len(restored.graph.edges) == 3


def test_networkx_external_requires_roles_codec_and_no_silent_attribute_loss():
    nx = pytest.importorskip("networkx")
    graph = nx.DiGraph()
    graph.add_node(1, kind="variable", label="input")
    graph.add_node(2, kind="mechanism", label="producer")
    graph.add_node(3, kind="variable")
    graph.add_edges_from([(1, 2), (2, 3)])
    with pytest.raises(TypeError, match="id_codec"):
        from_networkx(graph)
    result = from_networkx(graph, id_codec=str)
    assert result.graph.edges["2"].inputs == ("1",)
    assert cast(Any, result.graph.metadata["node_attributes"])["1"]["label"] == "input"
    assert result.edge_ids["2"] == 2
    with pytest.raises(ValueError, match="collision"):
        from_networkx(graph, id_codec=lambda _: "same")
    graph.edges[1, 2]["weight"] = 1
    with pytest.raises(ValueError, match="Arc metadata"):
        from_networkx(graph, id_codec=str)


def test_xgi_round_trip_and_undirected_refusal():
    xgi = pytest.importorskip("xgi")
    graph = specimen()
    assert from_xgi(to_xgi(graph)).graph == graph
    with pytest.raises(ValueError, match="undirected"):
        from_xgi(xgi.Hypergraph([["x", "y"]]))


def test_external_directed_xgi_has_no_required_native_marker():
    xgi = pytest.importorskip("xgi")
    graph = xgi.DiHypergraph()
    graph.add_node(1, label="source")
    graph.add_node(2)
    graph.add_node(3)
    graph.add_edge(([1], [2]), idx=4, weight=0.5)
    with pytest.raises(TypeError, match="id_codec"):
        from_xgi(graph)
    converted = from_xgi(graph, id_codec=str, metadata={"source": "external"})
    assert converted.graph.nodes == ("1", "2", "3")
    assert converted.graph.edges["4"].inputs == ("1",)
    assert converted.graph.edges["4"].outputs == ("2",)
    assert converted.graph.edges["4"].metadata == {"weight": 0.5}
    assert converted.graph.metadata == {
        "source_attributes": {"source": "external"},
        "node_attributes": {"1": {"label": "source"}, "2": {}, "3": {}},
    }
    assert converted.node_ids == {"1": 1, "2": 2, "3": 3}
    assert converted.edge_ids == {"4": 4}


@pytest.mark.parametrize("location", ["graph", "node", "edge"])
def test_native_graph_imports_refuse_attributes_they_cannot_preserve(location):
    pytest.importorskip("networkx")
    pytest.importorskip("xgi")
    graph = specimen()
    nx_graph, xgi_graph = to_networkx(graph), to_xgi(graph)
    if location == "graph":
        nx_graph.graph["label"] = "new"
        xgi_graph["label"] = "new"
    elif location == "node":
        nx_graph.nodes[("variable", "x")]["label"] = "new"
        xgi_graph.nodes["x"]["label"] = "new"
    else:
        nx_graph.nodes[("mechanism", "e")]["label"] = "new"
        xgi_graph.edges["e"]["label"] = "new"
    with pytest.raises(ValueError, match="Native NetworkX"):
        from_networkx(nx_graph)
    with pytest.raises(ValueError, match="Native XGI"):
        from_xgi(xgi_graph)


def test_seeded_arbitrary_incidence_round_trips_across_three_representations():
    pytest.importorskip("networkx")
    pytest.importorskip("xgi")
    for seed in range(12):
        rng = random.Random(seed)
        nodes = tuple(f"n{i}" for i in range(rng.randrange(0, 9)))
        graph = DirectedHypergraph(
            nodes,
            {
                f"n{i}": DirectedHyperedge(
                    f"n{i}",
                    [node for node in nodes if rng.random() < 0.3],
                    [node for node in nodes if rng.random() < 0.3],
                    {"index": i, "nested": {"values": [False, None, "label"]}},
                )
                for i in range(rng.randrange(0, 9))
            },
            {"seed": seed},
        )
        # Includes both-role incidences, cycles, node/edge name collisions,
        # empty boundaries, and isolates: all are valid generic structure.
        assert from_incidence(**to_incidence(graph)) == graph
        assert from_networkx(to_networkx(graph)).graph == graph
        assert from_xgi(to_xgi(graph)).graph == graph


def test_hypernetx_requires_explicit_roles_and_preserves_attributes():
    hnx = pytest.importorskip("hypernetx")
    graph = hnx.Hypergraph({"e": ["x", "y"]})
    with pytest.raises(ValueError, match="every incidence"):
        from_hypernetx(graph, roles={})
    result = from_hypernetx(graph, roles={("e", "x"): "both", ("e", "y"): "output"})
    assert result.graph.edges["e"].inputs == ("x",)
    assert result.graph.edges["e"].outputs == ("x", "y")
    assert "weight" in result.graph.edges["e"].metadata
    assert result.notes


def test_tables_reject_ambiguous_labels_or_missing_values():
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    assert records_from_dataframe(pd.DataFrame({"x": [1], "y": [2]})) == [{"x": 1, "y": 2}]
    assert records_from_array(np.array([[1, 2]]), ["x", "y"]) == [{"x": 1, "y": 2}]
    with pytest.raises(ValueError, match="Missing"):
        records_from_dataframe(pd.DataFrame({"x": [None]}))
    with pytest.raises(ValueError, match="unique"):
        records_from_dataframe(pd.DataFrame([[1, 2]], columns=["x", "x"]))
    with pytest.raises(ValueError, match="nonfinite"):
        records_from_array([[float("inf")]], ["x"])
    with pytest.raises(ValueError, match="two-dimensional"):
        records_from_array(np.array([1, 2]), ["x"])


def test_array_adapter_rejects_optional_and_standard_scalar_missing_values():
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    for value in (
        np.float32("nan"),
        np.float32("inf"),
        Decimal("NaN"),
        Decimal("Infinity"),
        pd.NA,
        np.datetime64("NaT", "ns"),
    ):
        with pytest.raises(ValueError, match="nonfinite"):
            records_from_array([[value]], ["x"])
