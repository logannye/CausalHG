from collections.abc import Mapping, MutableMapping
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from causal_hypergraphs import DeleteMechanism, Identified, identify
from causal_hypergraphs.graph import (
    CausalModelSpec,
    DirectedHyperedge,
    DirectedHypergraph,
    Mechanism,
    MechanismGraph,
    UnsupportedModelError,
)
from causal_hypergraphs.io import dumps, loads


def test_generic_incidence_retains_parallel_edges_and_overlapping_producers() -> None:
    graph = DirectedHypergraph(
        ("A", "B", "C", "isolated"),
        {
            "first": {"inputs": ("A", "B"), "outputs": ("C",)},
            "parallel": {"inputs": ("A", "B"), "outputs": ("C",)},
        },
    )
    restored = loads(dumps(graph))
    assert restored == graph
    assert tuple(restored.edges) == ("first", "parallel")
    assert "isolated" in restored.nodes
    with pytest.raises(UnsupportedModelError, match="multiple producers"):
        restored.to_mechanism_graph()


def test_self_incidence_is_valid_structure_but_not_a_mechanism_profile() -> None:
    graph = DirectedHypergraph(("A",), {"loop": {"inputs": "A", "outputs": "A"}})
    assert loads(dumps(graph)) == graph
    with pytest.raises(UnsupportedModelError, match="overlapping inputs and outputs"):
        graph.to_mechanism_graph()


def test_cycle_storage_and_conversion_do_not_claim_cyclic_identification() -> None:
    graph = DirectedHypergraph(
        ("a", "b"),
        {
            "ab": {"inputs": "a", "outputs": "b"},
            "ba": {"inputs": "b", "outputs": "a"},
        },
    )
    model = graph.to_mechanism_graph()
    assert not model.is_mechanism_acyclic()
    assert model.cyclic_mechanisms == {"ab", "ba"}
    assert loads(dumps(model)).fingerprint() == model.fingerprint()


def test_profile_conversion_preserves_joint_outputs_and_identification() -> None:
    graph = DirectedHypergraph(
        ("A", "C", "D", "Y"),
        {
            "joint": DirectedHyperedge("joint", inputs="A", outputs=("D", "C")),
            "downstream": {"inputs": ("C", "D"), "outputs": "Y"},
        },
        metadata={"source": "synthetic"},
    )
    specification = CausalModelSpec(graph, fallback_variables=("C", "D"))
    restored = loads(dumps(specification))
    model = restored.to_mechanism_graph()
    assert model.get_mechanism("joint").outputs == ("C", "D")
    assert model.fallback_variables == ("C", "D")
    assert restored.graph.metadata == {"source": "synthetic"}
    result = identify(model, DeleteMechanism("joint", outcomes="Y"))
    assert isinstance(result, Identified)
    assert "P0_joint(C,D)" in str(result.expression)
    assert restored.fingerprint() == specification.fingerprint()


def test_unspecified_or_unknown_profile_never_selects_a_different_causal_theory() -> None:
    graph = DirectedHypergraph(("A", "B"), {"ab": {"inputs": "A", "outputs": "B"}})
    specification = CausalModelSpec(graph, profile="pearl-admg")
    with pytest.raises(UnsupportedModelError, match="ADMGs have their own interface"):
        specification.to_mechanism_graph()
    with pytest.raises(ValueError, match="Observed variables not in graph"):
        CausalModelSpec(graph, observed_variables=("missing",))


@pytest.mark.parametrize("bad", [1, None, True, b"name"])
def test_native_ids_reject_implicit_conversion(bad: Any) -> None:
    with pytest.raises(TypeError, match="nonempty strings"):
        DirectedHypergraph((bad, "1"), {})
    with pytest.raises(TypeError, match="nonempty strings"):
        MechanismGraph(("A",), {bad: {"outputs": "A"}})
    with pytest.raises(TypeError, match="nonempty strings"):
        Mechanism(bad)


@pytest.mark.parametrize("bad", ["", " ", "\t"])
def test_empty_native_ids_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError, match="nonempty strings"):
        DirectedHyperedge(bad)
    with pytest.raises(ValueError, match="nonempty strings"):
        MechanismGraph((bad,), {})


def test_duplicate_incidence_and_name_disagreement_are_explicit_errors() -> None:
    with pytest.raises(ValueError, match="Duplicate identifiers"):
        DirectedHyperedge("edge", outputs=("A", "A"))
    with pytest.raises(ValueError, match="disagrees"):
        DirectedHypergraph(("A",), {"key": DirectedHyperedge("other", outputs="A")})
    with pytest.raises(ValueError, match="missing nodes"):
        DirectedHypergraph(("A",), {"edge": {"outputs": "B"}})


def test_generic_namespaces_do_not_silently_collapse_during_causal_conversion() -> None:
    graph = DirectedHypergraph(("same",), {"same": {"outputs": "same"}})
    assert loads(dumps(graph)) == graph
    with pytest.raises(UnsupportedModelError, match="distinct node and mechanism IDs"):
        graph.to_mechanism_graph()


def test_legacy_mechanism_graph_inputs_are_snapshotted_and_read_only() -> None:
    variables = ["A", "B"]
    outputs = ["B"]
    specifications = {"step": {"inputs": ["A"], "outputs": outputs}}
    graph = MechanismGraph(variables, specifications, variables, outputs, ("declared",))
    fingerprint = graph.fingerprint()
    variables.append("later")
    outputs.append("later")
    specifications.clear()
    assert graph.variables == ("A", "B")
    assert graph.get_mechanism("step").outputs == ("B",)
    assert graph.fingerprint() == fingerprint
    with pytest.raises(TypeError):
        cast(MutableMapping[str, Mechanism], graph.mechanisms)["other"] = Mechanism("other")
    with pytest.raises(FrozenInstanceError):
        cast(Any, graph).variables = ("changed",)
    assert loads(dumps(graph)) == graph
    assert hash(loads(dumps(graph))) == hash(graph)


def test_nested_metadata_is_copied_and_immutable_after_roundtrip() -> None:
    nested: dict[str, Any] = {"tags": ["original"], "nested": {"count": 3}}
    edge = DirectedHyperedge("edge", outputs="A", metadata=nested)
    graph = DirectedHypergraph(("A",), {"edge": edge}, metadata=nested)
    original = graph.fingerprint()
    nested["tags"].append("changed")
    nested["nested"]["count"] = 4
    restored = loads(dumps(graph))
    assert restored == graph
    assert graph.metadata["tags"] == ("original",)
    assert restored.fingerprint() == original
    with pytest.raises(TypeError):
        cast(MutableMapping[str, object], restored.edges)["new"] = edge
    with pytest.raises(TypeError):
        cast(MutableMapping[str, object], graph.metadata)["new"] = "value"
    with pytest.raises(TypeError):
        nested_mapping = cast(Mapping[str, object], edge.metadata["nested"])
        cast(MutableMapping[str, object], nested_mapping)["count"] = 9
    assert hash(restored) == hash(graph)


@pytest.mark.parametrize("bad", [lambda: None, float("nan"), float("inf"), {1: "value"}])
def test_metadata_rejects_non_data_values(bad: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        DirectedHypergraph(("A",), {}, metadata={"bad": bad})


def test_metadata_rejects_recursive_values() -> None:
    recursive: list[Any] = []
    recursive.append(recursive)
    with pytest.raises(ValueError, match="recursive"):
        DirectedHypergraph(("A",), {}, metadata={"recursive": recursive})


def test_fingerprints_are_order_independent_and_include_semantic_declarations() -> None:
    left = MechanismGraph(
        ("B", "A"), {"step": {"outputs": "B", "inputs": "A"}},
        observed_variables=("B", "A"),
    )
    right = MechanismGraph(
        ("A", "B"), {"step": {"inputs": "A", "outputs": "B"}},
        observed_variables=("A", "B"),
    )
    hidden = MechanismGraph(left.variables, left.mechanisms, observed_variables=("B",))
    no_policy = MechanismGraph(left.variables, left.mechanisms, fallback_variables=())
    assert left == right
    assert left.fingerprint() == right.fingerprint()
    assert left.fingerprint() != hidden.fingerprint()
    assert left.fingerprint() != no_policy.fingerprint()
    assert len(left.fingerprint()) == 64
    first = DirectedHypergraph(
        ("A", "B"), {"step": {"inputs": "A", "outputs": "B"}},
        metadata={"b": 2, "a": 1},
    )
    second = DirectedHypergraph(
        ("B", "A"), {"step": {"outputs": "B", "inputs": "A"}},
        metadata={"a": 1, "b": 2},
    )
    assert first.fingerprint() == second.fingerprint()
    assert dumps(first) == dumps(second)
