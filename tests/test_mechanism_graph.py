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


def test_the_dependency_graph_is_built_by_index_not_by_comparing_every_pair() -> None:
    """Construction must scale with edges, not with mechanisms squared.

    Every `MechanismGraph` validates acyclicity at construction, so the cost of building
    the dependency graph is the cost of *loading* a network. Comparing all pairs makes a
    genome-scale graph take tens of seconds before a single query is asked -- which would
    leave the whole affordable-query story true of one module and false of the tool.

    Measured as a growth ratio rather than a wall-clock budget: quadratic construction
    costs about 16x for a 4x larger network, linear about 4x. The gate sits between them
    with room for a noisy machine on either side, and it fires against the pairwise
    implementation this replaced.
    """
    import time

    def build(count: int) -> float:
        specs = {
            f"m{i}": {"inputs": (f"g{i}", f"g{i + 1}"), "outputs": (f"g{i + 2}",)}
            for i in range(count)
        }
        variables = {f"g{i}" for i in range(count + 3)}
        start = time.perf_counter()
        MechanismGraph(variables=variables, mechanisms=specs)
        return time.perf_counter() - start

    small = build(1_500)
    large = build(6_000)

    assert large / small < 8.0, f"{small:.3f}s -> {large:.3f}s is {large / small:.1f}x for 4x"


def test_the_dependency_graph_still_names_the_right_successors() -> None:
    """The index must produce exactly what the pairwise comparison did.

    A mechanism depends on another when one of its outputs is one of the other's inputs.
    Shared inputs create no edge, and a mechanism never depends on itself.
    """
    graph = MechanismGraph(
        variables={"a", "b", "c", "d", "e"},
        mechanisms={
            "source": {"inputs": ("a",), "outputs": ("b", "c")},
            "left": {"inputs": ("b",), "outputs": ("d",)},
            "right": {"inputs": ("c",), "outputs": ("e",)},
            "sibling": {"inputs": ("a",), "outputs": ()},
        },
    )

    assert graph.mechanism_dependencies() == {
        "source": {"left", "right"},
        "left": set(),
        "right": set(),
        "sibling": set(),
    }
