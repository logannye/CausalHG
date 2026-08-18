import pytest

from causal_hypergraphs import Mechanism, MechanismGraph


def test_a_cyclic_mechanism_graph_is_built_and_reports_its_cycle() -> None:
    """C1 used to be a construction-time veto. It is now a per-query condition.

    The graph is a legitimate object -- most regulatory networks have a feedback loop, and
    rejecting the whole thing meant no question about any part of it could be asked. What
    a cycle costs is checked where it is actually spent: see `test_cyclic_graphs.py`.
    """
    graph = MechanismGraph(
        variables={"A", "B"},
        mechanisms={
            "m1": {"inputs": {"B"}, "outputs": {"A"}},
            "m2": {"inputs": {"A"}, "outputs": {"B"}},
        },
    )

    assert not graph.is_mechanism_acyclic()
    assert graph.cyclic_mechanisms == frozenset({"m1", "m2"})


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
    """`mechanism_dependencies` must scale with incidences, not with mechanisms squared.

    Comparing all pairs makes a genome-scale graph take tens of seconds before a single
    query is asked, which would leave the whole affordable-query story true of one module
    and false of the tool.

    **Timed on the routine, not on construction, and that is the repair.** This gate used
    to build a `MechanismGraph` and time *that*, on the stated grounds that "every
    MechanismGraph validates acyclicity at construction, so the cost of building the
    dependency graph is the cost of loading a network". That was true when it was written
    and PR #9 made it false: lifting C1 to a query-time condition means construction no
    longer walks the dependency graph at all. The gate went on passing and stopped
    measuring anything -- restoring the pairwise implementation left construction at
    0.0038s/0.0154s, unchanged, while `mechanism_dependencies` itself went from 0.0006s to
    0.2764s at n=1500. A gate whose premise a later change invalidated is worse than no
    gate, because the green tick is read as coverage.

    Measured as a growth ratio rather than a wall-clock budget: quadratic costs about 16x
    for a 4x larger network, linear about 4x, and the gate sits between them with room for
    a noisy machine on either side.
    """
    import time

    def cost(count: int) -> float:
        """The best of several runs, after a warm-up.

        The minimum, not the mean: scheduling noise, allocator state and a cold cache only
        ever *add* time, so the fastest run is the closest estimate of the cost being
        measured. A single sample at this size is dominated by whatever ran before it in
        the session, which made an earlier form of this gate fire when an unrelated test
        file was added -- reporting a scaling regression that had not happened.
        """
        specs = {
            f"m{i}": {"inputs": (f"g{i}", f"g{i + 1}"), "outputs": (f"g{i + 2}",)}
            for i in range(count)
        }
        variables = {f"g{i}" for i in range(count + 3)}
        graph = MechanismGraph(variables=variables, mechanisms=specs)
        graph.mechanism_dependencies()  # warm-up, not timed
        best = float("inf")
        for _ in range(5):
            start = time.perf_counter()
            graph.mechanism_dependencies()
            best = min(best, time.perf_counter() - start)
        return best

    small = cost(1_500)
    large = cost(6_000)

    assert large / small < 8.0, f"{small:.4f}s -> {large:.4f}s is {large / small:.1f}x for 4x"


def test_construction_no_longer_walks_the_dependency_graph() -> None:
    """Pins the fact that invalidated the gate above, so it cannot silently come back.

    Since PR #9 acyclicity is a *query* condition, not a construction veto. If some later
    change reinstates a construction-time walk, the cost story changes and the gate above
    is measuring the wrong thing again -- so the premise is asserted rather than assumed.

    Counted, not timed. A timing proxy here would be the same mistake one layer down: it
    would pass whenever the machine happened to be fast, and it is the *call* that matters.
    """
    original = MechanismGraph.mechanism_dependencies
    calls = 0

    def counting(self: MechanismGraph) -> dict[str, set[str]]:
        nonlocal calls
        calls += 1
        return original(self)

    MechanismGraph.mechanism_dependencies = counting  # type: ignore[method-assign]
    try:
        graph = MechanismGraph(
            variables={"a", "b", "c"},
            mechanisms={
                "m1": {"inputs": ("a",), "outputs": ("b",)},
                "m2": {"inputs": ("b",), "outputs": ("c",)},
            },
        )
        assert calls == 0, f"construction walked the dependency graph {calls} time(s)"

        # And it is still reachable on demand -- the walk moved, it did not disappear.
        graph.is_mechanism_acyclic()
        assert calls == 1
    finally:
        MechanismGraph.mechanism_dependencies = original  # type: ignore[method-assign]


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
