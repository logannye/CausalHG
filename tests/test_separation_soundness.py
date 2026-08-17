"""Soundness tests for d*-separation.

`d_separated` is a conditional-independence *oracle*: a `True` verdict licenses an
independence claim. A false `True` is therefore the worst failure this module can have,
and these tests target that direction specifically.
"""
from __future__ import annotations

from causal_hypergraphs.examples import reaction_graph
from causal_hypergraphs.graph import Mechanism, MechanismGraph
from causal_hypergraphs.separation import d_separated, deterministic_closure


def test_equality_closure_pulls_in_the_coupled_output() -> None:
    """Premise for the tests below: conditioning on D determines C, since m1 declares C = D."""
    assert deterministic_closure(reaction_graph(), {"D"}) == frozenset({"C", "D"})


def test_partially_determined_x_does_not_make_the_whole_set_separated() -> None:
    """X = {C, E} is not separated from {F} given {D}, even though D determines C.

    C is determined by the conditioning set, so it carries no information. E is not:
    E -> m2 -> F is a direct open path in the bipartite blowup. Treating "some element of
    X is determined" as "X is separated" discards the rest of X and licenses an
    independence that does not hold.
    """
    graph = reaction_graph()
    assert d_separated(graph, {"E"}, {"F"}, given={"D"}) is False
    assert d_separated(graph, {"C", "E"}, {"F"}, given={"D"}) is False


def test_fully_determined_x_is_separated() -> None:
    """The sound half of the rule: if *all* of X is determined by Z, X is constant given Z."""
    assert d_separated(reaction_graph(), {"C"}, {"F"}, given={"D"}) is True


def _diamond_chain(depth: int) -> MechanismGraph:
    """A chain of `depth` diamonds: v0 -> {a_i, b_i} -> v_i -> ...

    The number of simple paths from v0 to v_depth is 2**depth, so any algorithm that
    enumerates simple paths is exponential here. Each variable has exactly one producing
    mechanism, so C4 holds.
    """
    variables = {"v0"}
    mechanisms: dict[str, Mechanism] = {}
    for i in range(1, depth + 1):
        previous, a, b, current = f"v{i - 1}", f"a{i}", f"b{i}", f"v{i}"
        variables |= {a, b, current}
        mechanisms[f"m{i}a"] = Mechanism(f"m{i}a", inputs=(previous,), outputs=(a,))
        mechanisms[f"m{i}b"] = Mechanism(f"m{i}b", inputs=(previous,), outputs=(b,))
        mechanisms[f"m{i}c"] = Mechanism(f"m{i}c", inputs=(a, b), outputs=(current,))
    return MechanismGraph(variables=variables, mechanisms=mechanisms)


def test_separation_is_exact_on_a_graph_with_exponentially_many_paths() -> None:
    """A verdict must not depend on how many paths an implementation got around to looking at.

    With depth 24 there are 2**24 simple paths from v0 to v24. Enumerating them is
    infeasible, and truncating the enumeration silently converts "I stopped looking" into
    "separated". Both verdicts below are decided by reachability, not by enumeration.
    """
    graph = _diamond_chain(24)

    # A directed path exists, so the endpoints are d-connected given nothing.
    assert d_separated(graph, {"v0"}, {"v24"}) is False
    # Conditioning on the first diamond's outputs blocks every route out of v0.
    assert d_separated(graph, {"v0"}, {"v24"}, given={"a1", "b1"}) is True
