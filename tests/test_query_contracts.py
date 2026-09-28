"""Stable query contracts preserve joint axes, identity, and intervention boundaries."""

from dataclasses import FrozenInstanceError

import pytest

from causal_hypergraphs.graph import MechanismGraph
from causal_hypergraphs.queries import (
    CausalQuery,
    Composite,
    Delete,
    EffectContrast,
    HardIntervention,
    JointPolicy,
    Replace,
    validate_intervention,
)


def _graph() -> MechanismGraph:
    return MechanismGraph(
        variables=("A", "B", "C", "Y"),
        mechanisms={
            "joint": {"inputs": ("A",), "outputs": ("B", "C")},
            "readout": {"inputs": ("B", "C"), "outputs": ("Y",)},
        },
    )


def test_query_snapshots_mutable_inputs_and_is_hashable() -> None:
    assignments = {"B": 0}
    intervention = HardIntervention(assignments)
    query = CausalQuery(("Y",), intervention, kind="expectation")
    assignments["B"] = 1
    assert intervention.assignments == {"B": 0}
    assert hash(query) == hash(CausalQuery(("Y",), HardIntervention({"B": 0}), "expectation"))
    with pytest.raises(TypeError):
        intervention.assignments["B"] = 2  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        query.kind = "distribution"  # type: ignore[misc]


def test_policy_preserves_explicit_non_alphabetical_axes_and_sparse_joint_support() -> None:
    masses = {(0, 1): 0.4, (1, 0): 0.6}
    policy = JointPolicy(("C", "B"), masses)
    masses[(0, 1)] = 0.9
    assert policy.variables == ("C", "B")
    assert policy.probability({"B": 1, "C": 0}) == 0.4
    assert policy.probability({"B": 0, "C": 0}) == 0
    assert sum(policy.probabilities.values()) == 1
    assert hash(policy) == hash(JointPolicy(("C", "B"), {(1, 0): 0.6, (0, 1): 0.4}))
    with pytest.raises(TypeError):
        policy.probabilities[(0, 0)] = 1  # type: ignore[index]


@pytest.mark.parametrize("bad", [1, None, "", "  "])
def test_identifiers_are_never_silently_coerced(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        HardIntervention({bad: 0})  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        JointPolicy((bad,), {(0,): 1})  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        Replace(bad, "new")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), [], None])
def test_nonfinite_or_mutable_intervention_values_are_rejected(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        HardIntervention({"B": bad})  # type: ignore[arg-type]
    if isinstance(bad, float):
        with pytest.raises(ValueError):
            JointPolicy(("B",), {(bad,): 1})


@pytest.mark.parametrize(
    "masses",
    [{}, {(0,): 0.4}, {(0,): -0.1, (1,): 1.1}, {(0,): float("nan")}, {(0,): float("inf")}],
)
def test_invalid_policy_mass_fails_at_construction(masses: dict) -> None:
    with pytest.raises(ValueError):
        JointPolicy(("B",), masses)


def test_duplicate_axes_and_malformed_table_coordinates_are_rejected() -> None:
    with pytest.raises(TypeError, match="axis order"):
        JointPolicy({"B", "C"}, {(0, 0): 1})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate"):
        JointPolicy(("B", "B"), {(0, 0): 1})
    with pytest.raises(ValueError, match="axes"):
        JointPolicy(("B", "C"), {(0,): 1})
    with pytest.raises(ValueError, match="axes"):
        JointPolicy(("B",), {0: 1})  # type: ignore[arg-type]


def test_composite_rejects_known_conflicts_without_a_graph() -> None:
    policy = JointPolicy(("B", "C"), {(0, 0): 1})
    for operations in (
        (HardIntervention({"B": 0}), HardIntervention({"B": 0})),
        (policy, HardIntervention({"C": 1})),
        (Delete("joint", policy), Replace("joint", "new")),
    ):
        with pytest.raises(ValueError, match="Conflicting"):
            Composite(operations)


def test_graph_validation_rejects_overlap_with_replacement_and_unknown_targets() -> None:
    for operations in (
        (Replace("joint", "new"), HardIntervention({"C": 1})),
        (HardIntervention({"C": 1}), Replace("joint", "new")),
    ):
        with pytest.raises(ValueError, match="Conflicting"):
            validate_intervention(_graph(), Composite(operations))
    with pytest.raises(ValueError, match="unknown"):
        validate_intervention(_graph(), HardIntervention({"missing": 0}))
    with pytest.raises(KeyError):
        validate_intervention(_graph(), Replace("missing", "new"))


def test_delete_policy_must_cover_all_outputs_and_can_use_a_different_axis_order() -> None:
    with pytest.raises(ValueError, match="exactly"):
        validate_intervention(_graph(), Delete("joint", JointPolicy(("B",), {(0,): 1})))
    deletion = Delete("joint", JointPolicy(("C", "B"), {(0, 1): 1}))
    operations = (deletion, HardIntervention({"A": 1}))
    assert validate_intervention(_graph(), Composite(operations)) == operations


def test_nested_composites_and_noop_have_unambiguous_flat_semantics() -> None:
    operation = HardIntervention({"B": 0})
    nested = Composite((Composite((operation,)),))  # type: ignore[arg-type]
    assert nested.operations == (operation,)
    assert validate_intervention(_graph(), Composite(())) == ()
    assert validate_intervention(_graph(), HardIntervention({})) == (HardIntervention({}),)


def test_quantity_and_contrast_contracts_reject_ambiguous_requests() -> None:
    do = HardIntervention({"B": 1})
    with pytest.raises(ValueError, match="one outcome"):
        CausalQuery(("C", "Y"), do, kind="expectation")
    with pytest.raises(ValueError, match="disjoint"):
        CausalQuery(("Y",), do, given=("Y",))
    with pytest.raises(ValueError, match="duplicate"):
        CausalQuery(("Y", "Y"), do)
    with pytest.raises(ValueError, match="kind"):
        CausalQuery(("Y",), do, kind="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="matching"):
        EffectContrast(CausalQuery(("Y",), do), CausalQuery(("C",), do))
    left = CausalQuery(("Y",), do, "expectation", given=("A",))
    right = CausalQuery(("Y",), HardIntervention({"B": 0}), "expectation", given=("A",))
    assert EffectContrast(left, right).left == left
