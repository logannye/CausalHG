"""Whole-set DAG adjustment checks, with confounding and collider oracles."""

import pytest

from causal_hypergraphs.graph import MechanismGraph
from causal_hypergraphs.identification.adjustment import validate_adjustment_set
from causal_hypergraphs.identification.pearl_id import ADMG


def test_confounder_is_required_and_the_whole_set_gets_a_certificate() -> None:
    dag = ADMG(("C", "X", "Y"), (("C", "X"), ("C", "Y"), ("X", "Y")))
    empty = validate_adjustment_set(dag, "X", "Y")
    adjusted = validate_adjustment_set(dag, "X", "Y", ("C",))
    assert empty.status == "fail"
    assert empty.connected_pairs == (("X", "Y"),)
    assert adjusted.status == "pass"
    assert adjusted.reason_code == "backdoor_criterion_satisfied"
    states = {assumption.code: assumption.state for assumption in adjusted.assumptions}
    assert states["Backdoor criterion"] == "structurally_checked"
    assert states["Population positivity"] == "unresolved"


def test_baseline_collider_can_make_an_entire_set_invalid() -> None:
    # The path X <- A -> C <- B -> Y is blocked without C, and opened by C.
    dag = ADMG(
        ("A", "B", "C", "X", "Y"),
        (
            ("A", "X"),
            ("A", "C"),
            ("B", "C"),
            ("B", "Y"),
            ("X", "Y"),
        ),
    )
    assert validate_adjustment_set(dag, "X", "Y").status == "pass"
    assert validate_adjustment_set(dag, "X", "Y", ("C",)).status == "fail"
    # C's individual hazard does not invalidate {A,C}: A blocks the opened path.
    assert validate_adjustment_set(dag, "X", "Y", ("A", "C")).status == "pass"


def test_descendant_of_collider_can_open_a_backdoor_path() -> None:
    dag = ADMG(
        ("A", "B", "C", "D", "X", "Y"),
        (
            ("A", "X"),
            ("A", "C"),
            ("B", "C"),
            ("B", "Y"),
            ("C", "D"),
        ),
    )
    result = validate_adjustment_set(dag, "X", "Y", "D")
    assert result.status == "fail"
    assert result.reason_code == "open_backdoor_path"


def test_mediator_and_other_treatment_descendants_fail_original_graph_check() -> None:
    dag = ADMG(("X", "M", "D", "Y"), (("X", "M"), ("M", "Y"), ("X", "D")))
    assert validate_adjustment_set(dag, "X", "Y").status == "pass"
    for covariate in ("M", "D"):
        result = validate_adjustment_set(dag, "X", "Y", covariate)
        assert result.status == "fail"
        assert result.reason_code == "treatment_descendant"
        assert result.descendants_in_set == (covariate,)


def test_multiple_treatment_and_outcome_sets_are_checked_pairwise() -> None:
    dag = ADMG(
        ("C", "X1", "X2", "Y1", "Y2"),
        (
            ("C", "X2"),
            ("C", "Y2"),
            ("X1", "Y1"),
            ("X2", "Y2"),
        ),
    )
    result = validate_adjustment_set(dag, ("X1", "X2"), ("Y1", "Y2"))
    assert result.connected_pairs == (("X2", "Y2"),)
    assert validate_adjustment_set(dag, ("X1", "X2"), ("Y1", "Y2"), "C").status == "pass"


def test_single_output_mechanism_graph_uses_the_same_dag_criterion() -> None:
    graph = MechanismGraph(
        ("C", "X", "Y"),
        {
            "treatment": {"inputs": ("C",), "outputs": ("X",)},
            "response": {"inputs": ("C", "X"), "outputs": ("Y",)},
        },
    )
    assert validate_adjustment_set(graph, "X", "Y", "C").status == "pass"
    assert validate_adjustment_set(graph, "X", "Y").status == "fail"


def test_other_model_families_are_explicitly_unsupported() -> None:
    admg = ADMG(("X", "Y"), (("X", "Y"),), (("X", "Y"),))
    assert validate_adjustment_set(admg, "X", "Y").reason_code == "bidirected_model"
    joint = MechanismGraph(("X", "Y"), {"m": {"outputs": ("X", "Y")}})
    assert validate_adjustment_set(joint, "X", "Y").reason_code == "joint_output_mechanism"
    hidden = MechanismGraph(("H", "X", "Y"), {}, observed_variables=("X", "Y"))
    assert validate_adjustment_set(hidden, "X", "Y").reason_code == "hidden_variables"
    cyclic = MechanismGraph(
        ("X", "Y"),
        {
            "a": {"inputs": ("X",), "outputs": ("Y",)},
            "b": {"inputs": ("Y",), "outputs": ("X",)},
        },
    )
    result = validate_adjustment_set(cyclic, "X", "Y")
    assert result.status == "unsupported" and result.reason_code == "cyclic_model"


@pytest.mark.parametrize(
    "x,y,z",
    [
        ((), ("Y",), ()),
        (("X",), ("X",), ()),
        (("X",), ("Y",), ("X",)),
        (("X",), ("Y",), ("Y",)),
        (("X", "X"), ("Y",), ()),
        (("missing",), ("Y",), ()),
    ],
)
def test_malformed_sets_fail_before_graphical_reasoning(x: tuple, y: tuple, z: tuple) -> None:
    with pytest.raises(ValueError):
        validate_adjustment_set(ADMG(("X", "Y")), x, y, z)


def test_integer_identifiers_are_not_coerced() -> None:
    with pytest.raises(TypeError):
        validate_adjustment_set(ADMG(("1", "Y")), (1,), "Y")  # type: ignore[arg-type]
