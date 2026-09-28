"""A nonidentification witness must affect the requested outcome, not any measurement."""

from causal_hypergraphs import DeleteMechanism, Identified, MechanismGraph, Unidentified, identify


def _graph() -> MechanismGraph:
    return MechanismGraph(
        variables=("h", "X", "Y", "Z"),
        mechanisms={
            "hidden": {"inputs": (), "outputs": ("h",)},
            "first": {"inputs": ("h",), "outputs": ("X",)},
            "second": {"inputs": ("X",), "outputs": ("Y",)},
            "unrelated": {"inputs": (), "outputs": ("Z",)},
        },
        observed_variables=("X", "Y", "Z"),
    )


def test_hidden_intervention_cannot_refute_an_unaffected_outcome() -> None:
    result = identify(_graph(), DeleteMechanism("hidden", outcomes=("Z",)), allow_t7=True)
    assert isinstance(result, Identified)
    assert str(result.expression) == "P(Z)"


def test_hidden_reach_traverses_intermediate_observed_variables_to_requested_outcome() -> None:
    result = identify(_graph(), DeleteMechanism("hidden", outcomes=("Y",)), allow_t7=True)
    assert isinstance(result, Unidentified)
    assert result.witness.observed_descendants == ("Y",)  # type: ignore[union-attr]


def test_joint_query_including_an_affected_outcome_retains_the_witness() -> None:
    result = identify(_graph(), DeleteMechanism("hidden", outcomes=("Y", "Z")), allow_t7=True)
    assert isinstance(result, Unidentified)


def test_unobserved_outcome_is_rejected_before_hidden_witness_reasoning() -> None:
    from causal_hypergraphs import Unknown

    graph = MechanismGraph(
        variables=("h", "unobserved", "Y"),
        mechanisms={"hidden": {"inputs": (), "outputs": ("h",)}},
        observed_variables=("Y",),
    )
    result = identify(
        graph, DeleteMechanism("hidden", outcomes=("unobserved",)), allow_t7=True
    )
    assert isinstance(result, Unknown)
    assert "observed" in result.reason
