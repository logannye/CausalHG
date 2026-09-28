import json

import pytest

from causal_hypergraphs import DeleteMechanism, Identified, MechanismGraph, identify
from causal_hypergraphs.expression import Probability, Product, Quotient, SumOut
from causal_hypergraphs.io import SerializationError, dumps, loads


def test_graph_query_estimand_roundtrip():
    graph = MechanismGraph(
        variables=("A", "B", "C", "Y"),
        mechanisms={
            "joint": {"inputs": ("A",), "outputs": ("B", "C")},
            "readout": {"inputs": ("B", "C"), "outputs": ("Y",)},
        },
    )
    query = DeleteMechanism("joint", outcomes=("Y",))
    result = identify(graph, query)
    assert isinstance(result, Identified)
    restored_graph, restored_query, restored_result = loads(dumps((graph, query, result)))
    assert restored_graph == graph
    assert restored_query == query
    assert restored_result.expression.canonical_key() == result.expression.canonical_key()
    assert restored_result.assumptions == result.assumptions
    assert restored_result.derivation == result.derivation
    assert dumps(restored_graph) == dumps(graph)


def test_bound_variables_and_aliases_survive_roundtrip():
    expression = SumOut(
        ("copy",),
        Quotient(Product((Probability("Y", given="copy"), Probability("copy"))), Probability("X")),
    )
    result = Identified(expression, "fixture", (), (), aliases={"copy": "X"})
    restored = loads(dumps(result))
    assert restored.expression.scope() == expression.scope()
    assert restored.expression.footprint() == expression.footprint()
    assert restored.aliases == {"copy": "X"}


def test_tuple_keys_preserve_joint_policy_axes():
    policy = {"joint": {(0, 0): 0.5, (1, 1): 0.5}}
    assert loads(dumps(policy)) == policy


def test_separate_admg_profile_roundtrip_and_identity_validation():
    from causal_hypergraphs import ADMG

    graph = ADMG(("A", "B"), (("A", "B"), ("A", "B")), (("A", "B"),))
    assert graph.directed_edges == (("A", "B"),)
    assert loads(dumps(graph)) == graph
    with pytest.raises(TypeError):
        ADMG((1, "1"))


def test_result_metadata_is_immutable_and_assumption_states_roundtrip():
    from causal_hypergraphs import Assumption, Unknown

    result = Identified(
        Probability("Y"),
        "fixture",
        (Assumption("C1", "checked", "structurally_checked"),),
        (),
        aliases={"copy": "X"},
    )
    with pytest.raises(TypeError):
        result.aliases["copy"] = "Z"  # type: ignore[index]  # intentional mutation rejection
    assert loads(dumps(result)).assumptions[0].state == "structurally_checked"
    assert loads(dumps(Unknown("not supported"))).reason_code == "unsupported_query"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), lambda: None])
def test_reject_executable_or_nonfinite_data(bad):
    with pytest.raises(SerializationError):
        dumps(bad)


def test_reject_unknown_type_version_and_fields():
    document = json.loads(dumps(DeleteMechanism("m")))
    document["schema_version"] = 999
    with pytest.raises(SerializationError):
        loads(json.dumps(document))
    document["schema_version"] = 1
    document["payload"]["type"] = "os.system"
    with pytest.raises(SerializationError):
        loads(json.dumps(document))


def test_compiled_mean_contrast_round_trip_keeps_executable_ast():
    from causal_hypergraphs import (
        CausalQuery,
        EffectContrast,
        HardIntervention,
        compile_query,
        evaluate_query,
    )
    from causal_hypergraphs.semantics import DiscreteModel

    graph = MechanismGraph({"X", "Y"}, {"m": {"inputs": {"X"}, "outputs": {"Y"}}})
    query = EffectContrast(
        CausalQuery(("Y",), HardIntervention({"X": 1}), kind="expectation"),
        CausalQuery(("Y",), HardIntervention({"X": 0}), kind="expectation"),
    )
    compiled = compile_query(graph, query)
    restored = loads(dumps(compiled))
    model = DiscreteModel({"X": (0, 1), "Y": (0, 1)}, {(0, 0): 0.5, (1, 1): 0.5})
    assert evaluate_query(restored, model) == evaluate_query(compiled, model) == {(): 1.0}


def test_estimate_round_trip_retains_support_plan_and_query_label():
    from causal_hypergraphs import (
        CausalQuery,
        Dataset,
        HardIntervention,
        compile_query,
        estimate_query,
    )

    graph = MechanismGraph({"X", "Y"}, {"m": {"inputs": {"X"}, "outputs": {"Y"}}})
    compiled = compile_query(graph, CausalQuery(("Y",), HardIntervention({"X": 1})))
    estimated = estimate_query(compiled, Dataset.from_records([{"X": 0, "Y": 0}, {"X": 1, "Y": 1}]))
    restored = loads(dumps(estimated))
    assert restored == estimated
    assert restored.summary() == estimated.summary()


def test_duplicate_json_keys_are_rejected():
    with pytest.raises(SerializationError):
        loads('{"schema_version": 1, "schema_version": 2, "library_version": "1", "payload": 0}')


def test_noise_record_round_trip_preserves_provenance():
    from causal_hypergraphs import Composite, HardIntervention, NoiseRecord

    record = NoiseRecord(
        {"X": 1}, {"m": (0.5, -1.2)}, intervention=Composite((HardIntervention({"Y": 0}),))
    )
    assert loads(dumps(record)) == record
