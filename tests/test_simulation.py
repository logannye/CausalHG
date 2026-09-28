"""Independent structural and analytic oracles for the executable SCM layer."""

import random
from typing import Any, cast

import numpy as np
import pytest

from causal_hypergraphs.graph import MechanismGraph
from causal_hypergraphs.queries import Composite, Delete, HardIntervention, JointPolicy, Replace
from causal_hypergraphs.simulation import (
    FiniteKernel,
    HypergraphSCM,
    LinearGaussianMechanism,
    NoiseRecord,
    StructuralMechanism,
)


def _joint_scm() -> HypergraphSCM:
    graph = MechanismGraph(
        variables=("A", "B", "C", "Y"),
        mechanisms={
            "joint": {"inputs": ("A",), "outputs": ("B", "C")},
            "readout": {"inputs": ("B", "C"), "outputs": ("Y",)},
        },
    )

    def joint(inputs: Any, noise: Any) -> dict:
        return {"B": inputs["A"] + noise, "C": inputs["A"] + noise}

    return HypergraphSCM(
        graph,
        {
            "joint": StructuralMechanism(
                ("A",),
                ("B", "C"),
                joint,
                lambda rng: int(rng.random() < 0.5),
                abduct=lambda inputs, outputs: float(outputs["B"]) - float(inputs["A"]),
            ),
            "readout": StructuralMechanism(
                ("B", "C"),
                ("Y",),
                lambda x, _: {"Y": float(x["B"]) + float(x["C"])},
                lambda _: None,
                abduct=lambda _x, _y: None,
            ),
        },
        {"A": lambda _: 0},
    )


def test_hard_do_preserves_untargeted_cooutput_and_its_individual_noise() -> None:
    scm = _joint_scm()
    for seed in range(20):
        factual, noise = scm.sample_with_noise(seed=seed)
        assert factual["B"] == factual["C"]
        counterfactual = scm.counterfactual(HardIntervention({"B": 0}), noise=noise)
        assert counterfactual["B"] == 0
        assert counterfactual["C"] == factual["C"]
        assert counterfactual["Y"] == factual["C"]
    rows = scm.sample(5000, seed=911, intervention=HardIntervention({"B": 0}))
    assert np.mean([float(row["Y"]) for row in rows]) == pytest.approx(0.5, abs=0.025)


def test_root_do_overrides_factual_exogenous_value_without_losing_mechanism_noise() -> None:
    scm = _joint_scm()
    factual, noise = scm.sample_with_noise(seed=12)
    alternative = scm.counterfactual(HardIntervention({"A": 3}), noise=noise)
    assert alternative["A"] == 3
    assert alternative["B"] == float(factual["B"]) + 3
    assert alternative["C"] == float(factual["C"]) + 3
    assert alternative["Y"] == float(factual["Y"]) + 6


def test_replay_is_exact_and_does_not_sample_or_advance_rng() -> None:
    calls = []
    graph = MechanismGraph(("Y",), {"m": {"outputs": ("Y",)}})

    def sampler(rng: random.Random) -> object:
        calls.append(1)
        return rng.random()

    scm = HypergraphSCM(
        graph,
        {"m": StructuralMechanism((), ("Y",), lambda _, noise: {"Y": cast(float, noise)}, sampler)},
    )
    rng = random.Random(22)
    factual, noise = scm.sample_with_noise(rng=rng)
    rng_state = rng.getstate()
    for _ in range(3):
        assert scm.evaluate_with_noise(noise) == factual
        assert scm.counterfactual(Composite(()), noise=noise) == factual
    assert len(calls) == 1
    assert rng.getstate() == rng_state
    with pytest.raises(ValueError, match="Missing mechanism noise"):
        scm.evaluate_with_noise(NoiseRecord({}, {}))


def test_joint_deletion_policy_is_not_split_and_uses_declared_axis_order() -> None:
    scm = _joint_scm()
    policy = JointPolicy(("C", "B"), {(0, 1): 0.25, (1, 0): 0.75}, name="switch")
    deletion = Delete("joint", policy)
    rows = scm.sample(5000, seed=882, intervention=deletion)
    assert all(float(row["B"]) + float(row["C"]) == row["Y"] == 1 for row in rows)
    assert np.mean([float(row["C"]) for row in rows]) == pytest.approx(0.75, abs=0.025)
    factual, noise = scm.sample_with_noise(seed=4, intervention=deletion)
    assert scm.evaluate_with_noise(noise) == factual
    assert noise.intervention == deletion
    assert scm.sample(20, seed=3) == scm.sample(20, seed=3)
    assert all(row["B"] == row["C"] for row in scm.sample(20, seed=3))


def test_a_structural_function_cannot_mutate_the_saved_noise_during_replay() -> None:
    graph = MechanismGraph(("Y",), {"m": {"outputs": ("Y",)}})

    def consume(_inputs: Any, noise: Any) -> dict:
        noise[0] += 1
        return {"Y": noise[0]}

    scm = HypergraphSCM(
        graph, {"m": StructuralMechanism((), ("Y",), consume, lambda _rng: [10])}
    )
    factual, noise = scm.sample_with_noise(seed=7)
    assert factual == {"Y": 11}
    assert noise.mechanisms["m"] == [10]
    for _ in range(3):
        assert scm.evaluate_with_noise(noise) == factual
        assert noise.mechanisms["m"] == [10]


def test_empty_hard_interventions_are_observational_noise_provenance() -> None:
    scm = _joint_scm()
    for empty in (HardIntervention({}), Composite((HardIntervention({}),))):
        factual, noise = scm.sample_with_noise(seed=7, intervention=empty)
        changed = scm.counterfactual(HardIntervention({"A": 2}), noise=noise)
        assert float(changed["B"]) == float(factual["B"]) + 2
        assert float(changed["C"]) == float(factual["C"]) + 2


def test_joint_policy_can_span_outputs_of_different_mechanisms_and_roots() -> None:
    scm = _joint_scm()
    policy = JointPolicy(("A", "Y"), {(1, 7): 0.3, (2, 9): 0.7})
    rows = scm.sample(100, seed=4, intervention=policy)
    assert all((row["A"], row["Y"]) in policy.probabilities for row in rows)
    assert all(row["B"] == row["C"] for row in rows)


def test_replacement_validates_incidence_and_retains_joint_outputs() -> None:
    scm = _joint_scm()
    replacement = StructuralMechanism(
        ("A",), ("C", "B"), lambda x, _: {"C": float(x["A"]) + 2, "B": 5}, lambda _: None
    )
    operation = Replace("joint", "new")
    bindings = {"new": replacement}
    row, noise = scm.sample_with_noise(seed=3, intervention=operation, replacements=bindings)
    assert row == {"A": 0, "B": 5, "C": 2, "Y": 7}
    assert scm.evaluate_with_noise(noise, replacements=bindings) == row
    with pytest.raises(ValueError, match="Missing executable replacement"):
        scm.sample(intervention=operation)
    with pytest.raises(ValueError, match="incidence"):
        scm.sample(
            intervention=operation,
            replacements={
                "new": StructuralMechanism((), ("B",), lambda _x, _u: {"B": 0}, lambda _: None)
            },
        )
    with pytest.raises(ValueError, match="Conflicting"):
        scm.sample(
            intervention=Composite((operation, HardIntervention({"B": 1}))), replacements=bindings
        )


def test_finite_joint_sampling_matches_exact_marginalization() -> None:
    graph = MechanismGraph(("A", "B", "C"), {"joint": {"inputs": ("A",), "outputs": ("B", "C")}})
    kernel = FiniteKernel(
        ("A",),
        ("C", "B"),
        {(0,): {(0, 0): 0.8, (1, 1): 0.2}, (1,): {(0, 1): 0.3, (1, 0): 0.7}},
        input_domains={"A": (0, 1)},
    )
    scm = HypergraphSCM(graph, {"joint": kernel}, {"A": lambda rng: int(rng.random() < 0.4)})
    rows = scm.sample(10000, seed=571)
    # Direct arithmetic on the declared population, independent of simulator logic.
    expected_b = 0.6 * 0.2 + 0.4 * 0.3
    expected_c = 0.6 * 0.2 + 0.4 * 0.7
    assert np.mean([float(row["B"]) for row in rows]) == pytest.approx(expected_b, abs=0.02)
    assert np.mean([float(row["C"]) for row in rows]) == pytest.approx(expected_c, abs=0.02)
    factual, noise = scm.sample_with_noise(seed=9)
    assert scm.evaluate_with_noise(noise) == factual
    with pytest.raises(ValueError, match="Kernel-only"):
        scm.counterfactual(HardIntervention({"A": 0}), noise=noise)
    with pytest.raises(ValueError, match="Kernel-only"):
        scm.evaluate_with_noise(noise, intervention=HardIntervention({"A": 0}))
    explicit = HypergraphSCM(graph, {"joint": kernel.as_structural()}, scm.exogenous)
    factual, noise = explicit.sample_with_noise(seed=9)
    assert explicit.counterfactual(Composite(()), noise=noise) == factual


@pytest.mark.parametrize(
    "table",
    [
        {(0,): {(0,): 1}},
        {(0,): {(0,): 0.9}, (1,): {(1,): 1}},
        {(0,): {(0,): -0.1, (1,): 1.1}, (1,): {(1,): 1}},
        {(0,): {(0, 1): 1}, (1,): {(1,): 1}},
    ],
)
def test_bad_finite_tables_are_rejected_before_sampling(table: dict) -> None:
    with pytest.raises(ValueError):
        FiniteKernel(("A",), ("B",), table, input_domains={"A": (0, 1)})


def test_conditioned_finite_kernel_requires_complete_declared_input_domains() -> None:
    with pytest.raises(ValueError, match="explicit domain"):
        FiniteKernel(("A",), ("B",), {(0,): {(0,): 1}})
    root = FiniteKernel((), ("Y",), {(): {(0,): 0.5, (1,): 0.5}})
    assert root.evaluate({}, 0.75) == {"Y": 1}


def test_gaussian_joint_moments_match_affine_closed_form() -> None:
    graph = MechanismGraph(("X", "Y", "Z"), {"joint": {"inputs": ("X",), "outputs": ("Y", "Z")}})
    binding = LinearGaussianMechanism(("X",), ("Y", "Z"), [[2], [-1]], [1, 3], [[1, 0.5], [0.5, 2]])
    scm = HypergraphSCM(graph, {"joint": binding}, {"X": lambda rng: rng.gauss(2, 1)})
    rows = scm.sample(12000, seed=774)
    yz = np.array([[row["Y"], row["Z"]] for row in rows])
    assert np.mean(yz, axis=0) == pytest.approx([5, 1], abs=0.07)
    assert np.cov(yz, rowvar=False) == pytest.approx(np.array([[5, -1.5], [-1.5, 3]]), abs=0.12)
    factual, noise = scm.sample_with_noise(seed=27)
    cf = scm.counterfactual(HardIntervention({"X": 3}), noise=noise)
    from_observation = scm.counterfactual(HardIntervention({"X": 3}), observation=factual)
    assert cf == pytest.approx(from_observation)
    assert float(cf["Y"]) - float(factual["Y"]) == pytest.approx(2 * (3 - float(factual["X"])))
    assert float(cf["Z"]) - float(factual["Z"]) == pytest.approx(-(3 - float(factual["X"])))


def test_singular_gaussian_noise_preserves_constraint_and_abduction_support() -> None:
    binding = LinearGaussianMechanism((), ("B", "C"), [[], []], [0, 0], [[1, 1], [1, 1]])
    graph = MechanismGraph(("B", "C"), {"m": {"outputs": ("B", "C")}})
    scm = HypergraphSCM(graph, {"m": binding})
    for row in scm.sample(20, seed=12):
        assert row["B"] == pytest.approx(row["C"], abs=1e-12)
        assert scm.evaluate_with_noise(scm.abduct(row)) == pytest.approx(row)
    with pytest.raises(ValueError, match="support"):
        scm.abduct({"B": 0, "C": 1})
    with pytest.raises(ValueError, match="semidefinite"):
        LinearGaussianMechanism((), ("B", "C"), [[], []], [0, 0], [[1, 2], [2, 1]])
    with pytest.raises(ValueError, match="dimensions"):
        LinearGaussianMechanism(("A",), ("B",), [[]], [0], [[1]])
    with pytest.raises(ValueError, match="finite"):
        LinearGaussianMechanism((), ("B",), [[]], [float("nan")], [[1]])


def test_abduction_checks_structural_consistency_and_requires_full_state() -> None:
    scm = _joint_scm()
    factual, _ = scm.sample_with_noise(seed=35)
    assert scm.counterfactual(Composite(()), observation=factual) == factual
    with pytest.raises(ValueError, match="reproduce"):
        scm.abduct({**factual, "Y": 999})
    with pytest.raises(ValueError, match="Complete observed state"):
        scm.abduct({"Y": 0})
    with pytest.raises(ValueError, match="cross-world"):
        scm.counterfactual(
            Delete("joint", JointPolicy(("B", "C"), {(0, 0): 1})), observation=factual
        )


def test_scm_rejects_incomplete_bindings_and_cycles() -> None:
    scm = _joint_scm()
    with pytest.raises(ValueError, match="every graph mechanism"):
        HypergraphSCM(scm.graph, {}, scm.exogenous)
    with pytest.raises(ValueError, match="exogenous"):
        HypergraphSCM(scm.graph, scm.mechanisms)
    cyclic = MechanismGraph(
        ("A", "B"),
        {
            "a": {"inputs": ("A",), "outputs": ("B",)},
            "b": {"inputs": ("B",), "outputs": ("A",)},
        },
    )
    with pytest.raises(ValueError, match="acyclic"):
        HypergraphSCM(cyclic, {})
