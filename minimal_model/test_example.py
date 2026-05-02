"""Verifies every numerical claim in MINIMAL_EXAMPLE.md.

Runnable directly:  python -m minimal_model.test_example
"""
from __future__ import annotations

import math

import numpy as np

from .examples import reaction_network, replacement_m1
from .scm import HypergraphSCM, Mechanism


# --------- Structural sanity ---------

def test_acyclic_and_topology():
    scm = reaction_network()
    assert scm.is_mechanism_acyclic(), "v1 requires C1 (mechanism-level acyclicity)"
    order = [m.name for m in scm.topological_order()]
    assert order == ["m1", "m2"], f"expected m1 before m2 in topo order, got {order}"


def test_bipartite_blowup_shape():
    scm = reaction_network()
    nodes, edges = scm.bipartite_blowup()
    assert nodes == {"A", "B", "C", "D", "E", "F", "m1", "m2"}
    expected_edges = {
        ("A", "m1"), ("B", "m1"), ("m1", "C"), ("m1", "D"),
        ("C", "m2"), ("E", "m2"), ("m2", "F"),
    }
    assert edges == expected_edges


# --------- Observational distribution ---------

def test_observational_joint_identity():
    """The structural identity C = D must hold on every sample."""
    scm = reaction_network()
    rng = np.random.default_rng(42)
    for _ in range(500):
        v = scm.sample(rng)
        assert math.isclose(v["C"], v["D"], abs_tol=1e-12), (
            f"Joint-output structural identity violated: C={v['C']}, D={v['D']}"
        )


def test_observational_correlation_one():
    """Empirical Corr(C, D) = 1 — the irreducibility witness."""
    scm = reaction_network()
    rng = np.random.default_rng(0)
    cs, ds = [], []
    for _ in range(2000):
        v = scm.sample(rng)
        cs.append(v["C"])
        ds.append(v["D"])
    corr = np.corrcoef(cs, ds)[0, 1]
    assert corr > 0.9999, f"Expected Corr(C,D) ≈ 1, got {corr}"


# --------- Three interventions ---------

def test_intervention_1_variable():
    """do(A = 4): standard variable intervention; joint identity preserved."""
    scm = reaction_network()
    int_scm = scm.do_node("A", 4.0)
    rng = np.random.default_rng(7)
    for _ in range(200):
        v = int_scm.sample(rng)
        assert v["A"] == 4.0
        assert math.isclose(v["C"], v["D"], abs_tol=1e-12)


def test_intervention_2_mechanism_deletion():
    """do(¬m1): C, D fall back to P_0 = δ_0; F driven by noise alone."""
    scm = reaction_network()
    int_scm = scm.do_delete_mechanism("m1")
    assert "m1" not in [m.name for m in int_scm.mechanisms]
    rng = np.random.default_rng(11)
    for _ in range(200):
        v = int_scm.sample(rng)
        assert v["C"] == 0.0, f"After do(¬m1), C should be 0 (P_0 fallback), got {v['C']}"
        assert v["D"] == 0.0
        # F = k2 * 0 * E + u2 = u2
        assert abs(v["F"]) < 0.5  # well within ~3σ of N(0, 0.01)


def test_intervention_3_mechanism_replacement():
    """do(m1 → m1' with k1' = 1.0): joint identity preserved, magnitude doubled."""
    scm = reaction_network()
    m1_prime = replacement_m1(k1_new=1.0)
    int_scm = scm.do_replace_mechanism("m1", m1_prime)
    rng = np.random.default_rng(13)
    base_rng = np.random.default_rng(13)
    base_vals = [reaction_network().sample(base_rng) for _ in range(100)]
    int_vals = [int_scm.sample(rng) for _ in range(100)]
    # Same noise streams (same seed) ⇒ int C ≈ 2 × base C in expectation
    base_mean = np.mean([v["C"] for v in base_vals])
    int_mean = np.mean([v["C"] for v in int_vals])
    ratio = int_mean / base_mean
    assert 1.8 < ratio < 2.2, f"Expected ~2x C under k1' = 1.0 vs k1 = 0.5, got ratio {ratio}"
    for v in int_vals:
        assert math.isclose(v["C"], v["D"], abs_tol=1e-12)


def test_replacement_rejects_incidence_mismatch():
    """Mechanism replacement enforces ρ(m') = ρ(m)."""
    scm = reaction_network()
    bogus = Mechanism(
        name="m1",
        inputs=("A",),  # missing B
        outputs=("C", "D"),
        f=lambda i, n: {"C": 0.0, "D": 0.0},
        sample_noise=lambda rng: 0.0,
    )
    try:
        scm.do_replace_mechanism("m1", bogus)
    except ValueError as e:
        assert "incidence" in str(e)
    else:
        raise AssertionError("Expected ValueError for incidence mismatch")


# --------- Counterfactual ---------

def test_counterfactual_variable_intervention():
    """Reproduces MINIMAL_EXAMPLE.md §5 to 6 decimal places."""
    scm = reaction_network()
    observed = {
        "A": 2.0, "B": 3.0, "E": 4.0,
        "C": 3.05, "D": 3.05, "F": 3.71,
    }
    cf = scm.counterfactual(
        observed=observed,
        intervention=lambda s: s.do_node("A", 4.0),
        query=["A", "B", "C", "D", "E", "F"],
    )
    # By hand: u1 = 0.05, u2 = 0.05; under do(A=4), C = D = 0.5*4*3 + 0.05 = 6.05; F = 0.3*6.05*4 + 0.05 = 7.31
    assert math.isclose(cf["A"], 4.0, abs_tol=1e-9)
    assert math.isclose(cf["B"], 3.0, abs_tol=1e-9)
    assert math.isclose(cf["C"], 6.05, abs_tol=1e-9), f"Expected C^cf = 6.05, got {cf['C']}"
    assert math.isclose(cf["D"], 6.05, abs_tol=1e-9), f"Expected D^cf = 6.05, got {cf['D']}"
    assert math.isclose(cf["F"], 7.31, abs_tol=1e-9), f"Expected F^cf = 7.31, got {cf['F']}"
    assert cf["C"] == cf["D"], "Joint identity must survive counterfactual"


def test_counterfactual_mechanism_deletion():
    """Reproduces MINIMAL_EXAMPLE.md §6: do(¬m1) counterfactual, F^cf = u2 = 0.05."""
    scm = reaction_network()
    observed = {
        "A": 2.0, "B": 3.0, "E": 4.0,
        "C": 3.05, "D": 3.05, "F": 3.71,
    }
    cf = scm.counterfactual(
        observed=observed,
        intervention=lambda s: s.do_delete_mechanism("m1"),
        query=["C", "D", "F"],
    )
    assert cf["C"] == 0.0
    assert cf["D"] == 0.0
    assert math.isclose(cf["F"], 0.05, abs_tol=1e-9), f"Expected F^cf = 0.05, got {cf['F']}"


def test_counterfactual_off_manifold_rejected():
    """Off-manifold observation (C ≠ D) must fail abduction loudly."""
    scm = reaction_network()
    observed = {
        "A": 2.0, "B": 3.0, "E": 4.0,
        "C": 3.05, "D": 4.00,  # inconsistent with joint mechanism
        "F": 3.71,
    }
    try:
        scm.counterfactual(
            observed=observed,
            intervention=lambda s: s.do_node("A", 4.0),
            query=["F"],
        )
    except ValueError as e:
        assert "Off-manifold" in str(e) or "C = D" in str(e)
    else:
        raise AssertionError("Expected ValueError for off-manifold observation")


# --------- Driver ---------

def _run_all():
    tests = [
        test_acyclic_and_topology,
        test_bipartite_blowup_shape,
        test_observational_joint_identity,
        test_observational_correlation_one,
        test_intervention_1_variable,
        test_intervention_2_mechanism_deletion,
        test_intervention_3_mechanism_replacement,
        test_replacement_rejects_incidence_mismatch,
        test_counterfactual_variable_intervention,
        test_counterfactual_mechanism_deletion,
        test_counterfactual_off_manifold_rejected,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{len(failures)} of {len(tests)} tests FAILED")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    _run_all()
