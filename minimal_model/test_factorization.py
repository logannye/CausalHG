"""Verifies Theorems T2 (mechanism-deletion truncation) and T3 (mechanism replacement).

Tests two layers:
1. Symbolic factor structure of P(V) and P(V | do(¬m)) matches T2's prediction.
2. Numerical post-intervention moments under direct simulation match the
   moments predicted by T2 applied to the worked example's parametric form.

Runnable directly:  python -m minimal_model.test_factorization
"""
from __future__ import annotations

import math

import numpy as np

from .examples import reaction_network, replacement_m1
from .scm import Factor, HypergraphSCM, Mechanism


# --------- Validation (C1, C3, C4) ---------

def test_validate_passes_on_canonical_example():
    scm = reaction_network()
    scm.validate()


def test_validate_rejects_C4_violation():
    """Two mechanisms producing the same variable violate C4."""
    scm = reaction_network()
    bogus = Mechanism(
        name="m_bogus",
        inputs=("E",),
        outputs=("C",),  # C is already produced by m1
        f=lambda i, n: {"C": 0.0},
        sample_noise=lambda rng: 0.0,
    )
    polluted = HypergraphSCM(
        variables=scm.variables,
        mechanisms=scm.mechanisms + (bogus,),
        exogenous=scm.exogenous,
    )
    try:
        polluted.validate()
    except ValueError as e:
        assert "C4" in str(e) or "multiple mechanisms" in str(e)
    else:
        raise AssertionError("Expected ValueError for C4 violation")


def test_validate_rejects_C3_violation():
    """A mechanism with overlapping inputs and outputs violates C3."""
    bogus = Mechanism(
        name="m_bogus",
        inputs=("A",),
        outputs=("A",),  # in ∩ out = {A}
        f=lambda i, n: {"A": 0.0},
        sample_noise=lambda rng: 0.0,
    )
    scm = HypergraphSCM(
        variables=("A",),
        mechanisms=(bogus,),
        exogenous={"A": lambda rng: 0.0},
    )
    try:
        scm.validate()
    except ValueError as e:
        assert "overlapping" in str(e) or "C3" in str(e)
    else:
        raise AssertionError("Expected ValueError for C3 violation")


# --------- Symbolic factorization (Lemma 1.1) ---------

def test_factorize_gives_expected_factors():
    """P(V) = P(A) P(B) P(E) · P(C, D | A, B) · P(F | C, E)."""
    scm = reaction_network()
    factors = scm.factorize()
    # Three exogenous factors + two mechanism factors
    assert len(factors) == 5
    sources = [f.source for f in factors]
    assert sources.count("exogenous") == 3
    assert "m1" in sources
    assert "m2" in sources

    m1_factor = next(f for f in factors if f.source == "m1")
    assert set(m1_factor.variables) == {"C", "D"}
    assert set(m1_factor.conditional_on) == {"A", "B"}

    m2_factor = next(f for f in factors if f.source == "m2")
    assert m2_factor.variables == ("F",)
    assert set(m2_factor.conditional_on) == {"C", "E"}


def test_factorize_only_exogenous_for_unproduced():
    """C, D, F are not exogenous in the original SCM (they have producers)."""
    scm = reaction_network()
    factors = scm.factorize()
    exo_vars = {f.variables[0] for f in factors if f.source == "exogenous"}
    assert exo_vars == {"A", "B", "E"}
    assert "C" not in exo_vars and "D" not in exo_vars and "F" not in exo_vars


# --------- T2: truncated factorization ---------

def test_truncated_factorization_drops_deleted_mechanism():
    """do(¬m1): mechanism m1 factor replaced by P_0 factors for C, D."""
    scm = reaction_network()
    factors = scm.truncated_factorization("m1")

    sources = [f.source for f in factors]
    assert "m1" not in sources, "Deleted mechanism factor must not survive"
    assert sources.count("P_0") == 2, "Both orphaned outputs need P_0 factors"
    assert "m2" in sources, "Other mechanism factor must survive"
    assert sources.count("exogenous") == 3, "Exogenous factors unchanged"


def test_truncated_factorization_P0_factors_are_for_orphaned_outputs():
    """The P_0 factors are for the orphaned outputs of m1, namely C and D."""
    scm = reaction_network()
    factors = scm.truncated_factorization("m1")
    p0_vars = {f.variables[0] for f in factors if f.source == "P_0"}
    assert p0_vars == {"C", "D"}


def test_truncated_factorization_rejects_C4_violation():
    """If a deleted mechanism's outputs have other producers, T2 should refuse."""
    scm = reaction_network()
    # Hypothetical: a second mechanism that also produces C
    bogus = Mechanism(
        name="m_bogus",
        inputs=("E",),
        outputs=("C",),
        f=lambda i, n: {"C": 0.0},
        sample_noise=lambda rng: 0.0,
    )
    polluted = HypergraphSCM(
        variables=scm.variables,
        mechanisms=scm.mechanisms + (bogus,),
        exogenous=scm.exogenous,
    )
    try:
        polluted.truncated_factorization("m1")
    except ValueError as e:
        assert "C4" in str(e) or "single producer" in str(e) or "also produced" in str(e)
    else:
        raise AssertionError("Expected ValueError for C4 violation in T2")


# --------- Numerical T2 verification ---------

def test_T2_predicts_post_intervention_moments():
    """T2 applied to the worked example: under do(¬m1), C, D = 0 a.s. and F ~ N(0, σ²).

    Direct simulation of M^{¬m1} should produce E[F] ≈ 0, Var[F] ≈ σ_2² = 0.01.
    This matches T2's analytic prediction (THEOREM_T2_T3.md §6).
    """
    scm = reaction_network()
    int_scm = scm.do_delete_mechanism("m1")
    rng = np.random.default_rng(2025)
    samples = [int_scm.sample(rng) for _ in range(5000)]

    cs = np.array([s["C"] for s in samples])
    ds = np.array([s["D"] for s in samples])
    fs = np.array([s["F"] for s in samples])

    # T2 predicts C ≡ 0, D ≡ 0 (P_0 = δ_0)
    assert np.all(cs == 0.0), "T2: C must be 0 a.s. under do(¬m1) with P_0 = δ_0"
    assert np.all(ds == 0.0)

    # T2 predicts F ~ N(0, σ_2²) since F = k_2 · 0 · E + u_2 = u_2
    assert abs(np.mean(fs)) < 0.01, f"E[F] should be ~0, got {np.mean(fs)}"
    assert 0.005 < np.var(fs) < 0.015, f"Var[F] should be ~0.01, got {np.var(fs)}"


def test_T3_predicts_post_replacement_moments():
    """T3 applied to do(m1 → m1') with k1' = 1.0: same structure, doubled rate.

    Under T3, P(C, D | A, B) factor is replaced by the new mechanism's factor.
    Prediction: E[C] = E[D] = k1' · E[A] · E[B] = 1.0 · 3 · 3 = 9.0.
    Original: 0.5 · 3 · 3 = 4.5. So replacement-vs-original ratio = 2.
    """
    scm = reaction_network()
    int_scm = scm.do_replace_mechanism("m1", replacement_m1(k1_new=1.0))
    rng = np.random.default_rng(2025)
    samples = [int_scm.sample(rng) for _ in range(5000)]

    cs = np.array([s["C"] for s in samples])
    e_c = float(np.mean(cs))
    # T3 prediction: E[C] = k1' * E[A] * E[B] = 1.0 * 3 * 3 = 9.0
    assert 8.5 < e_c < 9.5, f"T3: E[C] should be ~9.0 under k1' = 1.0, got {e_c}"


def test_T3_deletion_as_special_case():
    """Corollary T3.1: do(¬m) = do(m → m_trivial) where m_trivial returns P_0 outputs.

    Verified by constructing a trivial replacement and showing it gives the same
    post-intervention distribution as deletion.
    """
    scm = reaction_network()

    def trivial_f(inputs, noise):
        return {"C": 0.0, "D": 0.0}

    trivial_m1 = Mechanism(
        name="m1",
        inputs=("A", "B"),
        outputs=("C", "D"),
        f=trivial_f,
        sample_noise=lambda rng: 0.0,
        output_equalities=(("C", "D"),),
    )

    rng_d = np.random.default_rng(0)
    rng_r = np.random.default_rng(0)
    deletion_samples = [scm.do_delete_mechanism("m1").sample(rng_d) for _ in range(500)]
    replacement_samples = [scm.do_replace_mechanism("m1", trivial_m1).sample(rng_r) for _ in range(500)]

    for ds, rs in zip(deletion_samples, replacement_samples):
        assert ds["C"] == rs["C"] == 0.0
        assert ds["D"] == rs["D"] == 0.0
    # F means should agree (both driven by m_2 with C=0)
    assert math.isclose(
        np.mean([s["F"] for s in deletion_samples]),
        np.mean([s["F"] for s in replacement_samples]),
        abs_tol=0.02,
    )


# --------- Driver ---------

def _run_all():
    tests = [
        test_validate_passes_on_canonical_example,
        test_validate_rejects_C4_violation,
        test_validate_rejects_C3_violation,
        test_factorize_gives_expected_factors,
        test_factorize_only_exogenous_for_unproduced,
        test_truncated_factorization_drops_deleted_mechanism,
        test_truncated_factorization_P0_factors_are_for_orphaned_outputs,
        test_truncated_factorization_rejects_C4_violation,
        test_T2_predicts_post_intervention_moments,
        test_T3_predicts_post_replacement_moments,
        test_T3_deletion_as_special_case,
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
