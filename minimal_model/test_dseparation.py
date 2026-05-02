"""Verifies d*-separation predictions for the reaction-network worked example.

These tests exercise Theorem T1 (THEOREM_T1.md):
- Five cases handled correctly by plain d-separation.
- One case (test_dsep_A_perp_D_given_C) that requires the deterministic-relations
  augmentation — plain d-sep on the bipartite blowup would miss this independence.
- One empirical CI check that confirms the d*-separation prediction holds in samples.

Runnable directly:  python -m minimal_model.test_dseparation
"""
from __future__ import annotations

import numpy as np

from .dseparation import d_separated, deterministic_closure
from .examples import reaction_network


# --------- Determination closure ---------

def test_closure_propagates_equality():
    """{C} closes to {C, D} because m1 declares them equal."""
    scm = reaction_network()
    assert deterministic_closure(scm, {"C"}) == {"C", "D"}
    assert deterministic_closure(scm, {"D"}) == {"C", "D"}


def test_closure_idempotent():
    scm = reaction_network()
    once = deterministic_closure(scm, {"C", "E"})
    twice = deterministic_closure(scm, once)
    assert once == twice == {"C", "D", "E"}


def test_closure_trivial_when_nothing_to_propagate():
    scm = reaction_network()
    assert deterministic_closure(scm, {"A", "B"}) == {"A", "B"}
    assert deterministic_closure(scm, set()) == set()


# --------- d*-separation: plain cases ---------

def test_dsep_A_perp_E():
    """A and E are root-roots of disjoint subtrees: d-separated unconditionally."""
    scm = reaction_network()
    assert d_separated(scm, {"A"}, {"E"})


def test_dsep_A_not_perp_F():
    """A → m1 → C → m2 → F is an open chain."""
    scm = reaction_network()
    assert not d_separated(scm, {"A"}, {"F"})


def test_dsep_A_perp_F_given_C():
    """Conditioning on the chain mediator C blocks A → m1 → C → m2 → F."""
    scm = reaction_network()
    assert d_separated(scm, {"A"}, {"F"}, {"C"})


def test_dsep_C_not_perp_D():
    """Joint outputs of m1: path C ← m1 → D is an open fork."""
    scm = reaction_network()
    assert not d_separated(scm, {"C"}, {"D"})


def test_dsep_collider_at_m2():
    """A and E meet at the collider m2: d-separated until F (descendant of m2) is conditioned."""
    scm = reaction_network()
    assert d_separated(scm, {"A"}, {"E"})  # collider m2 not opened
    assert not d_separated(scm, {"A"}, {"E"}, {"F"})  # F is descendant of m2 → opens it


# --------- d*-separation: requires the deterministic augmentation ---------

def test_dsep_A_perp_D_given_C_requires_augmentation():
    """The hard case. Conditioning on C functionally determines D (since C ≡ D under m1).

    The augmentation Z* = Z ∪ Det(Z) adds D to the effective conditioning set, so the
    query becomes A ⊥ D | {C, D} which is degenerate (D is in Z_eff, returns True).

    Plain d-separation on B(M) without augmentation would predict A ⊬ D | C
    (path A → m1 → D traverses no node of {C}). The augmentation is what makes T1 sound.
    """
    scm = reaction_network()
    # Sanity check: D is in the determination closure of {C}
    assert "D" in deterministic_closure(scm, {"C"})
    # With the augmentation, A ⊥ D | C
    assert d_separated(scm, {"A"}, {"D"}, {"C"})


# --------- Empirical confirmation ---------

def test_empirical_CI_matches_dsep_prediction():
    """A ⊥ E (predicted by d-sep) confirmed empirically: sample correlation near zero."""
    scm = reaction_network()
    rng = np.random.default_rng(2025)
    samples = [scm.sample(rng) for _ in range(3000)]
    a = np.array([s["A"] for s in samples])
    e = np.array([s["E"] for s in samples])
    corr = float(np.corrcoef(a, e)[0, 1])
    assert abs(corr) < 0.1, f"Expected |corr(A, E)| < 0.1 (d-separated), got {corr}"


def test_empirical_dependence_matches_dsep_prediction():
    """A ⊬ F (predicted by d-sep) confirmed empirically: substantial correlation."""
    scm = reaction_network()
    rng = np.random.default_rng(2025)
    samples = [scm.sample(rng) for _ in range(3000)]
    a = np.array([s["A"] for s in samples])
    f = np.array([s["F"] for s in samples])
    corr = float(np.corrcoef(a, f)[0, 1])
    assert abs(corr) > 0.3, f"Expected substantial corr(A, F), got {corr}"


# --------- Disjointness enforcement ---------

def test_disjointness_required():
    scm = reaction_network()
    try:
        d_separated(scm, {"A"}, {"A"})
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-disjoint X, Y")


# --------- Driver ---------

def _run_all():
    tests = [
        test_closure_propagates_equality,
        test_closure_idempotent,
        test_closure_trivial_when_nothing_to_propagate,
        test_dsep_A_perp_E,
        test_dsep_A_not_perp_F,
        test_dsep_A_perp_F_given_C,
        test_dsep_C_not_perp_D,
        test_dsep_collider_at_m2,
        test_dsep_A_perp_D_given_C_requires_augmentation,
        test_empirical_CI_matches_dsep_prediction,
        test_empirical_dependence_matches_dsep_prediction,
        test_disjointness_required,
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
