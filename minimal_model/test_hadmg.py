"""Verifies Theorem T4 (mechanism-deletion identifiability under latent mechanisms).

The HADMG used here augments the reaction network with a latent mechanism m_lat
that jointly produces B and E from a bivariate-Gaussian noise — inducing
observational correlation between two inputs that are independent in the
causally-sufficient version. T4 predicts that mechanism-deletion remains
identifiable from P(V) alone, with the latent confounding preserved in the
surviving P(B, E) factor.

Runnable directly:  python -m minimal_model.test_hadmg
"""
from __future__ import annotations

import math

import numpy as np

from .examples import reaction_network, reaction_network_with_latent_BE
from .scm import HypergraphSCM


# --------- HADMG accessors ---------

def test_is_hadmg_recognizes_latent():
    plain = reaction_network()
    hadmg = reaction_network_with_latent_BE()
    assert not plain.is_hadmg()
    assert hadmg.is_hadmg()


def test_latent_and_observed_partition():
    hadmg = reaction_network_with_latent_BE()
    latent_names = {m.name for m in hadmg.latent_mechanisms()}
    observed_names = {m.name for m in hadmg.observed_mechanisms()}
    assert latent_names == {"m_lat"}
    assert observed_names == {"m1", "m2"}
    assert latent_names.isdisjoint(observed_names)


# --------- Latent confounding manifests in observational distribution ---------

def test_latent_induces_correlation_between_B_and_E():
    """The whole point of m_lat: B and E are observationally correlated."""
    hadmg = reaction_network_with_latent_BE(rho=0.6)
    rng = np.random.default_rng(0)
    samples = [hadmg.sample(rng) for _ in range(3000)]
    bs = np.array([s["B"] for s in samples])
    es = np.array([s["E"] for s in samples])
    corr = float(np.corrcoef(bs, es)[0, 1])
    # Empirical corr should be near rho = 0.6
    assert 0.5 < corr < 0.7, f"Expected corr(B, E) ≈ 0.6 from m_lat, got {corr}"


def test_plain_network_has_uncorrelated_BE():
    """Sanity: in the non-HADMG version, B and E are independent (uncorrelated)."""
    scm = reaction_network()
    rng = np.random.default_rng(0)
    samples = [scm.sample(rng) for _ in range(3000)]
    bs = np.array([s["B"] for s in samples])
    es = np.array([s["E"] for s in samples])
    corr = float(np.corrcoef(bs, es)[0, 1])
    assert abs(corr) < 0.05, f"Expected ~0 correlation in non-HADMG, got {corr}"


# --------- T4 identifiability verdict ---------

def test_T4_identifiable_for_observed_mechanism():
    """do(¬m1) is identifiable: m1's boundary {A, B, C, D} ⊆ V_obs."""
    hadmg = reaction_network_with_latent_BE()
    verdict, justification = hadmg.is_mechanism_deletion_identifiable("m1")
    assert verdict, f"T4 should accept do(¬m1); rejected with: {justification}"
    assert "T4" in justification


def test_T4_identifiable_for_latent_mechanism():
    """do(¬m_lat) is identifiable too — even though f_{m_lat} is unknown,
    the mechanism factor P(B, E) is observable as the joint marginal.

    This is the surprising positive direction: intervening on a latent mechanism
    has a closed-form identifying expression, with no Pearl analogue.
    """
    hadmg = reaction_network_with_latent_BE()
    verdict, justification = hadmg.is_mechanism_deletion_identifiable("m_lat")
    assert verdict, f"T4 should accept do(¬m_lat); rejected with: {justification}"


def test_T4_rejects_when_boundary_unobserved():
    """If we declare some boundary variables unobserved, T4 refuses identifiability."""
    hadmg = reaction_network_with_latent_BE()
    # Pretend B is unobserved
    obs = set(hadmg.variables) - {"B"}
    verdict, justification = hadmg.is_mechanism_deletion_identifiable("m1", observed_variables=obs)
    assert not verdict
    assert "B" in justification or "boundary" in justification


def test_T4_unknown_mechanism():
    hadmg = reaction_network_with_latent_BE()
    verdict, justification = hadmg.is_mechanism_deletion_identifiable("m_does_not_exist")
    assert not verdict


# --------- T4 formula verified against direct simulation ---------

def test_T4_post_intervention_C_D_orphaned_under_latent_confounding():
    """do(¬m1) on the HADMG: C, D should fall to P_0 = δ_0 regardless of latent.

    This is structural — deletion orphans the outputs. The latent confounding
    of B, E does not affect this.
    """
    hadmg = reaction_network_with_latent_BE()
    int_scm = hadmg.do_delete_mechanism("m1")
    rng = np.random.default_rng(0)
    samples = [int_scm.sample(rng) for _ in range(2000)]
    cs = np.array([s["C"] for s in samples])
    ds = np.array([s["D"] for s in samples])
    assert np.all(cs == 0.0), "C must be 0 under do(¬m1) with P_0 = δ_0"
    assert np.all(ds == 0.0)


def test_T4_post_intervention_BE_correlation_preserved():
    """Crucial T4 prediction: after do(¬m1), the latent B-E correlation survives.

    This is the §4 example's key payoff. The formula
        P^{¬m1}(V) = P(V) / P(C, D | A, B) · δ_0(C) δ_0(D)
    leaves the P(B, E) factor untouched, so the latent confounding propagates
    to the post-intervention distribution.
    """
    hadmg = reaction_network_with_latent_BE(rho=0.6)
    int_scm = hadmg.do_delete_mechanism("m1")
    rng = np.random.default_rng(0)
    samples = [int_scm.sample(rng) for _ in range(3000)]
    bs = np.array([s["B"] for s in samples])
    es = np.array([s["E"] for s in samples])
    corr = float(np.corrcoef(bs, es)[0, 1])
    assert 0.5 < corr < 0.7, f"Latent B-E correlation must survive intervention; got {corr}"


def test_T4_post_intervention_F_marginal_under_do_neg_m1():
    """Under do(¬m1) in the HADMG: F = u_2 ~ N(0, σ_2²), since C = 0.

    This is the same prediction as T2 in the causally-sufficient case — T4 says
    the latent confounding does not change F's marginal under this query because
    F's mechanism factor is m_2 evaluated at C = 0.
    """
    hadmg = reaction_network_with_latent_BE()
    int_scm = hadmg.do_delete_mechanism("m1")
    rng = np.random.default_rng(0)
    samples = [int_scm.sample(rng) for _ in range(5000)]
    fs = np.array([s["F"] for s in samples])
    assert abs(np.mean(fs)) < 0.01, f"E[F] should be ~0, got {np.mean(fs)}"
    assert 0.005 < np.var(fs) < 0.015, f"Var[F] should be ~0.01, got {np.var(fs)}"


def test_T4_intervening_on_latent_orphans_outputs():
    """do(¬m_lat): B, E fall to their P_0 fallbacks (set to 0 in the example)."""
    hadmg = reaction_network_with_latent_BE()
    int_scm = hadmg.do_delete_mechanism("m_lat")
    rng = np.random.default_rng(0)
    samples = [int_scm.sample(rng) for _ in range(500)]
    for s in samples:
        assert s["B"] == 0.0, "After do(¬m_lat), B must take its P_0 fallback"
        assert s["E"] == 0.0


def test_T4_intervening_on_latent_breaks_BE_correlation():
    """The whole point of intervening on the latent: the B-E correlation is destroyed.

    This is the most semantically rich query in the project: a closed-form
    identifying expression for "what would the system look like if we ablated
    the unobserved confounder?"
    """
    hadmg = reaction_network_with_latent_BE(rho=0.8)
    pre_int_rng = np.random.default_rng(0)
    pre_samples = [hadmg.sample(pre_int_rng) for _ in range(2000)]
    pre_corr = float(np.corrcoef(
        [s["B"] for s in pre_samples], [s["E"] for s in pre_samples]
    )[0, 1])
    assert 0.7 < pre_corr < 0.9, f"Setup: high pre-intervention B-E correlation, got {pre_corr}"

    int_scm = hadmg.do_delete_mechanism("m_lat")
    rng = np.random.default_rng(0)
    post_samples = [int_scm.sample(rng) for _ in range(2000)]
    # B and E both take constant value 0 from P_0; correlation undefined (constant series).
    bs = np.array([s["B"] for s in post_samples])
    es = np.array([s["E"] for s in post_samples])
    assert np.all(bs == 0.0)
    assert np.all(es == 0.0)


# --------- Lemma 1.1 (HADMG version): chain-rule factorization ---------

def test_chain_rule_factorization_includes_latent():
    """Lemma 1.1 in the HADMG: factorize() returns one factor per mechanism, latent included."""
    hadmg = reaction_network_with_latent_BE()
    factors = hadmg.factorize()
    sources = [f.source for f in factors]
    assert "m_lat" in sources
    assert "m1" in sources
    assert "m2" in sources
    # m_lat factor: P(B, E | ∅) — just the joint marginal.
    m_lat_factor = next(f for f in factors if f.source == "m_lat")
    assert set(m_lat_factor.variables) == {"B", "E"}
    assert m_lat_factor.conditional_on == ()


def test_truncated_factorization_preserves_latent_factor_for_observed_target():
    """do(¬m1): m_lat's factor P(B, E) survives the truncation."""
    hadmg = reaction_network_with_latent_BE()
    factors = hadmg.truncated_factorization("m1")
    sources = [f.source for f in factors]
    assert "m_lat" in sources, "Latent mechanism factor must survive deletion of m1"
    assert "m1" not in sources
    assert sources.count("P_0") == 2  # C, D orphaned


def test_truncated_factorization_can_target_latent():
    """do(¬m_lat): m_lat's factor is replaced by P_0 factors for B, E."""
    hadmg = reaction_network_with_latent_BE()
    factors = hadmg.truncated_factorization("m_lat")
    sources = [f.source for f in factors]
    assert "m_lat" not in sources
    p0_vars = {f.variables[0] for f in factors if f.source == "P_0"}
    assert p0_vars == {"B", "E"}, "B and E orphaned by deletion of m_lat"


# --------- Driver ---------

def _run_all():
    tests = [
        test_is_hadmg_recognizes_latent,
        test_latent_and_observed_partition,
        test_latent_induces_correlation_between_B_and_E,
        test_plain_network_has_uncorrelated_BE,
        test_T4_identifiable_for_observed_mechanism,
        test_T4_identifiable_for_latent_mechanism,
        test_T4_rejects_when_boundary_unobserved,
        test_T4_unknown_mechanism,
        test_T4_post_intervention_C_D_orphaned_under_latent_confounding,
        test_T4_post_intervention_BE_correlation_preserved,
        test_T4_post_intervention_F_marginal_under_do_neg_m1,
        test_T4_intervening_on_latent_orphans_outputs,
        test_T4_intervening_on_latent_breaks_BE_correlation,
        test_chain_rule_factorization_includes_latent,
        test_truncated_factorization_preserves_latent_factor_for_observed_target,
        test_truncated_factorization_can_target_latent,
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
