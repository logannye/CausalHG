"""Canonical worked example: 2-mechanism reaction network.

See MINIMAL_EXAMPLE.md for full derivations. This module builds the SCM whose
behavior is verified in test_example.py.
"""
from __future__ import annotations

import numpy as np

from .scm import HypergraphSCM, Mechanism

K1_DEFAULT = 0.5
K2 = 0.3
NOISE_SIGMA = 0.1


def _make_m1(k1: float = K1_DEFAULT, sigma: float = NOISE_SIGMA, name: str = "m1") -> Mechanism:
    def f(inputs, noise):
        produced = k1 * inputs["A"] * inputs["B"] + noise
        # Joint output: C and D share the same value, driven by a single noise term.
        # This is the structural identity that no Pearl SCM with independent
        # equations and nontrivial noise can express.
        return {"C": produced, "D": produced}

    def sample_noise(rng):
        return float(rng.normal(0.0, sigma))

    def abduct(input_vals, output_vals):
        # Over-determined case: |out| = 2, dim(noise) = 1. On-manifold (C == D),
        # abduction is unique. Off-manifold, raise to surface the inconsistency.
        if not np.isclose(output_vals["C"], output_vals["D"]):
            raise ValueError(
                f"Off-manifold observation: C={output_vals['C']} != D={output_vals['D']}; "
                "joint mechanism enforces C = D structurally."
            )
        return output_vals["C"] - k1 * input_vals["A"] * input_vals["B"]

    return Mechanism(
        name=name,
        inputs=("A", "B"),
        outputs=("C", "D"),
        f=f,
        sample_noise=sample_noise,
        abduct=abduct,
        output_equalities=(("C", "D"),),
    )


def _make_m2(k2: float = K2, sigma: float = NOISE_SIGMA) -> Mechanism:
    def f(inputs, noise):
        return {"F": k2 * inputs["C"] * inputs["E"] + noise}

    def sample_noise(rng):
        return float(rng.normal(0.0, sigma))

    def abduct(input_vals, output_vals):
        return output_vals["F"] - k2 * input_vals["C"] * input_vals["E"]

    return Mechanism(
        name="m2",
        inputs=("C", "E"),
        outputs=("F",),
        f=f,
        sample_noise=sample_noise,
        abduct=abduct,
    )


def reaction_network() -> HypergraphSCM:
    """Two coupled stoichiometric reactions: A+B → C+D, then C+E → F.

    See MINIMAL_EXAMPLE.md §1-2.
    """
    m1 = _make_m1()
    m2 = _make_m2()

    exogenous = {
        "A": lambda rng: float(rng.uniform(1.0, 5.0)),
        "B": lambda rng: float(rng.uniform(1.0, 5.0)),
        "E": lambda rng: float(rng.uniform(1.0, 5.0)),
        # P_0 fallbacks: concentration zero when the producing mechanism is deleted.
        "C": lambda rng: 0.0,
        "D": lambda rng: 0.0,
        "F": lambda rng: 0.0,
    }

    return HypergraphSCM(
        variables=("A", "B", "C", "D", "E", "F"),
        mechanisms=(m1, m2),
        exogenous=exogenous,
    )


def replacement_m1(k1_new: float = 1.0) -> Mechanism:
    """A drop-in replacement for m1 with a different rate constant. Same incidence."""
    return _make_m1(k1=k1_new, name="m1")


def reaction_network_with_hidden_W() -> HypergraphSCM:
    """HADMG with a hidden variable W as input to m_2.

    Variables: A, B, C, D, E, F (observed), W (hidden).
    Mechanisms:
    - m_W: ∅ → W (latent mechanism producing the hidden variable)
    - m_1: A, B → C, D (observed; boundary entirely observed → T6 applies)
    - m_2: C, E, W → F (observed; boundary contains W → T6 fails, T7 reduces to Pearl)

    Used to verify the hidden-variable identifiability behavior of THEOREM_H1_PLUS.md.
    Caller is responsible for declaring W as hidden via observed_variables= argument
    when invoking is_mechanism_deletion_identifiable.
    """
    def m_W_f(inputs, noise):
        return {"W": float(noise)}

    m_W = Mechanism(
        name="m_W",
        inputs=(),
        outputs=("W",),
        f=m_W_f,
        sample_noise=lambda rng: float(rng.normal(2.0, 0.5)),
        abduct=lambda inp, out: out["W"],
        latent=True,
    )

    m_1 = _make_m1()

    def m_2_with_W_f(inputs, noise):
        return {"F": K2 * inputs["C"] * inputs["E"] * inputs["W"] + noise}

    m_2 = Mechanism(
        name="m_2",
        inputs=("C", "E", "W"),
        outputs=("F",),
        f=m_2_with_W_f,
        sample_noise=lambda rng: float(rng.normal(0.0, NOISE_SIGMA)),
        abduct=lambda inp, out: out["F"] - K2 * inp["C"] * inp["E"] * inp["W"],
    )

    exogenous = {
        "A": lambda rng: float(rng.uniform(1.0, 5.0)),
        "B": lambda rng: float(rng.uniform(1.0, 5.0)),
        "E": lambda rng: float(rng.uniform(1.0, 5.0)),
        # Fallbacks (for post-deletion of producing mechanisms):
        "C": lambda rng: 0.0,
        "D": lambda rng: 0.0,
        "F": lambda rng: 0.0,
        "W": lambda rng: 0.0,
    }

    return HypergraphSCM(
        variables=("A", "B", "C", "D", "E", "F", "W"),
        mechanisms=(m_W, m_1, m_2),
        exogenous=exogenous,
    )


def reaction_network_with_latent_BE(
    rho: float = 0.6, sigma_BE: float = 0.5
) -> HypergraphSCM:
    """HADMG variant: a latent mechanism jointly produces B and E.

    Structurally identical to reaction_network() except A is the only exogenous
    input variable; B and E are produced by m_lat with correlated noise. Used to
    verify T4's identifiability claim under hidden-mechanism confounding
    (THEOREM_T4_T5.md §4).

    Parameters
    ----------
    rho : correlation coefficient between B and E induced by m_lat.
    sigma_BE : standard deviation of each output around its mean of 3.0.
    """
    # m_lat: ∅ → B, E, with B and E drawn from a bivariate Gaussian with
    # correlation rho. The "noise" is a 2-vector. f deterministically reads it out.
    cov = np.array([[sigma_BE**2, rho * sigma_BE**2], [rho * sigma_BE**2, sigma_BE**2]])
    mean_BE = np.array([3.0, 3.0])

    def m_lat_f(inputs, noise):
        return {"B": float(noise[0]), "E": float(noise[1])}

    def m_lat_sample_noise(rng):
        return rng.multivariate_normal(mean_BE, cov)

    def m_lat_abduct(input_vals, output_vals):
        return np.array([output_vals["B"], output_vals["E"]])

    m_lat = Mechanism(
        name="m_lat",
        inputs=(),
        outputs=("B", "E"),
        f=m_lat_f,
        sample_noise=m_lat_sample_noise,
        abduct=m_lat_abduct,
        latent=True,
    )

    m1 = _make_m1()
    m2 = _make_m2()

    exogenous = {
        "A": lambda rng: float(rng.uniform(1.0, 5.0)),
        # B, E now produced by m_lat — no exogenous samplers needed for the
        # observational distribution. Fallbacks below are used only post-deletion.
        "B": lambda rng: 0.0,
        "E": lambda rng: 0.0,
        "C": lambda rng: 0.0,
        "D": lambda rng: 0.0,
        "F": lambda rng: 0.0,
    }

    return HypergraphSCM(
        variables=("A", "B", "C", "D", "E", "F"),
        mechanisms=(m_lat, m1, m2),
        exogenous=exogenous,
    )
