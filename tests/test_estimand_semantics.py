"""Differential tests: compiled estimands evaluated against interventional ground truth.

The repository's two halves — the symbolic identification compiler in
``src/causal_hypergraphs`` and the numerical reference semantics in ``minimal_model``
— have never met. Every existing identification test compares a *rendered string*
against a hand-written expected string, so an estimand that is wrong in the same way
in both places passes. These tests close that gap: they compile an estimand, evaluate
it as a function, and compare it pointwise against an interventional law computed
independently of the compiler.

Models here are finite and discrete so that both sides are computed in exact
arithmetic; no Monte Carlo tolerance is involved. Ground truth is built in the test
from an explicit factorization, never by calling the code under test.
"""
from __future__ import annotations

import itertools

import pytest

from causal_hypergraphs import (
    DeleteMechanism,
    Identified,
    MechanismGraph,
    ReplaceMechanism,
    identify,
)
from causal_hypergraphs.semantics import DiscreteModel, evaluate

BINARY = (0, 1)

# --- A strictly positive reaction model -------------------------------------------
#
#   m1: {A, B} -> {C, D}      m2: {C, E} -> {F}
#
# Exogenous marginals.
P_A = {0: 0.7, 1: 0.3}
P_B = {0: 0.4, 1: 0.6}
P_E = {0: 0.6, 1: 0.4}


def p_cd_given_ab(c: int, d: int, a: int, b: int) -> float:
    """A strictly positive, genuinely *joint* kernel for m1 (C and D are correlated)."""
    weights = {
        (0, 0): 1.0 + a,
        (0, 1): 0.5,
        (1, 0): 0.5 + b,
        (1, 1): 1.0 + a * b,
    }
    total = sum(weights.values())
    return weights[(c, d)] / total


def p_f_given_ce(f: int, c: int, e: int) -> float:
    """A strictly positive kernel for m2."""
    p_one = 0.2 + 0.3 * c + 0.25 * e
    return p_one if f == 1 else 1.0 - p_one


# Fallback (post-deletion) laws for m1's outputs.
P0_C = {0: 0.75, 1: 0.25}
P0_D = {0: 0.5, 1: 0.5}

VARIABLES = ("A", "B", "C", "D", "E", "F")
DOMAINS = {v: BINARY for v in VARIABLES}


def _assignments() -> list[dict[str, int]]:
    return [dict(zip(VARIABLES, combo)) for combo in itertools.product(*(DOMAINS[v] for v in VARIABLES))]


def observational_joint() -> dict[tuple[int, ...], float]:
    """P(V) from the mechanism-level chain rule (Lemma 1.1), built here, not by the library."""
    joint = {}
    for x in _assignments():
        joint[tuple(x[v] for v in VARIABLES)] = (
            P_A[x["A"]]
            * P_B[x["B"]]
            * P_E[x["E"]]
            * p_cd_given_ab(x["C"], x["D"], x["A"], x["B"])
            * p_f_given_ce(x["F"], x["C"], x["E"])
        )
    return joint


def interventional_joint_delete_m1() -> dict[tuple[int, ...], float]:
    """P(V | do(not m1)): m1's factor is replaced by the declared fallback laws."""
    joint = {}
    for x in _assignments():
        joint[tuple(x[v] for v in VARIABLES)] = (
            P_A[x["A"]]
            * P_B[x["B"]]
            * P_E[x["E"]]
            * P0_C[x["C"]]
            * P0_D[x["D"]]
            * p_f_given_ce(x["F"], x["C"], x["E"])
        )
    return joint


def reaction_mechanism_graph() -> MechanismGraph:
    return MechanismGraph(
        variables=set(VARIABLES),
        mechanisms={
            "m1": {"inputs": {"A", "B"}, "outputs": {"C", "D"}},
            "m2": {"inputs": {"C", "E"}, "outputs": {"F"}},
        },
        observed_variables=set(VARIABLES),
    )


def test_observational_and_interventional_laws_are_normalized() -> None:
    """Guard the fixtures themselves: a test built on an unnormalized law proves nothing."""
    assert sum(observational_joint().values()) == pytest.approx(1.0, abs=1e-12)
    assert sum(interventional_joint_delete_m1().values()) == pytest.approx(1.0, abs=1e-12)


def test_t2_deletion_estimand_reproduces_interventional_law_on_positive_model() -> None:
    """The compiled T2 estimand, evaluated, must equal P(V | do(not m1)) pointwise.

    This is the check the repository has never performed. It is a genuine identification
    test: the estimand may reference only observational quantities plus the declared
    fallback laws, and must recover the interventional law from them alone.
    """
    graph = reaction_mechanism_graph()
    result = identify(graph, DeleteMechanism("m1"))
    assert isinstance(result, Identified)
    assert result.theorem == "T2"

    model = DiscreteModel(
        domains=DOMAINS,
        joint=observational_joint(),
        fallbacks={"C": P0_C, "D": P0_D},
    )
    truth = interventional_joint_delete_m1()

    for x in _assignments():
        key = tuple(x[v] for v in VARIABLES)
        assert evaluate(result.expression, model, x) == pytest.approx(truth[key], abs=1e-12)


# --- A deterministic (stoichiometrically coupled) mechanism ------------------------
#
# C2 posits deterministic structural functions driven by exogenous noise. When that
# noise carries fewer degrees of freedom than |out(m)| -- the canonical case, and what
# the repository's own flagship example does (`minimal_model/examples.py` returns one
# draw twice, so C == D always) -- the mechanism factor P(out(m) | in(m)) is singular.
#
# Deleting such a mechanism moves probability mass onto {C != D}, which the
# observational law never visits. So the interventional law is strictly positive
# exactly where the observational mechanism factor vanishes.


def q_coupled(a: int, b: int) -> float:
    """P(C = D = 1 | A, B) for the coupled mechanism; the complement gives C = D = 0."""
    return 0.2 + 0.3 * a + 0.25 * b


def p_cd_given_ab_coupled(c: int, d: int, a: int, b: int) -> float:
    if c != d:
        return 0.0  # stoichiometric coupling: C and D are produced from one shared draw
    return q_coupled(a, b) if c == 1 else 1.0 - q_coupled(a, b)


def observational_joint_coupled() -> dict[tuple[int, ...], float]:
    joint = {}
    for x in _assignments():
        joint[tuple(x[v] for v in VARIABLES)] = (
            P_A[x["A"]]
            * P_B[x["B"]]
            * P_E[x["E"]]
            * p_cd_given_ab_coupled(x["C"], x["D"], x["A"], x["B"])
            * p_f_given_ce(x["F"], x["C"], x["E"])
        )
    return joint


def test_coupled_fixture_is_singular_where_the_intervention_puts_mass() -> None:
    """Pin the premise: the observational law is zero on {C != D}, the truth is not."""
    observational = observational_joint_coupled()
    truth = interventional_joint_delete_m1()
    off_diagonal = [x for x in _assignments() if x["C"] != x["D"]]
    assert off_diagonal
    for x in off_diagonal:
        key = tuple(x[v] for v in VARIABLES)
        assert observational[key] == 0.0
        assert truth[key] > 0.0


def test_deletion_of_a_deterministic_mechanism_is_identified_on_the_full_support() -> None:
    """Deleting a coupled mechanism must yield an estimand defined everywhere it puts mass.

    The v1 identifier is a density quotient P(V) / P(out | in), which is 0/0 on
    {C != D}. An identifier that cannot be evaluated on the support of its own
    interventional law has not identified the query; the compiler must either emit a
    division-free form or refuse.
    """
    graph = reaction_mechanism_graph()
    result = identify(graph, DeleteMechanism("m1"))
    assert isinstance(result, Identified)

    model = DiscreteModel(
        domains=DOMAINS,
        joint=observational_joint_coupled(),
        fallbacks={"C": P0_C, "D": P0_D},
    )
    truth = interventional_joint_delete_m1()

    for x in _assignments():
        key = tuple(x[v] for v in VARIABLES)
        assert evaluate(result.expression, model, x) == pytest.approx(truth[key], abs=1e-12)


# --- Replacement of a coupled mechanism by a decoupled one ------------------------
#
# rho(m') = rho(m) constrains incidence, not the function. So replacing a
# stoichiometrically coupled mechanism with one whose outputs are independent is a
# legal -- and physically natural -- query: "what if this enzyme complex were two
# independent enzymes?" It is also the case the v1 quotient identifier cannot express.

P_PRIME_C = {0: 0.4, 1: 0.6}
P_PRIME_D = {0: 0.7, 1: 0.3}


def p_cd_given_ab_replacement(c: int, d: int, a: int, b: int) -> float:
    """A full-support replacement kernel for m1: C and D become independent."""
    del a, b  # the replacement ignores its inputs; incidence is preserved, not behaviour
    return P_PRIME_C[c] * P_PRIME_D[d]


def interventional_joint_replace_m1() -> dict[tuple[int, ...], float]:
    joint = {}
    for x in _assignments():
        joint[tuple(x[v] for v in VARIABLES)] = (
            P_A[x["A"]]
            * P_B[x["B"]]
            * P_E[x["E"]]
            * p_cd_given_ab_replacement(x["C"], x["D"], x["A"], x["B"])
            * p_f_given_ce(x["F"], x["C"], x["E"])
        )
    return joint


def replacement_kernel_table() -> dict[tuple[tuple[int, ...], tuple[int, ...]], float]:
    """P_m1_prime(C,D | A,B) keyed as ((c, d), (a, b)) in sorted-variable order."""
    return {
        ((c, d), (a, b)): p_cd_given_ab_replacement(c, d, a, b)
        for c, d, a, b in itertools.product(BINARY, repeat=4)
    }


def test_replacement_of_a_deterministic_mechanism_is_identified_on_the_full_support() -> None:
    """T3 must stay defined when the *replaced* mechanism is singular.

    Ground truth is strictly positive on {C != D} because the replacement decouples the
    outputs, while the observational mechanism factor is zero there. The v1 quotient
    P(V) / P(C,D | A,B) is therefore 0/0 exactly where the replacement puts new mass.
    """
    graph = reaction_mechanism_graph()
    result = identify(graph, ReplaceMechanism("m1", replacement="m1_prime"))
    assert isinstance(result, Identified)

    model = DiscreteModel(
        domains=DOMAINS,
        joint=observational_joint_coupled(),
        replacements={"m1_prime": replacement_kernel_table()},
    )
    truth = interventional_joint_replace_m1()

    for x in _assignments():
        key = tuple(x[v] for v in VARIABLES)
        assert evaluate(result.expression, model, x) == pytest.approx(truth[key], abs=1e-12)
