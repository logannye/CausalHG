"""`P0` is a per-mechanism joint kernel, not a product of per-variable laws.

Deleting a mechanism orphans *all* of its outputs at once, so the policy that says what
happens to them is one object over `out(m)`, not one object per variable. While `P0`
factorized as a product, `delete(m)` silently *forced* the orphaned outputs independent,
and the framework could not state its own motivating case: removing a mechanism whose
outputs are stoichiometrically coupled, where the coupling survives the removal.

These tests pin the generalization from the outside. The load-bearing one is
`test_a_non_factorizing_fallback_is_expressible`: its target law is not a product
measure, so no product of per-variable fallbacks can produce it at any parameter values.
That test is unsatisfiable under the old type and passes under the new one, which is what
makes this a change in expressive power rather than a change in spelling.

`THEOREM_T2_T3.md` Remark T3.3 records that every result goes through unchanged: the
proofs only use that the replacement factor is a fixed kernel not depending on the rest
of the model, never that it factorizes.
"""
from __future__ import annotations

import itertools

import pytest

from causal_hypergraphs import DeleteMechanism, Fallback, Identified, MechanismGraph, identify
from causal_hypergraphs.semantics import DiscreteModel, MissingKernel, evaluate

BINARY = (0, 1)

# P(A) for the one exogenous variable.
P_A = {0: 0.4, 1: 0.6}

# The coupled fallback: C and D are equal with probability one, each marginally uniform.
# Its marginals are uniform, so the product of its own marginals puts 0.25 on every cell
# -- the product form cannot reach this law from any parameterization.
COUPLED_FALLBACK = {(0, 0): 0.5, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.5}

# A fallback that does factorize, to show the product form is a special case and not lost.
FACTORIZING_FALLBACK = {(0, 0): 0.12, (0, 1): 0.28, (1, 0): 0.18, (1, 1): 0.42}


def _graph() -> MechanismGraph:
    """A -> m1 -> {C, D}. Deleting m1 orphans C and D together."""
    return MechanismGraph(
        variables={"A", "C", "D"},
        mechanisms={"m1": {"inputs": ("A",), "outputs": ("C", "D")}},
    )


def _observational_joint() -> dict[tuple[int, ...], float]:
    """P(A, C, D) with C and D driven by A through a coupled kernel."""
    # P(C, D | A): equal to A with probability 0.8, jointly flipped otherwise.
    kernel = {
        0: {(0, 0): 0.8, (0, 1): 0.05, (1, 0): 0.05, (1, 1): 0.1},
        1: {(0, 0): 0.1, (0, 1): 0.05, (1, 0): 0.05, (1, 1): 0.8},
    }
    return {
        (a, c, d): P_A[a] * kernel[a][(c, d)]
        for a, c, d in itertools.product(BINARY, BINARY, BINARY)
    }


def _model(fallback: dict[tuple[int, ...], float]) -> DiscreteModel:
    return DiscreteModel(
        domains={"A": BINARY, "C": BINARY, "D": BINARY},
        joint=_observational_joint(),
        fallbacks={"m1": fallback},
    )


def _identified() -> Identified:
    result = identify(_graph(), DeleteMechanism("m1"))
    assert isinstance(result, Identified), result
    return result


def _fallback_nodes(expression: object) -> list[Fallback]:
    from causal_hypergraphs import Product

    if isinstance(expression, Fallback):
        return [expression]
    if isinstance(expression, Product):
        return [node for factor in expression.factors for node in _fallback_nodes(factor)]
    return []


def test_deletion_emits_one_joint_fallback_over_all_orphaned_outputs() -> None:
    """One mechanism deleted, one fallback node -- over `out(m)`, not per variable."""
    nodes = _fallback_nodes(_identified().expression)

    assert len(nodes) == 1, f"expected a single joint fallback, got {nodes}"
    assert nodes[0].variables == ("C", "D")
    assert nodes[0].mechanism == "m1"


def test_a_non_factorizing_fallback_is_expressible() -> None:
    """The case the product form could not state, checked against the exact law.

    Target law: P(A) * P0^m1(C, D) with P0 supported on the diagonal. Any product of
    per-variable fallbacks yields a law under which C and D are conditionally independent
    given A; this one is not, so agreement here is only reachable with a joint kernel.
    """
    identified = _identified()
    model = _model(COUPLED_FALLBACK)

    for a, c, d in itertools.product(BINARY, BINARY, BINARY):
        assignment = {"A": a, "C": c, "D": d}
        expected = P_A[a] * COUPLED_FALLBACK[(c, d)]
        assert evaluate(identified.expression, model, assignment) == pytest.approx(expected)


def test_the_diagonal_fallback_is_not_a_product_measure() -> None:
    """Guard the guard: if the target law factorized, the test above would prove nothing."""
    marginal_c = {c: sum(COUPLED_FALLBACK[(c, d)] for d in BINARY) for c in BINARY}
    marginal_d = {d: sum(COUPLED_FALLBACK[(c, d)] for c in BINARY) for d in BINARY}

    assert any(
        COUPLED_FALLBACK[(c, d)] != pytest.approx(marginal_c[c] * marginal_d[d])
        for c, d in itertools.product(BINARY, BINARY)
    ), "the fixture factorizes, so it cannot discriminate joint from product fallbacks"


def test_a_factorizing_fallback_still_works() -> None:
    """Generalizing the type must not cost the case that already worked."""
    identified = _identified()
    model = _model(FACTORIZING_FALLBACK)

    for a, c, d in itertools.product(BINARY, BINARY, BINARY):
        assignment = {"A": a, "C": c, "D": d}
        expected = P_A[a] * FACTORIZING_FALLBACK[(c, d)]
        assert evaluate(identified.expression, model, assignment) == pytest.approx(expected)


def test_a_missing_fallback_kernel_is_loud() -> None:
    """A fallback the model does not supply must raise, never default to a marginal."""
    identified = _identified()
    model = DiscreteModel(
        domains={"A": BINARY, "C": BINARY, "D": BINARY},
        joint=_observational_joint(),
        fallbacks={},
    )

    with pytest.raises(MissingKernel):
        evaluate(identified.expression, model, {"A": 0, "C": 0, "D": 0})


def test_a_fallback_table_missing_a_cell_is_loud() -> None:
    """A partially specified joint is a modelling error, not an implicit zero."""
    identified = _identified()
    model = _model({(0, 0): 0.5, (1, 1): 0.5})  # (0,1) and (1,0) absent

    with pytest.raises(MissingKernel):
        evaluate(identified.expression, model, {"A": 0, "C": 0, "D": 1})
