from causal_hypergraphs import (
    Fallback,
    Kernel,
    Probability,
    Product,
    Quotient,
    ReplacementFactor,
    SumOut,
)


def test_product_has_canonical_equality_and_hashing() -> None:
    left = Product([Fallback("m2", ("D",)), Fallback("m1", ("C",))])
    right = Product([Fallback("m1", ("C",)), Fallback("m2", ("D",))])

    assert left == right
    assert hash(left) == hash(right)
    assert left.fingerprint() == right.fingerprint()
    assert str(left) == "P0_m1(C) * P0_m2(D)"


def test_a_joint_fallback_is_distinct_from_the_same_variables_under_another_mechanism() -> None:
    """The mechanism tag is part of the policy's identity, not decoration.

    Two mechanisms cannot produce the same variable (C4), so this pair cannot co-occur in
    one estimand -- but the nodes must still be distinguishable, or a cache keyed on the
    fingerprint would serve one policy's values for the other.
    """
    assert Fallback("m1", ("C", "D")) != Fallback("m2", ("C", "D"))
    assert Fallback("m1", ("C", "D")).fingerprint() != Fallback("m2", ("C", "D")).fingerprint()
    assert str(Fallback("m1", ("D", "C"))) == "P0_m1(C,D)"


def test_expression_scope_conditioning_and_kernel_introspection() -> None:
    expression = Product(
        [
            Quotient(Probability({"A", "B", "C"}), Probability("C", given={"A", "B"})),
            ReplacementFactor("m_prime", "C", given={"A", "B"}),
        ]
    )

    assert expression.scope() == frozenset({"A", "B", "C"})
    assert expression.conditioned_on() == frozenset({"A", "B"})
    assert expression.kernels() == (
        Kernel("probability", "P", ("A", "B", "C")),
        Kernel("probability", "P", ("C",), ("A", "B")),
        Kernel("replacement", "m_prime", ("C",), ("A", "B")),
    )


def test_sumout_renders_and_removes_eliminated_variables_from_scope() -> None:
    expression = SumOut(
        {"Z"},
        Product([Probability("Y", given={"X", "Z"}), Probability("Z", given="X")]),
    )

    assert str(expression) == "sum_{Z} P(Y | X,Z) * P(Z | X)"
    assert expression.to_latex() == r"\sum_{Z} P(Y \mid X,Z) \cdot P(Z \mid X)"
    assert expression.scope() == frozenset({"X", "Y"})
    assert expression.conditioned_on() == frozenset({"X"})
