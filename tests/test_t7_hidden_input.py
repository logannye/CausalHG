"""The genuine T7 case: a hidden *input* leaves the target's boundary unobserved.

Hidden outputs are settled elsewhere -- one that reaches an observation is unidentifiable,
one that reaches nothing is summed out of the declared policy. What is left is the case the
theory actually describes: `in(m)` contains a variable nobody measured, so the mechanism
factor `P(out(m) | in(m))` is not observable and the local factor swap is unavailable.

`THEOREM_H1_PLUS.md` section 3 says this reduces to Pearl identification on the
latent-projected ADMG, via the mixture

    P(Y | delete m) = sum_x P0(x) * P(Y | do(out(m) = x))

which holds because a deletion policy is *unconditional*: it does not read `in(m)`, so it
factors straight out of the truncated factorization. Each term is an ordinary Pearl query
and the sum is over a table the caller supplied.

Two consequences the previous vertical slice got wrong. `do(out(m))` is a multi-variable
intervention when the mechanism has several outputs, which Shpitser-Pearl handles natively
-- the old `len(out(m)) != 1` refusal was a limit of the three-case stub behind it, not of
the theory. And a *replacement* kernel reads `in(m)`, so the mixture above does not apply
to it at all.
"""
from __future__ import annotations

import itertools

import pytest

from causal_hypergraphs import (
    DeleteMechanism,
    Identified,
    MechanismGraph,
    ReplaceMechanism,
    Unknown,
    identify,
)
from causal_hypergraphs.semantics import DiscreteModel, evaluate, with_aliases

BINARY = (0, 1)


def _multi_output_hidden_input() -> MechanismGraph:
    """`m` has a hidden input and TWO outputs -- the shape the old slice refused."""
    return MechanismGraph(
        variables={"h", "A", "C", "D", "Y"},
        observed_variables={"A", "C", "D", "Y"},
        mechanisms={
            "m_h": {"inputs": (), "outputs": ("h",), "latent": True},
            "m": {"inputs": ("A", "h"), "outputs": ("C", "D")},
            "m_y": {"inputs": ("h",), "outputs": ("Y",)},
        },
    )


def test_a_multi_output_target_is_no_longer_refused_for_its_arity() -> None:
    """`do(out(m))` over several variables is one Pearl query, not an unsupported case."""
    result = identify(
        _multi_output_hidden_input(), DeleteMechanism("m", outcomes={"Y"}), allow_t7=True
    )

    assert not (
        isinstance(result, Unknown) and "one target output" in result.reason
    ), result


def test_the_reduction_is_the_mixture_over_the_declared_policy() -> None:
    """The estimand must be `sum over out(m) of P0 times a Pearl term`, visibly."""
    graph = MechanismGraph(
        variables={"W", "X", "Y", "Z"},
        observed_variables={"X", "Y", "Z"},
        mechanisms={
            "m_W": {"inputs": (), "outputs": ("W",), "latent": True},
            "m_x": {"inputs": ("W",), "outputs": ("X",)},
            "m_z": {"inputs": ("X",), "outputs": ("Z",)},
            "m_y": {"inputs": ("W", "Z"), "outputs": ("Y",)},
        },
    )
    result = identify(graph, DeleteMechanism("m_x", outcomes={"Y"}), allow_t7=True)

    assert isinstance(result, Identified), result
    assert str(result.expression).startswith("sum_{X} P0_m_x(X) *")
    assert result.expression.scope() == frozenset({"Y"})
    assert "W" not in result.expression.footprint()  # the hidden input never appears


def test_the_t7_estimand_reproduces_the_exact_interventional_law() -> None:
    """The whole point, checked as a number against ground truth the compiler never sees.

    The model is built by hand rather than generated, because the conformance generator
    never produces a hidden variable with two observed children -- so it cannot make this
    case at all, and a sweep over it would be reporting on something else.
    """
    graph = MechanismGraph(
        variables={"W", "X", "Y", "Z"},
        observed_variables={"X", "Y", "Z"},
        mechanisms={
            "m_W": {"inputs": (), "outputs": ("W",), "latent": True},
            "m_x": {"inputs": ("W",), "outputs": ("X",)},
            "m_z": {"inputs": ("X",), "outputs": ("Z",)},
            "m_y": {"inputs": ("W", "Z"), "outputs": ("Y",)},
        },
    )
    result = identify(graph, DeleteMechanism("m_x", outcomes={"Y"}), allow_t7=True)
    assert isinstance(result, Identified), result

    policy = {(0,): 0.35, (1,): 0.65}
    for seed in range(20):
        law, truth = _front_door_model(seed, policy)
        model = with_aliases(
            DiscreteModel(
                domains={name: BINARY for name in ("X", "Y", "Z")},
                joint=law,
                fallbacks={"m_x": policy},
            ),
            result.aliases,
        )
        for y in BINARY:
            got = evaluate(result.expression, model, {"Y": y})
            assert got == pytest.approx(truth[y], abs=1e-12), (seed, y, got, truth[y])


def _front_door_model(seed: int, policy: dict) -> tuple[dict, dict]:
    """`W -> X -> Z -> Y`, `W -> Y`, with `W` hidden. Exact laws, by enumeration."""
    import random

    rng = random.Random(seed)
    p_w = rng.uniform(0.25, 0.75)
    p_x = {w: rng.uniform(0.15, 0.85) for w in BINARY}
    p_z = {x: rng.uniform(0.15, 0.85) for x in BINARY}
    p_y = {(w, z): rng.uniform(0.15, 0.85) for w in BINARY for z in BINARY}

    def w_weight(w: int) -> float:
        return p_w if w == 1 else 1.0 - p_w

    observational: dict[tuple[int, ...], float] = {}
    for x, y, z in itertools.product(BINARY, BINARY, BINARY):
        total = 0.0
        for w in BINARY:
            weight = w_weight(w)
            weight *= p_x[w] if x == 1 else 1.0 - p_x[w]
            weight *= p_z[x] if z == 1 else 1.0 - p_z[x]
            weight *= p_y[(w, z)] if y == 1 else 1.0 - p_y[(w, z)]
            total += weight
        observational[(x, y, z)] = total

    # delete(m_x): X is drawn from the policy instead of from W.
    truth = {y: 0.0 for y in BINARY}
    for x, y, z in itertools.product(BINARY, BINARY, BINARY):
        for w in BINARY:
            weight = w_weight(w) * policy[(x,)]
            weight *= p_z[x] if z == 1 else 1.0 - p_z[x]
            weight *= p_y[(w, z)] if y == 1 else 1.0 - p_y[(w, z)]
            truth[y] += weight
    return observational, truth


# --- replacement is a different identity, and is refused --------------------------


def test_a_replacement_under_a_hidden_boundary_is_refused_with_the_reason() -> None:
    """A replacement kernel reads `in(m)`, so the deletion mixture does not apply to it.

    `P(Y | replace m -> m') = sum over out(m), in(m) of P_m'(out | in) * P(Y, in(m) | do(out(m)))`
    -- the policy is conditional, so the Pearl query it needs is the *joint* of the outcome
    with the mechanism's inputs, which is a strictly harder query than deletion's. Folding
    it into the deletion identity by symmetry would be wrong, so it is refused and the
    refusal carries the identity that would be needed.
    """
    graph = MechanismGraph(
        variables={"W", "X", "Y"},
        observed_variables={"X", "Y"},
        mechanisms={
            "m_W": {"inputs": (), "outputs": ("W",), "latent": True},
            "m_x": {"inputs": ("W",), "outputs": ("X",)},
            "m_y": {"inputs": ("W", "X"), "outputs": ("Y",)},
        },
    )
    result = identify(
        graph, ReplaceMechanism("m_x", "m_x_prime", {"Y"}), allow_t7=True
    )

    assert isinstance(result, Unknown), result
    assert "conditional" in result.reason.lower() or "reads" in result.reason.lower()
    assert "in(m" in result.reason or "inputs" in result.reason.lower()


def test_a_pearl_hedge_is_reported_as_open_not_as_refuted() -> None:
    """A hedge in the projection is evidence, not a proof about the mechanism query.

    Two gaps stand between them, and neither is closed. Shpitser-Pearl completeness refutes
    identifiability over *all* semi-Markovian models of the ADMG, while the models this
    projection can come from are a strictly smaller class -- one noise per mechanism whose
    children are exactly `out(m)`, one producer per variable, plus any declared output
    equalities. And the mechanism query is a *mixture* against a policy the caller supplies;
    every term of a mixture failing does not make the mixture fail.

    Closing that gap is conjecture H1+, which `THEOREM_H1_PLUS.md` marks open. So the
    verdict is `Unknown` with the hedge attached, not `Unidentified`. `Unidentified` is the
    strongest claim the library makes and it is not available here.
    """
    from causal_hypergraphs import Unidentified

    graph = MechanismGraph(
        variables={"W", "X", "Y"},
        observed_variables={"X", "Y"},
        mechanisms={
            "m_W": {"inputs": (), "outputs": ("W",), "latent": True},
            "m_x": {"inputs": ("W",), "outputs": ("X",)},
            "m_y": {"inputs": ("W", "X"), "outputs": ("Y",)},
        },
    )
    result = identify(graph, DeleteMechanism("m_x", outcomes={"Y"}), allow_t7=True)

    assert not isinstance(result, Unidentified), result
    assert isinstance(result, Unknown), result
    assert "H1+" in result.reason
    assert any("hedge" in s.lower() for s in result.suggestions), result.suggestions
