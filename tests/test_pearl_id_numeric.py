"""Pearl estimands, checked as numbers rather than as strings.

`test_pearl_id_backend.py` pins the front-door estimand by comparing its rendered text to
a literal. That test passes today against an expression that **cannot be evaluated at
all**: it names `X_prime`, a fresh copy of `X` introduced so the inner sum does not capture
the outer do-value, and nothing in the library says what `X_prime` means. Ask the evaluator
for it and you get `KeyError: 'X_prime'`.

So the one non-trivial Pearl formula the library ships has never been checked against a
number. String equality cannot notice that, which is the argument for this file.

The oracle is generic and independent: `tests/idcorpus` builds a concrete binary SCM for an
ADMG -- one latent per bidirected edge, one noise per node -- and computes both `P(V)` and
`P(Y | do(X))` exactly by enumeration. An identifying formula is then evaluated against the
first and compared to the second. Nothing in the oracle knows how the formula was derived.
"""
from __future__ import annotations

import pytest

from causal_hypergraphs.identification import ADMG, Identified, identify_effect
from causal_hypergraphs.semantics import DiscreteModel, evaluate, with_aliases
from tests.idcorpus import random_scm

BINARY = (0, 1)


def _check(admg: ADMG, outcomes: tuple[str, ...], interventions: tuple[str, ...], seeds: int = 30):
    """Max absolute error of the identified estimand against the exact interventional law."""
    result = identify_effect(admg, outcomes=list(outcomes), interventions=list(interventions))
    assert isinstance(result, Identified), result

    worst = 0.0
    for seed in range(seeds):
        model = random_scm(admg, seed)
        observational = with_aliases(
            DiscreteModel(
                domains={name: BINARY for name in admg.nodes}, joint=model.joint()
            ),
            result.aliases,
        )
        for values in _points(len(interventions)):
            do = dict(zip(interventions, values, strict=True))
            truth = model.interventional(list(outcomes), do)
            for point in _points(len(outcomes)):
                assignment = {**do, **dict(zip(outcomes, point, strict=True))}
                got = evaluate(result.expression, observational, assignment)
                worst = max(worst, abs(got - truth[point]))
    return result, worst


def _points(width: int) -> list[tuple[int, ...]]:
    import itertools

    return list(itertools.product(BINARY, repeat=width))


def test_the_frontdoor_estimand_reproduces_the_exact_interventional_law() -> None:
    """The formula the library already ships, finally evaluated."""
    admg = ADMG(
        nodes={"X", "Y", "Z"},
        directed_edges={("X", "Z"), ("Z", "Y")},
        bidirected_edges={("X", "Y")},
    )
    result, worst = _check(admg, ("Y",), ("X",))

    assert "X_prime" in str(result.expression)
    assert result.aliases == {"X_prime": "X"}
    assert worst < 1e-12, worst


def test_the_frontdoor_estimand_is_not_merely_the_observational_law() -> None:
    """The control. A front-door graph is confounded, so an estimand that ignored the
    intervention would still be a valid-looking formula -- and would fail here.

    Without this, the check above would pass for `P(Y | X)`, or for `P(Y)`, on any model
    where the confounding happened to be weak.
    """
    admg = ADMG(
        nodes={"X", "Y", "Z"},
        directed_edges={("X", "Z"), ("Z", "Y")},
        bidirected_edges={("X", "Y")},
    )
    gaps = []
    for seed in range(30):
        model = random_scm(admg, seed)
        joint = model.joint()
        observed = {
            y: sum(p for key, p in joint.items() if key[1] == y) for y in BINARY
        }
        truth = model.interventional(["Y"], {"X": 1})
        gaps.append(abs(truth[(1,)] - observed[1]))

    assert max(gaps) > 0.02, f"confounding too weak to test anything: {max(gaps)}"


def test_a_markovian_estimand_reproduces_the_exact_interventional_law() -> None:
    """The unconfounded case, as a floor: if this fails, the oracle is wrong, not the ID."""
    admg = ADMG(
        nodes={"X", "Y", "Z"},
        directed_edges={("Z", "X"), ("X", "Y"), ("Z", "Y")},
    )
    _, worst = _check(admg, ("Y",), ("X",))

    assert worst < 1e-12, worst


def test_the_alias_is_resolved_per_kernel_not_once_for_the_model() -> None:
    """A copy is a copy *inside one kernel*, not a global rebasing of the variable.

    `P(X_prime)` and `P(Y | X_prime, Z)` must each be looked up under `X`, at the copy's
    own value, while `P(Z | X)` in the same expression keeps reading the outer `X`. A model
    that rebased `X` globally would evaluate `P(Z | X)` at the copy's value too, and the
    front-door estimand would come out wrong by a wide margin rather than by a rounding
    error -- which is exactly why the numeric gate above is the one that catches it.
    """
    admg = ADMG(
        nodes={"X", "Y", "Z"},
        directed_edges={("X", "Z"), ("Z", "Y")},
        bidirected_edges={("X", "Y")},
    )
    result = identify_effect(admg, outcomes=["Y"], interventions=["X"])
    assert isinstance(result, Identified)

    model = random_scm(admg, 0)
    honest = with_aliases(
        DiscreteModel(domains={name: BINARY for name in admg.nodes}, joint=model.joint()),
        result.aliases,
    )
    # The outer `P(Z | X)` must still read the outer X: changing only the outer X value
    # has to change the answer.
    at_one = evaluate(result.expression, honest, {"X": 1, "Y": 1})
    at_zero = evaluate(result.expression, honest, {"X": 0, "Y": 1})
    assert at_one != pytest.approx(at_zero, abs=1e-9), (at_one, at_zero)


def test_a_copy_and_its_base_coexist_inside_one_sum() -> None:
    """The shape that actually exercises per-kernel resolution.

    The front-door estimand the library ships is *nested*: `P(Z | X)` sits outside the sum
    that binds `X_prime`, so it is evaluated before the copy has a value and a model that
    rebased `X` globally would look identical. A real ID algorithm emits a **flat** sum of
    products, where a factor reading the outer `X` and a factor reading the copy live under
    the same binder -- and there a global rebase silently reads the copy's value for both.

    Written out by hand because that is the point: the mechanism has to be correct for the
    expressions the algorithm will produce, not only for the one already in the tree. The
    flat form is equal to the nested one, since `P(Z | X)` does not depend on the copy.
    """
    from causal_hypergraphs import Probability, Product, SumOut

    admg = ADMG(
        nodes={"X", "Y", "Z"},
        directed_edges={("X", "Z"), ("Z", "Y")},
        bidirected_edges={("X", "Y")},
    )
    nested = identify_effect(admg, outcomes=["Y"], interventions=["X"])
    assert isinstance(nested, Identified)

    flat = SumOut(
        ("Z", "X_prime"),
        Product(
            [
                Probability(("Z",), given=("X",)),
                Probability(("X_prime",)),
                Probability(("Y",), given=("X_prime", "Z")),
            ]
        ),
    )

    worst = 0.0
    for seed in range(30):
        model = random_scm(admg, seed)
        observational = with_aliases(
            DiscreteModel(
                domains={name: BINARY for name in admg.nodes}, joint=model.joint()
            ),
            {"X_prime": "X"},
        )
        for x in BINARY:
            truth = model.interventional(["Y"], {"X": x})
            for y in BINARY:
                got = evaluate(flat, observational, {"X": x, "Y": y})
                worst = max(worst, abs(got - truth[(y,)]))

    assert worst < 1e-12, worst
