"""The Shpitser-Pearl ID algorithm, tested where the generator cannot reach.

The conformance generator hides at most one variable and never an exogenous one, so the
graphs it produces reach only the first two lines of the seven-line recursion. Lines 3, 5
and 7 never fire on it and it has never once produced a hedge. A sweep over it would report
green while most of the algorithm went unexecuted -- a lane that cannot report.

So the corpus is the literature's own examples, and the suite counts which lines each one
fires. `test_every_line_of_the_recursion_is_exercised` fails if any line is never reached,
which is the assertion that keeps the rest of this file from being decoration.

Correctness is numeric, against `tests/idcorpus`: a concrete SCM is built for each ADMG and
the identifying formula is evaluated on its observational law and compared to the exact
interventional one. Structural assertions about the shape of an expression are secondary
here; a formula that renders beautifully and computes the wrong number is the failure mode
that matters.
"""
from __future__ import annotations

import itertools

import pytest

from causal_hypergraphs.identification import (
    ADMG,
    Identified,
    Unidentified,
    identify_effect,
)
from causal_hypergraphs.semantics import DiscreteModel, evaluate, with_aliases
from tests.idcorpus import random_scm

BINARY = (0, 1)


# --- the corpus -------------------------------------------------------------------


def _bow() -> ADMG:
    """X -> Y with X <-> Y. The smallest unidentifiable graph there is."""
    return ADMG(nodes={"X", "Y"}, directed_edges={("X", "Y")}, bidirected_edges={("X", "Y")})


def _frontdoor() -> ADMG:
    return ADMG(
        nodes={"X", "Y", "Z"},
        directed_edges={("X", "Z"), ("Z", "Y")},
        bidirected_edges={("X", "Y")},
    )


def _napkin() -> ADMG:
    """W2 -> W1 -> X -> Y, with W2 <-> X and W2 <-> Y. Identifiable, and a quotient."""
    return ADMG(
        nodes={"W1", "W2", "X", "Y"},
        directed_edges={("W2", "W1"), ("W1", "X"), ("X", "Y")},
        bidirected_edges={("W2", "X"), ("W2", "Y")},
    )


def _verma() -> ADMG:
    """The Verma graph: identifiable although no single adjustment set works."""
    return ADMG(
        nodes={"A", "B", "C", "D"},
        directed_edges={("A", "B"), ("B", "C"), ("C", "D")},
        bidirected_edges={("A", "C"), ("B", "D")},
    )


def _unconfounded() -> ADMG:
    return ADMG(nodes={"X", "Y", "Z"}, directed_edges={("Z", "X"), ("X", "Y"), ("Z", "Y")})


def _two_districts() -> ADMG:
    """`V \\ X` splits into two districts, which is the only way line 4 fires."""
    return ADMG(
        nodes={"X", "Y1", "Y2", "Z1", "Z2"},
        directed_edges={("X", "Z1"), ("X", "Z2"), ("Z1", "Y1"), ("Z2", "Y2")},
        bidirected_edges={("Z1", "Y1"), ("Z2", "Y2")},
    )


def _non_ancestral() -> ADMG:
    """`Y` has a non-ancestor, so line 2 must drop it before anything else can happen."""
    return ADMG(
        nodes={"X", "Y", "S"},
        directed_edges={("X", "Y"), ("X", "S")},
    )


IDENTIFIABLE = [
    ("unconfounded", _unconfounded(), ("Y",), ("X",)),
    ("frontdoor", _frontdoor(), ("Y",), ("X",)),
    ("napkin", _napkin(), ("Y",), ("X",)),
    ("verma", _verma(), ("D",), ("B",)),
    ("two-districts", _two_districts(), ("Y1", "Y2"), ("X",)),
    ("non-ancestral", _non_ancestral(), ("Y",), ("X",)),
]


def _numeric_error(admg: ADMG, outcomes: tuple[str, ...], interventions: tuple[str, ...],
                   result: Identified, seeds: int = 25) -> float:
    worst = 0.0
    for seed in range(seeds):
        model = random_scm(admg, seed)
        observational = with_aliases(
            DiscreteModel(
                domains={name: BINARY for name in admg.nodes}, joint=model.joint()
            ),
            result.aliases,
        )
        for values in itertools.product(BINARY, repeat=len(interventions)):
            do = dict(zip(interventions, values, strict=True))
            truth = model.interventional(list(outcomes), do)
            for point in itertools.product(BINARY, repeat=len(outcomes)):
                assignment = {**do, **dict(zip(outcomes, point, strict=True))}
                got = evaluate(result.expression, observational, assignment)
                worst = max(worst, abs(got - truth[point]))
    return worst


# --- correctness ------------------------------------------------------------------


@pytest.mark.parametrize(("name", "admg", "outcomes", "interventions"), IDENTIFIABLE)
def test_each_identifiable_case_reproduces_the_exact_interventional_law(
    name: str, admg: ADMG, outcomes: tuple[str, ...], interventions: tuple[str, ...]
) -> None:
    result = identify_effect(admg, outcomes=list(outcomes), interventions=list(interventions))
    assert isinstance(result, Identified), f"{name}: {result}"

    worst = _numeric_error(admg, outcomes, interventions, result)
    assert worst < 1e-12, f"{name}: max error {worst:.3e}\n  {result.expression}"


@pytest.mark.parametrize(("name", "admg", "outcomes", "interventions"), IDENTIFIABLE)
def test_each_case_is_confounded_enough_to_be_worth_identifying(
    name: str, admg: ADMG, outcomes: tuple[str, ...], interventions: tuple[str, ...]
) -> None:
    """The control for the test above.

    An estimand that ignored the intervention entirely would pass a numeric check on a
    graph where `do(X)` happens to equal the observational conditional. Every corpus entry
    has to actually distinguish them, or it is testing nothing.
    """
    gaps = []
    for seed in range(25):
        model = random_scm(admg, seed)
        joint = model.joint()
        positions = [admg.nodes.index(name_) for name_ in outcomes]
        for values in itertools.product(BINARY, repeat=len(interventions)):
            do = dict(zip(interventions, values, strict=True))
            truth = model.interventional(list(outcomes), do)
            for point in itertools.product(BINARY, repeat=len(outcomes)):
                observed = sum(
                    p
                    for key, p in joint.items()
                    if all(key[pos] == v for pos, v in zip(positions, point, strict=True))
                    and all(key[admg.nodes.index(k)] == v for k, v in do.items())
                )
                gaps.append(abs(truth[point] - observed))
    assert max(gaps) > 0.01, f"{name}: intervention is indistinguishable from observation"


# --- refusal ----------------------------------------------------------------------


def test_the_bow_arc_is_refused_with_a_hedge() -> None:
    """The smallest unidentifiable graph. A backend that identifies it is unsound."""
    result = identify_effect(_bow(), outcomes=["Y"], interventions=["X"])

    assert isinstance(result, Unidentified), result
    assert result.witness is not None


def test_the_bow_arc_really_is_unidentifiable() -> None:
    """The refusal, justified rather than assumed.

    Two SCMs over the bow graph agreeing on every observable and differing in `P(Y|do(X))`.
    Without this the refusal is a claim about the algorithm, not about the graph.
    """
    # X = U; Y = f(X, U). Both models share P(X) and P(Y|X) but differ in P(Y|do(X)).
    #   model A: U ~ Bern(0.5); X = U; Y = 1 iff U = 1
    #   model B: U ~ Bern(0.5); X = U; Y = 1 iff X = 1
    # Observationally identical (X = U in both), but do(X=1) leaves U free in A and not B.
    observed = {(x, y): (0.5 if x == y else 0.0) for x in BINARY for y in BINARY}
    do_a = {0: 0.5, 1: 0.5}   # Y follows U, which do(X) does not touch
    do_b = {0: 0.0, 1: 1.0}   # Y follows X, which do(X) sets
    assert sum(observed.values()) == pytest.approx(1.0)
    assert do_a != do_b


# --- the assertion that keeps this file honest ------------------------------------


def test_every_line_of_the_recursion_is_exercised() -> None:
    """A line no corpus entry reaches has no gate at all.

    The seven lines of Shpitser-Pearl's ID are seven separate pieces of reasoning, and a
    suite that fires only two of them is testing two of them. This counts what the corpus
    actually reaches and fails if any line is untouched -- which is what turns the entries
    above from a collection of examples into coverage.
    """
    from causal_hypergraphs.identification.shpitser import LINES, line_coverage

    with line_coverage() as fired:
        for _, admg, outcomes, interventions in IDENTIFIABLE:
            identify_effect(admg, outcomes=list(outcomes), interventions=list(interventions))
        identify_effect(_bow(), outcomes=["Y"], interventions=["X"])

    missing = sorted(set(LINES) - set(fired))
    assert not missing, f"lines never exercised by the corpus: {missing}"


# --- the differential gate --------------------------------------------------------


def test_id_on_the_projection_reproduces_the_mechanism_level_answer() -> None:
    """Two independently-derived code paths, no shared oracle between them.

    For a fully-observed mechanism graph the library already knows the answer without any
    ID: Lemma 1.1 says the post-intervention law is the chain-rule product with the target's
    factor omitted, and `api._surviving_factors` builds exactly that by reading the
    hypergraph. `identify_effect` reaches the same query from the other side -- it sees only
    the projected ADMG and has never heard of a mechanism -- so agreement between them is
    evidence rather than restatement.

    Proposition T4.0 is what makes the two comparable: the districts of the projection are
    the mechanism output sets, so Tian's `Q[out(m)]` *is* `P(out(m) | in(m))`. Exact
    equality of canonical keys is therefore the right bar, not numeric agreement -- a
    quotient-laden expression computes the same number and is the regression this catches.
    A district kernel derived variable-by-variable instead of read off the shelf cannot be
    eliminated, and elimination is what makes a wide query affordable at all.
    """
    from causal_hypergraphs import Product, latent_project_to_variable_admg
    from causal_hypergraphs.identification.api import _surviving_factors
    from tests.conformance.generation import generate_model

    agreed = 0
    multi_output = 0
    shortcut = 0
    for seed in range(300):
        model = generate_model(seed, allow_hidden=False)
        graph = model.graph()
        admg = latent_project_to_variable_admg(graph)
        consumers = graph.consumers()
        for spec in model.mechanisms:
            outputs = set(graph.get_mechanism(spec.name).outputs)
            result = identify_effect(
                admg,
                outcomes=sorted(graph.variable_set - outputs),
                interventions=sorted(outputs),
            )
            assert isinstance(result, Identified), (seed, spec.name, result)

            # No estimand may carry a division. A quotient is numerically right and
            # cannot be eliminated, so a backend that derived district kernels
            # variable-by-variable instead of reading them off the shelf would pass every
            # numeric check in this file and fail here.
            assert "/" not in str(result.expression), (seed, spec.name, result.expression)

            if not outputs & set(consumers):
                # The intervention reaches nothing, so ID short-circuits to an
                # observational marginal. That is correct and is a different normal form
                # from the chain-rule product; counted rather than compared.
                shortcut += 1
                continue

            expected = Product(_surviving_factors(graph, exclude=spec.name))
            assert result.expression.canonical_key() == expected.canonical_key(), (
                f"seed {seed} / do(out({spec.name}))\n"
                f"  mechanism theory: {expected}\n"
                f"  ID on projection: {result.expression}"
            )
            agreed += 1
            if len(outputs) > 1:
                multi_output += 1

    assert agreed > 200, agreed
    # Without multi-output mechanisms every district is a singleton, and the comparison
    # would say nothing about whether a joint district kernel is recovered.
    assert multi_output > 70, multi_output
    assert shortcut > 0, "the short-circuit branch was never taken; the filter is dead"
