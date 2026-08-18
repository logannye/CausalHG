"""Feedback is a fact about biology, and C1 rejected it at construction.

`MechanismGraph.validate` raised `C1 violation: mechanism dependency graph is cyclic`, so a
fifty-link chain with one *completely disjoint* two-cycle somewhere else could not be built
at all. Not "the loop is unanswerable" -- the whole object was unavailable, and no question
about any part of it could be asked. Most regulatory networks have a loop.

What replaces it is a per-query condition rather than a global one. Lemma 1.1's proof needs
acyclicity of the sub-system it is applied to, not of the ambient graph: an
ancestrally-closed set of variables is self-contained, its mechanisms' noises are
independent by C2, and its marginal therefore factorizes on its own. So the question is
whether the *closure of the query* is acyclic, and a cycle somewhere else is irrelevant.

Both halves of that are measured, not argued:

- **Sufficient.** With a genuine two-cycle downstream, the observational conditional still
  recovers the structural kernel and the estimand matches the true post-deletion law to
  Monte-Carlo error. `test_a_cycle_downstream_leaves_the_answer_exact`.
- **Necessary.** With the cycle *inside* the closure, the same machinery is wrong by 68%,
  because for a mechanism on a cycle the observational conditional is not its structural
  kernel -- the two variables are mutually determined.
  `test_the_condition_is_not_decorative`.

And a deletion whose target lies on a cycle is not identifiable at all: over the two-cycle
`X = aY + u1`, `Y = bX + u2`, there are thousands of parameter settings with the identical
observational law and different post-deletion laws, some with the opposite sign.
"""
from __future__ import annotations

import numpy as np
import pytest

from causal_hypergraphs import (
    DeleteMechanism,
    Identified,
    MechanismGraph,
    identify,
)


def _two_cycle(**kwargs) -> MechanismGraph:
    """`X <-> Y` as mechanisms: `m1: Y -> X` and `m2: X -> Y`. C4 holds -- one producer each."""
    return MechanismGraph(
        variables={"X", "Y"},
        mechanisms={
            "m1": {"inputs": ("Y",), "outputs": ("X",)},
            "m2": {"inputs": ("X",), "outputs": ("Y",)},
        },
        **kwargs,
    )


# --- it can be built at all -------------------------------------------------------


def test_a_cyclic_mechanism_graph_can_be_constructed() -> None:
    graph = _two_cycle()

    assert not graph.is_mechanism_acyclic()
    assert graph.variable_set == frozenset({"X", "Y"})


def test_the_graph_names_its_own_cycles() -> None:
    """A caller has to be able to see the loop, or the lifted constraint is just silence."""
    graph = _two_cycle()

    assert graph.cyclic_mechanisms == frozenset({"m1", "m2"})
    assert graph.mechanism_components() == (("m1", "m2"),)


def test_an_acyclic_graph_reports_no_cycles_and_singleton_components() -> None:
    graph = MechanismGraph(
        variables={"A", "B", "C"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("B",)},
            "m2": {"inputs": ("B",), "outputs": ("C",)},
        },
    )

    assert graph.is_mechanism_acyclic()
    assert graph.cyclic_mechanisms == frozenset()
    assert graph.mechanism_components() == (("m1",), ("m2",))


# --- the capability that motivates the whole change -------------------------------


def _chain_with(loop: bool) -> MechanismGraph:
    variables = {f"g{i}" for i in range(21)}
    mechanisms: dict[str, dict] = {
        f"m{i}": {"inputs": (f"g{i}",), "outputs": (f"g{i + 1}",)} for i in range(20)
    }
    if loop:
        variables |= {"L1", "L2"}
        mechanisms["loop_a"] = {"inputs": ("L2",), "outputs": ("L1",)}
        mechanisms["loop_b"] = {"inputs": ("L1",), "outputs": ("L2",)}
    return MechanismGraph(variables=variables, mechanisms=mechanisms)


def test_a_disjoint_cycle_does_not_change_an_unrelated_answer() -> None:
    """The headline, and the non-regression gate in one assertion.

    The same query against the same chain, once without a loop elsewhere in the graph and
    once with one, must produce the *identical* estimand. Anything else means the cycle
    leaked into a part of the system it has no path to.
    """
    query = DeleteMechanism("m0", outcomes={"g5"})

    without = identify(_chain_with(loop=False), query)
    with_loop = identify(_chain_with(loop=True), query)

    assert isinstance(without, Identified)
    assert isinstance(with_loop, Identified), with_loop
    assert with_loop.expression.canonical_key() == without.expression.canonical_key()
    assert with_loop.theorem == without.theorem


def test_a_query_whose_closure_is_acyclic_identifies_even_when_the_graph_is_not() -> None:
    """The cycle is downstream of the outcome, so it is not in the closure at all."""
    graph = MechanismGraph(
        variables={"A", "B", "C", "D", "E"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("B",)},
            "m2": {"inputs": ("B",), "outputs": ("C",)},
            "m3": {"inputs": ("C", "E"), "outputs": ("D",)},   # D <-> E, downstream of C
            "m4": {"inputs": ("D",), "outputs": ("E",)},
        },
    )
    result = identify(graph, DeleteMechanism("m1", outcomes={"C"}))

    assert isinstance(result, Identified), result
    assert "D" not in result.expression.footprint()
    assert "E" not in result.expression.footprint()


# --- and the refusal --------------------------------------------------------------


def test_a_query_whose_closure_is_cyclic_is_refused() -> None:
    """The closure of `Y` is `{X, Y}`, which is the cycle itself."""
    result = identify(_two_cycle(), DeleteMechanism("m1", outcomes={"Y"}))

    assert not isinstance(result, Identified), result
    assert "cycl" in getattr(result, "reason", "").lower()


def test_the_refusal_names_the_mechanisms_on_the_cycle() -> None:
    result = identify(_two_cycle(), DeleteMechanism("m1", outcomes={"Y"}))

    assert set(getattr(result, "missing_variables", ())) or "m2" in str(result.__dict__)


def test_a_full_joint_query_on_a_cyclic_graph_is_refused() -> None:
    """With no `outcomes` the query is the whole joint, so every mechanism's kernel is
    needed and any cycle anywhere reaches it.

    This is the one query for which "a cycle elsewhere is irrelevant" is never true, since
    there is no elsewhere. The ancestral reduction is what makes a cycle avoidable, and
    without outcomes there is nothing to reduce to.
    """
    graph = MechanismGraph(
        variables={"X", "Y", "Q", "R"},
        mechanisms={
            "m1": {"inputs": ("Y",), "outputs": ("X",)},
            "m2": {"inputs": ("X",), "outputs": ("Y",)},
            "far": {"inputs": ("Q",), "outputs": ("R",)},   # nowhere near the loop
        },
    )

    narrow = identify(graph, DeleteMechanism("far", outcomes={"R"}))
    assert isinstance(narrow, Identified), narrow

    whole = identify(graph, DeleteMechanism("far"))
    assert not isinstance(whole, Identified), whole
    assert "cycl" in getattr(whole, "reason", "").lower()


# --- the two measurements the condition rests on ----------------------------------


def _solve_two_cycle(a: float, b: float, u1: np.ndarray, u2: np.ndarray):
    """Exact fixed point of `X = aY + u1`, `Y = bX + u2`. Unique when `|ab| < 1`."""
    d = 1.0 - a * b
    x = (u1 + a * u2) / d
    y = (b * u1 + u2) / d
    assert np.abs(x - (a * y + u1)).max() < 1e-9
    assert np.abs(y - (b * x + u2)).max() < 1e-9
    return x, y


def test_a_cycle_downstream_leaves_the_answer_exact() -> None:
    """Sufficiency, measured. A genuine two-cycle downstream changes nothing upstream.

    `A -> B -> C`, then `C` feeds a two-cycle `D <-> E`. The estimand for
    `P(C | delete(m1))` uses the observational `P(C | B)`, and that has to still be m2's
    structural kernel even though the ambient system is not acyclic.
    """
    rng = np.random.default_rng(0)
    n = 400_000
    a1, a2, a3c, a3e, a4 = 0.7, 0.4, 0.5, 0.6, 0.5

    upstream = rng.normal(0, 1, n)
    u1, u2, u3, u4 = (rng.normal(0, 1, n) for _ in range(4))
    b = a1 * upstream + u1
    c = a2 * b + u2
    d = (a3c * c + a3e * u4 + u3) / (1 - a3e * a4)
    e = a4 * d + u4
    assert np.abs(d - (a3c * c + a3e * e + u3)).max() < 1e-9

    slope = np.cov(c, b)[0, 1] / np.var(b)
    residual = np.var(c - slope * b)
    assert slope == pytest.approx(a2, abs=5e-3)
    assert residual == pytest.approx(1.0, abs=1e-2)

    policy_variance = 4.0
    estimand = slope**2 * policy_variance + residual
    drawn = rng.normal(0, np.sqrt(policy_variance), n)
    truth = np.var(a2 * drawn + rng.normal(0, 1, n))
    assert estimand == pytest.approx(truth, rel=0.02)


def test_the_condition_is_not_decorative() -> None:
    """Necessity, measured. Inside the closure the same machinery is badly wrong.

    For a mechanism on a cycle the observational conditional is *not* its structural
    kernel, because the two variables are mutually determined. Applying the truncated
    factorization anyway gives an answer off by more than half, so the check that refuses
    it is load-bearing rather than cautious.
    """
    rng = np.random.default_rng(1)
    n = 400_000
    a, b = 0.6, 0.5
    u1, u2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    x, y = _solve_two_cycle(a, b, u1, u2)

    slope = np.cov(y, x)[0, 1] / np.var(x)
    residual = np.var(y - slope * x)
    assert slope != pytest.approx(b, abs=0.1), "the fixture must actually be confounded"

    policy_variance = 4.0
    estimand = slope**2 * policy_variance + residual
    drawn = rng.normal(0, np.sqrt(policy_variance), n)
    truth = np.var(b * drawn + rng.normal(0, 1, n))
    assert abs(estimand - truth) / truth > 0.4


def test_a_deletion_on_a_cycle_is_not_identifiable_at_all() -> None:
    """Two models over the same graph, identical observationally, different after the
    deletion -- which is what non-identifiability means.

    Found by solving rather than by searching: fixing `a` pins `b` through a quadratic and
    then the noise variances follow, so the observationally-equivalent set is a curve and a
    grid over both coefficients would miss it entirely.
    """

    def observational(a, b, s1, s2):
        d = (1 - a * b) ** 2
        return np.array([(b * b * s1 + s2) / d, (s1 + a * a * s2) / d, (b * s1 + a * s2) / d])

    def after_delete(a, b, s1, s2, *, policy_variance):
        # delete(m1): X is drawn from the policy; only m2 survives, so only b and s2 do.
        return np.array([b * b * policy_variance + s2, policy_variance, b * policy_variance])

    reference = (0.6, 0.5, 1.0, 1.0)
    vx, vy, cxy = observational(*reference)

    equivalents = []
    for a in np.linspace(-0.95, 0.95, 3801):
        if abs(a) < 0.2:
            continue
        quad = (a * a * cxy - a * vy, vy - a * a * vx, a * vx - cxy)
        discriminant = quad[1] ** 2 - 4 * quad[0] * quad[2]
        if abs(quad[0]) < 1e-12 or discriminant < 0:
            continue
        for root in (
            (-quad[1] + np.sqrt(discriminant)) / (2 * quad[0]),
            (-quad[1] - np.sqrt(discriminant)) / (2 * quad[0]),
        ):
            if abs(a * root) >= 0.98 or abs(root) < 0.2:
                continue
            s1 = (1 - a * root) * (vy - a * cxy)
            s2 = (1 - a * root) * (cxy - root * vy) / a
            if s1 <= 0.05 or s2 <= 0.05:
                continue
            candidate = (a, root, s1, s2)
            if np.max(np.abs(observational(*candidate) - np.array([vx, vy, cxy]))) > 1e-9:
                continue
            equivalents.append(candidate)

    assert len(equivalents) > 100, len(equivalents)
    baseline = after_delete(*reference, policy_variance=1.0)
    gaps = [
        np.max(np.abs(baseline - after_delete(*other, policy_variance=1.0)))
        for other in equivalents
    ]
    assert max(gaps) > 1.0, max(gaps)
    # Both coefficients stay well away from zero, so this is not a degenerate model whose
    # edge is simply absent -- the cycle is really there in both.
    assert any(abs(other[1]) > 0.25 for other in equivalents)


# --- what must refuse rather than answer ------------------------------------------


def test_d_separation_refuses_on_a_cyclic_graph() -> None:
    """`T1`'s soundness proof rests on C1, so outside C1 its verdict guarantees nothing.

    Lemma 2.1 of `THEOREM_T1.md` establishes that the noise-augmented blowup is a DAG, and
    its proof ends "C1 forbids cycles in `G_E`". Step 1 of the soundness proof then uses
    exactly that: ancestral sampling on a DAG is what makes the law Markov with respect to
    it. Remove C1 and the lemma is false, the Markov property is unavailable, and a
    returned boolean would be a verdict with nothing behind it.

    Returning `False` would be no safer than returning `True`: callers read a separation
    oracle in both directions. (For *linear* cyclic systems d-separation does happen to be
    sound -- Spirtes 1995 -- but the library assumes no linearity, so it cannot rely on it.)
    """
    from causal_hypergraphs import d_separated

    with pytest.raises(Exception, match="cycl"):
        d_separated(_two_cycle(), "X", "Y", ())


def test_the_covariate_check_refuses_on_a_cyclic_graph() -> None:
    """It is built on `d_separated`, so it inherits the same limit rather than hiding it."""
    from causal_hypergraphs import check_covariates

    graph = MechanismGraph(
        variables={"X", "Y", "Z"},
        mechanisms={
            "m1": {"inputs": ("Y",), "outputs": ("X",)},
            "m2": {"inputs": ("X",), "outputs": ("Y",)},
            "m3": {"inputs": ("X",), "outputs": ("Z",)},
        },
    )
    with pytest.raises(Exception, match="cycl"):
        check_covariates(graph, DeleteMechanism("m3"), "Z", ["X"])


def test_the_latent_projection_refuses_by_name_on_a_cyclic_graph() -> None:
    """A Pearl ADMG is acyclic by definition, so a cyclic mechanism graph has no projection.

    It used to reach `ADMG.__init__` and surface as `ADMG directed component must be
    acyclic` -- an error about an internal invariant, from a function the caller asked
    about a mechanism graph.
    """
    from causal_hypergraphs import latent_project_to_variable_admg

    with pytest.raises(ValueError) as raised:
        latent_project_to_variable_admg(_two_cycle())

    message = str(raised.value)
    assert "mechanism graph" in message, message
    assert "m1" in message and "m2" in message, message
    assert "ADMG directed component" not in message, message


def test_an_acyclic_graph_is_untouched_by_any_of_these_guards() -> None:
    """The guards must be invisible where C1 holds, which is every existing use."""
    from causal_hypergraphs import (
        check_covariates,
        d_separated,
        latent_project_to_variable_admg,
    )

    graph = MechanismGraph(
        variables={"A", "B", "C"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("B",)},
            "m2": {"inputs": ("B",), "outputs": ("C",)},
        },
    )

    assert isinstance(d_separated(graph, "A", "C", ("B",)), bool)
    assert latent_project_to_variable_admg(graph).nodes == ("A", "B", "C")
    assert check_covariates(graph, DeleteMechanism("m2"), "C", ["A"]).verdicts


def test_relevant_mechanisms_is_keyed_on_outputs_not_inputs() -> None:
    """Pinned directly, because the cycle check cannot distinguish the two.

    A mechanism reading *from* the closure and writing outside it contributes no kernel to
    the estimand -- its factor is dropped by the ancestral reduction. Keying on inputs
    would include it. On a *cycle* the two criteria happen to coincide, because a cycle
    member's inputs are another member's outputs and the closure takes output groups whole,
    so `cycles_reaching` agrees under either definition and a mutation there survives every
    other test in this file. The definition is therefore pinned where it is visible.
    """
    from causal_hypergraphs.identification.api import (
        ancestral_closure,
        relevant_mechanisms,
    )

    graph = MechanismGraph(
        variables={"A", "B", "C"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("B",)},
            "m2": {"inputs": ("B",), "outputs": ("C",)},  # reads B, writes outside
        },
    )
    closure = ancestral_closure(graph, ("B",))
    assert closure == frozenset({"A", "B"})

    assert relevant_mechanisms(graph, ("B",)) == frozenset({"m1"})
    keyed_on_inputs = {
        name
        for name in graph.mechanisms
        if frozenset(graph.get_mechanism(name).inputs) <= closure
    }
    assert keyed_on_inputs == frozenset({"m1", "m2"}), "the fixture must distinguish them"


def test_a_mechanism_with_no_outputs_supplies_no_kernel() -> None:
    """`P(nothing | in)` is not a factor, so such a mechanism is never 'needed'.

    It cannot lie on a cycle either -- with no outputs it has no outgoing dependency edge --
    so this is about the definition being right rather than about a reachable bug.
    """
    from causal_hypergraphs.identification.api import relevant_mechanisms

    graph = MechanismGraph(
        variables={"A", "B"},
        mechanisms={
            "m1": {"inputs": ("A",), "outputs": ("B",)},
            "sink": {"inputs": ("B",), "outputs": ()},
        },
    )

    assert relevant_mechanisms(graph, ("B",)) == frozenset({"m1"})
    assert relevant_mechanisms(graph, ()) == frozenset({"m1"})


def test_the_upstream_cycle_refinement_is_not_safe_yet() -> None:
    """A tempting weakening of the check, and the reason it is not taken.

    A deletion severs the target's own cycle, so on the *post-deletion* graph a cycle
    strictly upstream of the intervention is unreachable and the query looks answerable --
    which would be a valuable carve-out, since feedback upstream of a knockdown is the
    normal case in biology.

    It is unsound as the code stands. `_restrict_to_ancestry` computes its closure on the
    *observational* graph, so it keeps the cyclic factors, and a cyclic product does not
    integrate to one: `sum_{x,y} P(x|y) P(y|x)` runs over `[1, 2]` on random binary joints,
    so the estimand can be wrong by a factor of two. Taking the carve-out needs the
    reduction to drop those factors too, justified by the post-*intervention* law rather
    than by "every factor outside the closure integrates to one" -- the sentence that stops
    being true under cycles.

    This test fails if the check is weakened without that, which is the only way the trap
    gets remembered.
    """
    graph = MechanismGraph(
        variables={"X", "Y", "Z", "W"},
        mechanisms={
            "m1": {"inputs": ("Y",), "outputs": ("X",)},
            "m2": {"inputs": ("X",), "outputs": ("Y",)},   # cycle, strictly upstream
            "m3": {"inputs": ("X",), "outputs": ("Z",)},   # the intervention
            "m4": {"inputs": ("Z",), "outputs": ("W",)},
        },
    )
    result = identify(graph, DeleteMechanism("m3", outcomes={"W"}))

    assert not isinstance(result, Identified), (
        "if this now identifies, check that the emitted estimand does not carry "
        f"P(X | Y) * P(Y | X): {getattr(result, 'expression', None)}"
    )

    # The arithmetic that makes it unsound, so the reason is checked and not just asserted.
    rng = np.random.default_rng(7)
    totals = []
    for _ in range(4000):
        joint = rng.random((2, 2)) ** rng.uniform(0.2, 6.0)
        joint /= joint.sum()
        marginal_x, marginal_y = joint.sum(1), joint.sum(0)
        if min(marginal_x.min(), marginal_y.min()) < 1e-6:
            continue
        x_given_y = joint / marginal_y[None, :]
        y_given_x = joint / marginal_x[:, None]
        totals.append(float((x_given_y * y_given_x).sum()))

    assert max(totals) > 1.5, max(totals)
    assert min(totals) == pytest.approx(1.0, abs=1e-3)


def test_an_answer_on_a_cyclic_graph_declares_what_it_rests_on() -> None:
    """C1 is no longer true of the graph, so claiming it would be false.

    Two things replace it. `C1 (local)` says what actually holds -- the query's own closure
    is acyclic, which is all Lemma 1.1 needs. `Solvability` says what the law needs in
    order to exist at all: under C1 the distribution is defined by sampling in topological
    order, which is total, and without C1 there is no such procedure -- the law is the
    pushforward of the noise through the solution of `V = F(V, U)`, which may have none or
    many. The compiler never sees `F`, so it records that rather than checking it, exactly
    as it does for C2.
    """
    graph = MechanismGraph(
        variables={"X", "Y", "Q", "R"},
        mechanisms={
            "m1": {"inputs": ("Y",), "outputs": ("X",)},
            "m2": {"inputs": ("X",), "outputs": ("Y",)},
            "far": {"inputs": ("Q",), "outputs": ("R",)},
        },
    )
    result = identify(graph, DeleteMechanism("far", outcomes={"R"}))
    assert isinstance(result, Identified), result

    codes = {item.code for item in result.assumptions}
    assert "Solvability" in codes
    assert "C1 (local)" in codes
    assert "C1" not in codes, "the graph is cyclic; claiming C1 outright would be false"


def test_an_acyclic_answer_still_declares_plain_c1_and_no_solvability() -> None:
    """The declaration must not drift for the graphs that already worked."""
    graph = MechanismGraph(
        variables={"A", "B"}, mechanisms={"m1": {"inputs": ("A",), "outputs": ("B",)}}
    )
    result = identify(graph, DeleteMechanism("m1", outcomes={"B"}))
    assert isinstance(result, Identified)

    codes = {item.code for item in result.assumptions}
    assert "C1" in codes
    assert "Solvability" not in codes
    assert "C1 (local)" not in codes


def test_the_observational_law_constrains_a_cyclic_deletion_not_at_all() -> None:
    """Stronger than a two-model gap: the identified set is unbounded and both-signed.

    The models sharing an observational law form a curve, not a pair. Fixing `a` pins `b`
    and then both noise variances, so the curve can be walked exactly, and along it the
    post-deletion variance runs to `4.7e8` while the covariance covers both signs.

    This is why the refusal is worded the way it is. It is not that identification here is
    hard, or that this compiler has not implemented it -- there is nothing in `P(V)` to
    identify the answer with.
    """
    vx, vy, cxy = 2.55102041, 2.77551020, 2.24489796  # from (a,b,s1,s2) = (0.6,0.5,1,1)
    variances = []
    covariances = []
    for a in np.linspace(-40.0, 40.0, 80_001):
        denominator = a * cxy - vy
        if abs(denominator) < 1e-9:
            continue
        b = (a * vx - cxy) / denominator
        if abs(1 - a * b) < 1e-9:
            continue
        s1 = (1 - a * b) * (vy - a * cxy)
        s2 = (1 - a * b) * (vx - b * cxy)
        if s1 <= 0 or s2 <= 0:
            continue
        # after delete(m1) with a unit-variance policy: Var(X) = b^2 + s2, Cov(X,Y) = b
        variances.append(b * b + s2)
        covariances.append(b)

    assert len(variances) > 10_000, len(variances)
    assert max(variances) > 1e6, max(variances)
    assert min(covariances) < -1.0 and max(covariances) > 1.0
