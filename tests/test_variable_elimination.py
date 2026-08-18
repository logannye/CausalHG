"""The cost of a query should be its treewidth, not the size of its ancestry.

The marginal reduction cut a query down to the ancestral closure of its outcome, which
is exact and necessary but not sufficient: two hops into a sparse 20,000-gene network the
closure is already ~56 variables, and enumerating `2**56` assignments is no more possible
than enumerating `2**20000`. The reduction changed the exponent; it did not remove one.

Variable elimination does. Summation distributes over a product, so a factor that does not
mention the variable being summed can be pulled outside the sum:

    sum_{a,b,c} f(a) g(a,b) h(b,c)  =  sum_c ( sum_b h(b,c) ( sum_a f(a) g(a,b) ) )

Each inner sum is computed once and reused, and the largest object ever built is a table
over one *bucket* -- the variables that appear together at a single elimination step --
rather than over the whole ancestry. The cost becomes exponential in the induced width of
the elimination order, which for a sparse regulatory chain is a small constant.

Three things have to be true, and each is tested separately because passing one says
nothing about the others:

1. **It computes the same number.** Elimination is an evaluation strategy, not a new
   semantics. `evaluate` remains the reference -- it is what the conformance sweep
   verified -- and `eliminate` is checked against it, never the other way round.

2. **It is actually cheaper.** Measured by counting kernel lookups, with the same gate
   applied to the enumerating evaluator so that the gate is shown to be capable of
   failing. A cost claim without a firing control is a claim about nothing.

3. **The savings are real, not bookkeeping.** The sixty-link chain is the un-fakeable
   check: enumeration there is `2**60` assignments, so an implementation that secretly
   enumerated would not finish, and no assertion has to be trusted for that to be true.
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from causal_hypergraphs import (
    DeleteMechanism,
    Identified,
    MechanismGraph,
    ReplaceMechanism,
    identify,
    identify_expectation,
)
from causal_hypergraphs.semantics import (
    DiscreteModel,
    IntractableQuery,
    SemanticsError,
    UndefinedEstimand,
    eliminate,
    evaluate,
    plan_elimination,
)

BINARY = (0, 1)


# --- a chain, wide enough that enumeration is not an option -------------------------


def _chain(length: int) -> MechanismGraph:
    """v0 -> m0 -> v1 -> m1 -> v2 -> ... a pure chain of `length` mechanisms."""
    return MechanismGraph(
        variables={f"v{i}" for i in range(length + 1)},
        mechanisms={
            f"m{i}": {"inputs": (f"v{i}",), "outputs": (f"v{i + 1}",)}
            for i in range(length)
        },
    )


class _ChainModel:
    """Analytic kernels for a chain: one prior, one transition matrix, one policy.

    A `DiscreteModel` cannot be built for a sixty-link chain -- its joint would have
    `2**61` entries -- so the model is given as functions instead. That is exactly what
    the `Model` protocol is for, and it keeps this test independent of the estimation
    package's own factored model.
    """

    PRIOR = (0.4, 0.6)
    TRANSITION = ((0.99, 0.01), (0.02, 0.98))
    """Deliberately slow-mixing.

    A chain with an ordinary transition matrix has forgotten where it started long before
    the sixtieth link: the answer converges to the stationary law, and would then agree
    with an implementation that dropped the deletion policy altogether. At a second
    eigenvalue of 0.97 the policy is still worth ~16% of the answer after 59 steps, so the
    fixture can tell a correct computation from a plausible one. See
    `test_the_sixty_link_answer_still_depends_on_the_intervention`.
    """
    POLICY = (0.45, 0.55)

    def __init__(self, length: int, policy: tuple[float, float] | None = None) -> None:
        self.length = length
        self.policy = policy if policy is not None else self.POLICY

    @property
    def domains(self) -> Mapping[str, tuple[Any, ...]]:
        return {f"v{i}": BINARY for i in range(self.length + 1)}

    def conditional(
        self, variables: Sequence[str], given: Sequence[str], assignment: Mapping[str, Any]
    ) -> float:
        (variable,) = variables
        if not given:
            return self.PRIOR[assignment[variable]]
        (parent,) = given
        return self.TRANSITION[assignment[parent]][assignment[variable]]

    def fallback(
        self, mechanism: str, variables: Sequence[str], assignment: Mapping[str, Any]
    ) -> float:
        (variable,) = variables
        return self.policy[assignment[variable]]

    def conditional_expectation(
        self, target: str, given: Sequence[str], assignment: Mapping[str, Any]
    ) -> float:  # pragma: no cover - not reached by these tests
        raise NotImplementedError

    def replacement(
        self,
        mechanism: str,
        variables: Sequence[str],
        given: Sequence[str],
        assignment: Mapping[str, Any],
    ) -> float:  # pragma: no cover - not reached by these tests
        raise NotImplementedError


class _CountingModel:
    """Wraps a model and counts every kernel lookup an evaluator performs.

    Lookups, not seconds: a wall-clock gate would measure the machine as much as the
    algorithm, and would not say *why* one path is cheaper.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0

    @property
    def domains(self) -> Mapping[str, tuple[Any, ...]]:
        return self._inner.domains

    def conditional(
        self, variables: Sequence[str], given: Sequence[str], assignment: Mapping[str, Any]
    ) -> float:
        self.calls += 1
        return self._inner.conditional(variables, given, assignment)

    def fallback(
        self, mechanism: str, variables: Sequence[str], assignment: Mapping[str, Any]
    ) -> float:
        self.calls += 1
        return self._inner.fallback(mechanism, variables, assignment)

    def conditional_expectation(
        self, target: str, given: Sequence[str], assignment: Mapping[str, Any]
    ) -> float:
        self.calls += 1
        return self._inner.conditional_expectation(target, given, assignment)

    def replacement(
        self,
        mechanism: str,
        variables: Sequence[str],
        given: Sequence[str],
        assignment: Mapping[str, Any],
    ) -> float:
        self.calls += 1
        return self._inner.replacement(mechanism, variables, given, assignment)


def _chain_estimand(length: int) -> Identified:
    result = identify(_chain(length), DeleteMechanism("m0", outcomes={f"v{length}"}))
    assert isinstance(result, Identified), result
    return result


def _chain_truth(length: int, policy: tuple[float, float] | None = None) -> tuple[float, float]:
    """`P(v_length | delete(m0))` by propagating the policy along the chain by hand.

    The estimand is `sum P(v0) * P0_m0(v1) * P(v2|v1) * ... * P(v_L|v_{L-1})`; `P(v0)`
    sums to one and drops, leaving a row vector times `L-1` transition matrices. Computed
    here with plain arithmetic so the reference does not share code with the thing under
    test.
    """
    distribution = list(policy if policy is not None else _ChainModel.POLICY)
    for _ in range(length - 1):
        distribution = [
            sum(
                distribution[before] * _ChainModel.TRANSITION[before][after]
                for before in (0, 1)
            )
            for after in (0, 1)
        ]
    return distribution[0], distribution[1]


# --- 1. it computes the same number -------------------------------------------------


def test_elimination_agrees_with_enumeration_across_generated_models() -> None:
    """The differential test: `eliminate` must reproduce `evaluate`, point for point.

    `evaluate` is the verified reference -- the conformance sweep checked it against
    exact interventional laws -- so agreement with it is what makes elimination a
    strategy rather than a second, unverified semantics.
    """
    from tests.conformance.generation import generate_model

    checked = 0
    for seed in range(30):
        model = generate_model(seed, allow_hidden=False, shapes=("positive",))
        graph = model.graph()
        discrete = DiscreteModel(
            domains=model.domains, joint=model.joint(), fallbacks=dict(model.fallbacks)
        )
        for spec in model.mechanisms:
            for outcome in model.variables:
                result = identify(graph, DeleteMechanism(spec.name, outcomes={outcome}))
                assert isinstance(result, Identified)
                for value in BINARY:
                    assignment = {outcome: value}
                    reference = evaluate(result.expression, discrete, assignment)
                    got = eliminate(result.expression, discrete, assignment)
                    assert got == pytest.approx(reference, abs=1e-12), (
                        f"seed {seed} / delete({spec.name}) / {outcome}={value}"
                    )
                    checked += 1

    assert checked > 500, f"only {checked} point(s) checked"


def test_elimination_agrees_on_replacement_queries() -> None:
    """A shape `delete` never produces: an installed factor that conditions on its inputs.

    A deletion policy is a bare joint over the orphaned outputs, so every sweep above sees
    the intervention factor as a leaf with no conditioning. A replacement keeps the
    incidence, so its factor joins the interaction graph differently -- and an
    implementation that special-cased the intervention factor would pass every test so far.
    """
    from tests.conformance.generation import generate_model

    checked = 0
    for seed in range(20):
        model = generate_model(seed, allow_hidden=False, shapes=("positive",))
        graph = model.graph()
        for spec in model.mechanisms:
            discrete = DiscreteModel(
                domains=model.domains,
                joint=model.joint(),
                replacements={f"{spec.name}_prime": model.replacement_table(spec.name)},
            )
            for outcome in model.variables:
                result = identify(
                    graph, ReplaceMechanism(spec.name, f"{spec.name}_prime", {outcome})
                )
                assert isinstance(result, Identified)
                for value in BINARY:
                    assignment = {outcome: value}
                    assert eliminate(result.expression, discrete, assignment) == pytest.approx(
                        evaluate(result.expression, discrete, assignment), abs=1e-12
                    ), f"seed {seed} / replace({spec.name}) / {outcome}={value}"
                    checked += 1

    assert checked > 300, f"only {checked} replacement point(s) checked"


def test_elimination_agrees_on_full_joint_queries_with_nothing_to_eliminate() -> None:
    """The degenerate case: no sum at all. It must not be special-cased into a wrong answer."""
    from tests.conformance.generation import generate_model

    for seed in range(10):
        model = generate_model(seed, allow_hidden=False, shapes=("positive",))
        graph = model.graph()
        discrete = DiscreteModel(
            domains=model.domains, joint=model.joint(), fallbacks=dict(model.fallbacks)
        )
        result = identify(graph, DeleteMechanism(model.mechanisms[0].name))
        assert isinstance(result, Identified)
        for combination in itertools.product(*(BINARY for _ in model.variables)):
            assignment = dict(zip(model.variables, combination, strict=True))
            assert eliminate(result.expression, discrete, assignment) == pytest.approx(
                evaluate(result.expression, discrete, assignment), abs=1e-12
            )


def test_elimination_agrees_on_expectation_estimands() -> None:
    """`E[Y | do]` carries a factor that is not a probability; it must eliminate like one."""
    from tests.conformance.generation import generate_model

    checked = 0
    for seed in range(20):
        model = generate_model(seed, allow_hidden=False, shapes=("positive",))
        graph = model.graph()
        discrete = DiscreteModel(
            domains=model.domains, joint=model.joint(), fallbacks=dict(model.fallbacks)
        )
        for spec in model.mechanisms:
            for outcome in model.variables:
                if outcome in spec.outputs:
                    continue
                result = identify_expectation(graph, DeleteMechanism(spec.name), outcome)
                assert isinstance(result, Identified)
                assert eliminate(result.expression, discrete, {}) == pytest.approx(
                    evaluate(result.expression, discrete, {}), abs=1e-12
                ), f"seed {seed} / delete({spec.name}) / E[{outcome}]"
                checked += 1

    assert checked > 100, f"only {checked} expectation(s) checked"


def test_elimination_agrees_on_the_hidden_variable_quotient_form() -> None:
    """A quotient does not factor, so it eliminates as one opaque factor -- correctly.

    There is no saving here and the implementation must not pretend otherwise, but a
    wrong answer would be worse than an expensive one.
    """
    from tests.conformance.generation import generate_model

    checked = 0
    for seed in range(40):
        model = generate_model(seed, shapes=("positive",))
        if set(model.observed) == set(model.variables):
            continue
        graph = model.graph()
        discrete = DiscreteModel(
            domains=model.domains, joint=model.joint(), fallbacks=dict(model.fallbacks)
        )
        for spec in model.mechanisms:
            for outcome in model.observed:
                result = identify(graph, DeleteMechanism(spec.name, outcomes={outcome}))
                if not isinstance(result, Identified):
                    continue
                for value in BINARY:
                    assignment = {outcome: value}
                    try:
                        reference = evaluate(result.expression, discrete, assignment)
                    except UndefinedEstimand:
                        continue
                    assert eliminate(result.expression, discrete, assignment) == pytest.approx(
                        reference, abs=1e-12
                    )
                    checked += 1

    assert checked > 20, f"only {checked} hidden-variable point(s) checked"


# --- 2. it is actually cheaper ------------------------------------------------------


def _kernel_calls(evaluator: Any, length: int) -> int:
    counted = _CountingModel(_ChainModel(length))
    evaluator(_chain_estimand(length).expression, counted, {f"v{length}": 1})
    return counted.calls


def test_elimination_reads_far_fewer_kernels_than_enumeration() -> None:
    """A twelve-link chain: enumeration reads tens of thousands of cells, elimination dozens."""
    eliminated = _kernel_calls(eliminate, 12)
    enumerated = _kernel_calls(evaluate, 12)

    assert eliminated * 100 < enumerated, (eliminated, enumerated)
    # Every leaf factor is read once per cell of its own scope and never again: the prior
    # on v0 (2 cells), the deletion policy on v1 (2 cells), ten interior transitions
    # (4 cells each), and the last transition at only 2 cells because v12 is bound.
    assert eliminated == 2 + 2 + 4 * 10 + 2


def test_the_cost_gate_fires_against_an_evaluator_that_enumerates() -> None:
    """The control. A gate a wrong computation passes is not a gate.

    The property that matters is not "elimination is cheap on this instance" but "its cost
    does not grow exponentially in the length of the chain". So the same measurement is
    applied to the enumerating evaluator, and it must *fail* -- otherwise the gate is
    measuring the fixture rather than the algorithm.
    """

    def growth(evaluator: Any) -> float:
        return _kernel_calls(evaluator, 14) / _kernel_calls(evaluator, 10)

    # Four more links: enumeration pays 2**4 times over, elimination pays four more factors.
    assert growth(eliminate) < 2.0
    assert growth(evaluate) > 10.0


# --- 3. the savings are real --------------------------------------------------------


def test_a_sixty_link_chain_is_answerable_and_enumeration_is_not() -> None:
    """The un-fakeable check: `2**60` assignments would not finish, and this does.

    No assertion in this test has to be trusted for that to be true. If elimination were
    secretly enumerating, the test would never report at all.
    """
    result = _chain_estimand(60)
    model = _ChainModel(60)
    plan = plan_elimination(result.expression, model.domains)

    assert plan.naive_entries == 2**60
    assert plan.max_entries == 4  # a chain eliminates two variables at a time
    assert plan.induced_width == 1

    low, high = _chain_truth(60)
    assert eliminate(result.expression, model, {"v60": 0}) == pytest.approx(low, abs=1e-12)
    assert eliminate(result.expression, model, {"v60": 1}) == pytest.approx(high, abs=1e-12)


def test_the_sixty_link_answer_still_depends_on_the_intervention() -> None:
    """The control for the test above: it must be able to tell the policies apart.

    A chain that has mixed by its sixtieth link returns its stationary law whatever the
    intervention installed, and a correctness check against that number would pass for an
    implementation that never read the deletion policy at all. So the fixture is checked
    for discriminating power directly: two policies, two different answers, and the
    hand-propagated reference tracks both.
    """
    result = _chain_estimand(60)
    expression = result.expression

    off = _ChainModel(60, policy=(0.05, 0.95))
    on = _ChainModel(60, policy=(0.95, 0.05))
    answer_off = eliminate(expression, off, {"v60": 0})
    answer_on = eliminate(expression, on, {"v60": 0})

    assert abs(answer_on - answer_off) > 0.1, (answer_off, answer_on)
    assert answer_off == pytest.approx(_chain_truth(60, (0.05, 0.95))[0], abs=1e-12)
    assert answer_on == pytest.approx(_chain_truth(60, (0.95, 0.05))[0], abs=1e-12)


def test_the_plan_reports_the_cost_before_anything_is_paid() -> None:
    """Knowing a query is unaffordable is worth more than discovering it after an hour."""
    result = _chain_estimand(3)
    plan = plan_elimination(result.expression, _ChainModel(3).domains)

    assert plan.order == ("v0", "v1", "v2")
    assert plan.naive_entries == 2**3
    assert plan.max_entries == 4
    assert plan.induced_width == 1

    summary = plan.summary()
    assert "width 1" in summary
    assert "4" in summary and "8" in summary, summary


def test_the_reported_largest_table_is_the_largest_table_actually_built() -> None:
    """A plan nobody honours is decoration. Pin the number from both sides.

    The bound is stated as a refusal threshold, so setting it to what the plan promised
    must succeed and setting it one entry lower must not. That pins `max_entries` to the
    real high-water mark without this test having to predict the number.

    The fixture is chosen so the high-water mark comes from a *merge* rather than from one
    wide factor -- three regulators of a common readout, each driven by a common upstream
    node, so eliminating a regulator pulls the readout's factor and the driver together.
    On a graph where the widest bucket merely ties the widest single factor, this test
    would pass against a plan whose bucket arithmetic was wrong, because the leaf would
    set the maximum either way. `assert max_entries > max(leaf_entries)` is what keeps
    that from being an accident of the fixture.
    """
    graph = MechanismGraph(
        variables={"w", "z", "x1", "x2", "x3", "y"},
        mechanisms={
            "root": {"inputs": ("w",), "outputs": ("z",)},
            "hub": {"inputs": ("x1", "x2", "x3"), "outputs": ("y",)},
            **{f"drive{i}": {"inputs": ("z",), "outputs": (f"x{i}",)} for i in (1, 2, 3)},
        },
    )
    result = identify(graph, DeleteMechanism("root", outcomes={"y"}))
    assert isinstance(result, Identified)
    domains = {name: BINARY for name in graph.variable_set}
    plan = plan_elimination(result.expression, domains)

    assert plan.max_entries < plan.naive_entries
    assert plan.max_entries > max(plan.leaf_entries), plan

    model = _fan_in_model(domains)
    eliminate(result.expression, model, {"y": 1}, max_entries=plan.max_entries)
    with pytest.raises(IntractableQuery) as raised:
        eliminate(result.expression, model, {"y": 1}, max_entries=plan.max_entries - 1)
    assert "width" in str(raised.value).lower()
    assert raised.value.entries == plan.max_entries


def _fan_in_model(domains: Mapping[str, tuple[Any, ...]]) -> Any:
    """A model for the fan-in graph whose kernels are constants: only the cost is at issue."""

    class _Flat:
        @property
        def domains(self) -> Mapping[str, tuple[Any, ...]]:
            return domains

        def conditional(
            self, variables: Sequence[str], given: Sequence[str], assignment: Mapping[str, Any]
        ) -> float:
            return 0.5

        def fallback(
            self, mechanism: str, variables: Sequence[str], assignment: Mapping[str, Any]
        ) -> float:
            return 0.5

        def conditional_expectation(
            self, target: str, given: Sequence[str], assignment: Mapping[str, Any]
        ) -> float:  # pragma: no cover - not reached
            raise NotImplementedError

        def replacement(
            self,
            mechanism: str,
            variables: Sequence[str],
            given: Sequence[str],
            assignment: Mapping[str, Any],
        ) -> float:  # pragma: no cover - not reached
            raise NotImplementedError

    return _Flat()


def test_the_answer_does_not_depend_on_the_elimination_order() -> None:
    """Order changes the cost and nothing else. If it changes the answer, a bucket is wrong.

    The strongest internal check available: min-fill is a heuristic, so an implementation
    that merged buckets incorrectly could still look right under the one order it happens
    to pick. Every permutation has to give the same number.
    """
    from tests.conformance.generation import generate_model

    checked = 0
    for seed in range(10):
        model = generate_model(seed, allow_hidden=False, shapes=("positive",))
        graph = model.graph()
        discrete = DiscreteModel(
            domains=model.domains, joint=model.joint(), fallbacks=dict(model.fallbacks)
        )
        for spec in model.mechanisms:
            for outcome in model.variables:
                result = identify(graph, DeleteMechanism(spec.name, outcomes={outcome}))
                assert isinstance(result, Identified)
                summed = sorted(result.expression.footprint() - {outcome})
                if len(summed) < 2:
                    continue
                default = eliminate(result.expression, discrete, {outcome: 1})
                for order in itertools.permutations(summed):
                    assert eliminate(
                        result.expression, discrete, {outcome: 1}, order=order
                    ) == pytest.approx(default, abs=1e-12), (seed, spec.name, outcome, order)
                checked += 1

    assert checked > 20, f"only {checked} estimand(s) permuted"


def test_summing_a_variable_no_factor_mentions_multiplies_by_its_domain() -> None:
    """`sum_q 1` is the size of q's domain, not one.

    Unreachable from today's compiler -- every variable in an ancestral closure appears in
    some retained factor -- but `eliminate` is public and takes any expression, and a
    bucket with no factors in it is precisely where an implementation would quietly return
    the wrong constant. Checked against `evaluate`, which enumerates and so gets it right
    by construction.
    """
    from causal_hypergraphs import Probability, Product, SumOut

    expression = SumOut(("q",), Product([Probability(("x",))]))
    model = DiscreteModel(
        domains={"q": (0, 1, 2), "x": BINARY},
        joint={(q, x): (0.3 if x == 0 else 0.7) / 3 for q in (0, 1, 2) for x in BINARY},
    )
    model.validate()

    reference = evaluate(expression, model, {"x": 0})
    assert reference == pytest.approx(3 * 0.3, abs=1e-12)
    assert eliminate(expression, model, {"x": 0}) == pytest.approx(reference, abs=1e-12)


# --- 4. through the estimator, against real rows ------------------------------------
#
# Elimination on its own does not make a wide query affordable, because the model has to
# supply the kernels and the obvious way to build an empirical model is to tabulate a joint
# over everything the estimand mentions -- which is the same exponential, moved one layer
# down. So the estimation path counts each factor over its own variables and never
# assembles a joint at all. These tests are what make the claim true of the artifact rather
# than of one module.


def _chain_rows(length: int, count: int, seed: int) -> list[dict[str, int]]:
    import random as _random

    rng = _random.Random(seed)
    rows: list[dict[str, int]] = []
    for _ in range(count):
        row = {"v0": 1 if rng.random() < 0.4 else 0}
        for index in range(length):
            previous = row[f"v{index}"]
            stay = 0.75 if previous == 0 else 0.65
            row[f"v{index + 1}"] = previous if rng.random() < stay else 1 - previous
        rows.append(row)
    return rows


def _empirical_chain_truth(
    rows: list[dict[str, int]], length: int, policy: Mapping[tuple[int], float]
) -> tuple[float, float]:
    """`P(v_length | delete(m0))` propagated through the empirical transitions by hand."""
    distribution = [policy[(0,)], policy[(1,)]]
    for index in range(1, length):
        transition = []
        for before in BINARY:
            matching = [row for row in rows if row[f"v{index}"] == before]
            transition.append(
                [
                    sum(1 for row in matching if row[f"v{index + 1}"] == after) / len(matching)
                    for after in BINARY
                ]
            )
        distribution = [
            sum(distribution[before] * transition[before][after] for before in BINARY)
            for after in BINARY
        ]
    return distribution[0], distribution[1]


def test_the_estimator_answers_a_query_no_joint_table_could_hold() -> None:
    """Forty links, three thousand rows, `2**40` assignments never visited.

    The artifact-level check, and it is un-fakeable for the same reason as the sixty-link
    one: an implementation that tabulated the joint over the estimand's footprint, or that
    enumerated it, would not return.
    """
    from causal_hypergraphs.estimation import Dataset, estimate

    length = 40
    rows = _chain_rows(length, 3_000, seed=11)
    data = Dataset.from_records(rows)
    policy = {(0,): 0.3, (1,): 0.7}

    result = identify(_chain(length), DeleteMechanism("m0", outcomes={f"v{length}"}))
    assert isinstance(result, Identified)
    est = estimate(result, data, fallbacks={"m0": policy})

    assert est.plan is not None
    assert est.plan.naive_entries == 2**length
    assert est.plan.max_entries <= 4

    low, high = _empirical_chain_truth(rows, length, policy)
    assert est.values[(0,)] == pytest.approx(low, abs=1e-9)
    assert est.values[(1,)] == pytest.approx(high, abs=1e-9)
    assert est.support.holds


def test_the_two_evaluation_strategies_agree_through_the_estimator() -> None:
    """Same estimate, same certificates, either way. The strategy is not part of the answer."""
    import random as _random

    from causal_hypergraphs.estimation import Dataset, estimate
    from tests.conformance.generation import generate_model

    compared = 0
    for seed in range(8):
        model = generate_model(seed, allow_hidden=False, shapes=("positive",))
        graph = model.graph()
        counts = model.sample_counts(model.joint(), 4_000, _random.Random(seed))
        data = Dataset.from_counts(
            counts, model.variables, domains={v: BINARY for v in model.variables}
        )
        for spec in model.mechanisms:
            for outcome in model.variables:
                result = identify(graph, DeleteMechanism(spec.name, outcomes={outcome}))
                assert isinstance(result, Identified)
                kwargs = {"fallbacks": {spec.name: model.fallbacks[spec.name]}}
                fast = estimate(result, data, method="eliminate", **kwargs)  # type: ignore[arg-type]
                slow = estimate(result, data, method="enumerate", **kwargs)  # type: ignore[arg-type]

                assert set(fast.values) == set(slow.values)
                for point, value in slow.values.items():
                    assert fast.values[point] == pytest.approx(value, abs=1e-12)
                assert fast.support.checked == slow.support.checked
                assert fast.support.min_stratum_count == slow.support.min_stratum_count
                compared += 1

    assert compared > 50, f"only {compared} estimate(s) compared"


def test_an_unaffordable_query_is_refused_by_name_rather_than_attempted() -> None:
    """The bound is reachable from the estimator, so a wide query fails in a second."""
    from causal_hypergraphs.estimation import Dataset, estimate

    rows = _chain_rows(6, 400, seed=3)
    data = Dataset.from_records(rows)
    result = identify(_chain(6), DeleteMechanism("m0", outcomes={"v6"}))
    assert isinstance(result, Identified)

    with pytest.raises(IntractableQuery):
        estimate(result, data, fallbacks={"m0": {(0,): 0.5, (1,): 0.5}}, max_entries=2)


def test_the_two_strategies_agree_on_a_continuous_readout_too() -> None:
    """`E[Y | do]` reaches the rows through a different door, so it needs its own check.

    An expectation is served by a group mean over the dataset rather than by a counted
    kernel, and it is the one factor elimination multiplies in without having tallied.
    """
    import random as _random

    from causal_hypergraphs.estimation import Dataset, estimate

    rng = _random.Random(21)
    rows = []
    for _ in range(2_000):
        v0 = 1 if rng.random() < 0.4 else 0
        v1 = v0 if rng.random() < 0.7 else 1 - v0
        v2 = v1 if rng.random() < 0.75 else 1 - v1
        rows.append({"v0": v0, "v1": v1, "v2": v2, "y": rng.gauss(2.0 + 4.0 * v2, 0.5)})
    data = Dataset.from_records(rows, measures=("y",))

    graph = MechanismGraph(
        variables={"v0", "v1", "v2", "y"},
        mechanisms={
            "m0": {"inputs": ("v0",), "outputs": ("v1",)},
            "m1": {"inputs": ("v1",), "outputs": ("v2",)},
            "m2": {"inputs": ("v2",), "outputs": ("y",)},
        },
    )
    result = identify_expectation(graph, DeleteMechanism("m0"), "y")
    assert isinstance(result, Identified)
    policy = {"m0": {(0,): 0.35, (1,): 0.65}}

    fast = estimate(result, data, fallbacks=policy, method="eliminate")
    slow = estimate(result, data, fallbacks=policy, method="enumerate")

    assert fast.values[()] == pytest.approx(slow.values[()], abs=1e-12)
    assert fast.support.min_stratum_count == slow.support.min_stratum_count


def test_an_indivisible_factor_says_so_rather_than_blaming_the_order() -> None:
    """The hidden-variable quotient is one kernel over everything observed.

    No elimination order splits a single factor, so the refusal must not suggest reordering
    or a narrower outcome as the fix. Two causes of the same failure, two different pieces
    of advice, and this is the one the estimator will actually meet on wide data.
    """
    graph = MechanismGraph(
        variables={"a", "b", "c", "h"},
        mechanisms={
            "m1": {"inputs": ("a",), "outputs": ("b",)},
            "m2": {"inputs": ("b", "h"), "outputs": ("c",)},
        },
        observed_variables={"a", "b", "c"},
    )
    result = identify(graph, DeleteMechanism("m1", outcomes={"c"}))
    assert isinstance(result, Identified)

    domains = {name: BINARY for name in ("a", "b", "c", "h")}
    with pytest.raises(IntractableQuery) as raised:
        eliminate(result.expression, _fan_in_model(domains), {"c": 0}, max_entries=3)

    message = str(raised.value)
    assert "single kernel" in message, message
    assert "narrower outcome" not in message, message
    assert set(raised.value.bucket) == {"a", "b"}


def test_the_estimate_reports_what_the_query_cost() -> None:
    """A number with no cost attached invites the next query to be a thousand times worse."""
    from causal_hypergraphs.estimation import Dataset, estimate

    rows = _chain_rows(8, 600, seed=5)
    data = Dataset.from_records(rows)
    result = identify(_chain(8), DeleteMechanism("m0", outcomes={"v8"}))
    assert isinstance(result, Identified)

    est = estimate(result, data, fallbacks={"m0": {(0,): 0.5, (1,): 0.5}})

    assert est.plan is not None
    assert est.plan.naive_entries == 2**8
    assert "width" in est.summary()


# --- 5. where the new frontier actually is ------------------------------------------


def _sparse_grn(genes: int = 20_000, roots: int = 500, fan_in: int = 3, seed: int = 0):
    """A synthetic sparse regulatory network: each gene produced from `fan_in` earlier ones.

    Regulators are drawn from a moving window rather than from the whole genome, which is
    what makes ancestries overlap at depth -- and overlapping ancestries are exactly what
    drives the width up. A network of independent trees would flatter the result.
    """
    import random as _random

    rng = _random.Random(seed)
    names = [f"g{index}" for index in range(genes)]
    mechanisms = {}
    for index in range(roots, genes):
        window = max(0, index - 2_000)
        inputs: set[str] = set()
        while len(inputs) < fan_in:
            inputs.add(names[rng.randrange(window, index)])
        mechanisms[f"m{index}"] = {"inputs": tuple(sorted(inputs)), "outputs": (names[index],)}
    return MechanismGraph(variables=set(names), mechanisms=mechanisms)


def test_a_sparse_gene_network_is_affordable_near_the_intervention_and_not_far_from_it() -> None:
    """Both halves of the honest claim, on a 20,000-gene network.

    Near the intervention, elimination is transformative: the outcome's ancestry passes a
    hundred variables -- `2**100` assignments, which is not a large number so much as an
    impossible one -- while the largest table stays at a few dozen entries.

    Far from it, elimination hits a wall of its own. Around seven hops the ancestries of
    different branches start to overlap, the width climbs past twenty, and the query is
    unaffordable again. That wall is real and this test pins it, because the useful thing
    to know about a tool is where it stops working.
    """
    from causal_hypergraphs.semantics import DEFAULT_MAX_ENTRIES

    graph = _sparse_grn()
    target = "m600"
    children: dict[str, list[str]] = {}
    for name in graph.mechanisms:
        mechanism = graph.get_mechanism(name)
        for parent in mechanism.inputs:
            children.setdefault(parent, []).extend(mechanism.outputs)

    domains = {name: BINARY for name in graph.variable_set}
    frontier = [graph.get_mechanism(target).outputs[0]]
    seen = set(frontier)
    near: list[tuple[int, int]] = []
    far: list[tuple[int, int]] = []

    for hop in range(9):
        readout = frontier[0]
        result = identify(graph, DeleteMechanism(target, outcomes={readout}))
        assert isinstance(result, Identified)
        plan = plan_elimination(result.expression, domains)
        record = (len(result.expression.footprint()), plan.max_entries)
        (near if hop <= 5 else far).append(record)

        nxt = [c for f in frontier for c in children.get(f, []) if c not in seen]
        seen.update(nxt)
        if not nxt:
            break
        frontier = nxt

    widest_near_ancestry = max(footprint for footprint, _ in near)
    largest_near_table = max(entries for _, entries in near)
    assert widest_near_ancestry > 100, near
    assert largest_near_table <= 64, near

    # And the wall. `eliminate` refuses rather than attempting these.
    assert max(entries for _, entries in far) > DEFAULT_MAX_ENTRIES, far


# --- contract -----------------------------------------------------------------------


def test_a_free_variable_the_assignment_does_not_bind_is_named() -> None:
    """Silently treating an unbound free variable as summed would answer a different query."""
    result = _chain_estimand(3)

    with pytest.raises(SemanticsError, match="v3"):
        eliminate(result.expression, _ChainModel(3), {})


def test_an_undefined_kernel_still_raises_with_its_stratum_named() -> None:
    """Positivity comes due the same way on both paths, or the certificate discharge is a lie."""
    graph = MechanismGraph(
        variables={"a", "b", "c"},
        mechanisms={
            "m1": {"inputs": ("a",), "outputs": ("b",)},
            "m2": {"inputs": ("b",), "outputs": ("c",)},
        },
    )
    # b=1 never occurs, so P(c | b=1) is undefined -- and the estimand needs it, because
    # the deletion policy puts mass there.
    joint = dict.fromkeys(itertools.product(BINARY, BINARY, BINARY), 0.0)
    joint[(0, 0, 0)] = 0.5
    joint[(1, 0, 1)] = 0.5
    model = DiscreteModel(
        domains={name: BINARY for name in ("a", "b", "c")},
        joint=joint,
        fallbacks={"m1": {(0,): 0.5, (1,): 0.5}},
    )
    model.validate()
    result = identify(graph, DeleteMechanism("m1", outcomes={"c"}))
    assert isinstance(result, Identified)

    with pytest.raises(UndefinedEstimand) as raised:
        eliminate(result.expression, model, {"c": 0})
    assert raised.value.stratum == {"b": 1}


def test_elimination_touches_the_same_kernel_cells_as_enumeration() -> None:
    """Same cells read means the same positivity certificates come due.

    The discharge in `estimation` is defined as "whatever the evaluator touched", so if
    elimination touched a different set the certificate population would silently change
    with the evaluation strategy. Checked where the estimand is everywhere defined, since
    both paths stop at the first undefined cell and would then stop in different places.
    """
    from tests.conformance.generation import generate_model

    class _Recording(_CountingModel):
        def __init__(self, inner: Any) -> None:
            super().__init__(inner)
            self.cells: set[tuple[str, ...]] = set()

        def conditional(
            self, variables: Sequence[str], given: Sequence[str], assignment: Mapping[str, Any]
        ) -> float:
            names = tuple(sorted(set(variables) | set(given)))
            self.cells.add(
                (f"P({','.join(variables)}|{','.join(given)})",)
                + tuple(f"{n}={assignment[n]}" for n in names)
            )
            return super().conditional(variables, given, assignment)

    compared = 0
    for seed in range(15):
        model = generate_model(seed, allow_hidden=False, shapes=("positive",))
        graph = model.graph()
        discrete = DiscreteModel(
            domains=model.domains, joint=model.joint(), fallbacks=dict(model.fallbacks)
        )
        for spec in model.mechanisms:
            for outcome in model.variables:
                result = identify(graph, DeleteMechanism(spec.name, outcomes={outcome}))
                assert isinstance(result, Identified)
                enumerated = _Recording(discrete)
                eliminated = _Recording(discrete)
                evaluate(result.expression, enumerated, {outcome: 0})
                eliminate(result.expression, eliminated, {outcome: 0})
                assert eliminated.cells == enumerated.cells, (
                    f"seed {seed} / delete({spec.name}) / {outcome}"
                )
                compared += 1

    assert compared > 100, f"only {compared} estimand(s) compared"
