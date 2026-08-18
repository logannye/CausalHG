"""The Shpitser-Pearl ID algorithm over Pearl ADMGs.

Seven mutually recursive lines (Shpitser & Pearl 2006, Fig. 3), sound and complete for
`P(y | do(x))` in a semi-Markovian model. Where it returns a formula, that formula is
correct; where it fails it exhibits a **hedge**, which is a structural obstruction rather
than a limitation of the search.

Two commitments shape the code.

**The output has to be estimable, not merely correct.** Tian's `Q[S]` conditions each
variable on its whole topological prefix, which on any real graph means a conditional on
the entire ancestry -- a stratum no dataset has a row for. Every conditioning set is cut
back to what the graph says it depends on, and the per-variable pieces are folded back into
the joint kernel they came from. On a mechanism hypergraph, whose districts are exactly the
mechanism output sets, that recovers `P(out(m) | in(m))` exactly -- which is how the result
can be checked against the library's own mechanism-level answer.

**A copy is a copy.** Line 2 restricts to the ancestors of the outcome, which marginalizes
away any do-variable that is not one. That variable is bound outside, so summing over the
same name would capture it. Such variables are renamed and the renaming is returned with
the result, where `semantics.with_aliases` gives it a meaning.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from causal_hypergraphs.expression import (
    Expression,
    Probability,
    Product,
    Quotient,
    SumOut,
)

from .pearl_id import ADMG

LINES = ("1", "2", "3", "4", "5", "6", "7")
"""The seven lines of Fig. 3. Named so a test can assert every one is exercised."""

_fired: list[set[str]] = []


@contextlib.contextmanager
def line_coverage() -> Iterator[set[str]]:
    """Record which lines of the recursion fire inside the block.

    A line no test reaches has no gate at all, and seven lines are seven separate pieces
    of reasoning. This is the instrument that lets a suite assert its own coverage of them.
    """
    seen: set[str] = set()
    _fired.append(seen)
    try:
        yield seen
    finally:
        _fired.pop()


def _fire(line: str) -> None:
    for seen in _fired:
        seen.add(line)


@dataclass(frozen=True)
class Hedge:
    """The structural obstruction ID exhibits when an effect is not identifiable.

    `forest` is the C-forest the recursion bottomed out in and `subforest` the c-component
    of the graph with the interventions removed. `F' subsetneq F` rooted at the same set is
    what makes the effect unidentifiable (Shpitser & Pearl 2006, Definition 5).
    """

    forest: tuple[str, ...]
    subforest: tuple[str, ...]
    outcomes: tuple[str, ...]
    interventions: tuple[str, ...]

    def __str__(self) -> str:
        return (
            f"hedge for P({','.join(self.outcomes)} | do({','.join(self.interventions)})): "
            f"F = {list(self.forest)}, F' = {list(self.subforest)}"
        )


class HedgeFound(Exception):
    def __init__(self, hedge: Hedge) -> None:
        super().__init__(str(hedge))
        self.hedge = hedge


# --- the carried distribution -----------------------------------------------------


@dataclass(frozen=True)
class _Law:
    """`sum_{summed} prod(factors)`, read as a distribution over `variables`."""

    variables: frozenset[str]
    summed: tuple[str, ...] = ()
    factors: tuple[Expression, ...] = ()

    def expression(self) -> Expression:
        body: Expression = (
            self.factors[0] if len(self.factors) == 1 else Product(self.factors)
        )
        return SumOut(self.summed, body) if self.summed else body

    def marginal(self, keep: frozenset[str]) -> _Law:
        """Sum this law down to `keep`, cancelling what cancels exactly."""
        drop = set(self.variables) - set(keep)
        factors = list(self.factors)
        summed = list(self.summed)

        progress = True
        while progress and drop:
            progress = False
            for variable in sorted(drop):
                touching = [f for f in factors if variable in f.scope()]
                if len(touching) != 1:
                    continue
                factor = touching[0]
                if not isinstance(factor, Probability) or variable in factor.given:
                    continue
                factors.remove(factor)
                remaining = tuple(n for n in factor.variables if n != variable)
                if remaining:
                    # sum_v P(A, v | B) = P(A | B)
                    factors.append(Probability(remaining, given=factor.given))
                # else sum_v P(v | B) = 1, so the factor leaves entirely
                drop.discard(variable)
                progress = True
                break

        summed.extend(sorted(drop))
        return _Law(
            variables=frozenset(keep),
            summed=tuple(sorted(set(summed))),
            factors=tuple(sorted(factors, key=lambda f: f.render())),
        )

    def conditional(
        self, variable: str, prefix: Sequence[str], graph: ADMG | None = None
    ) -> Expression:
        """`P(variable | prefix)` from this law."""
        given = tuple(sorted(prefix))
        # The common case: the law is one joint kernel, so the conditional is a kernel too
        # rather than a ratio of two sums nobody can read or eliminate.
        if not self.summed and len(self.factors) == 1:
            only = self.factors[0]
            if isinstance(only, Probability) and not only.given:
                if graph is not None:
                    given = _narrow(graph, variable, given)
                return Probability((variable,), given=given)
        numerator = self.marginal(frozenset(given) | {variable})
        denominator = self.marginal(frozenset(given))
        if not given:
            return numerator.expression()
        return Quotient(numerator.expression(), denominator.expression())

    def factorize(
        self, district: frozenset[str], order: Sequence[str], graph: ADMG | None = None
    ) -> tuple[Expression, ...]:
        """The factors of `Q[district]`: one per variable, then folded back together.

        Tian's formula is per-variable, and left that way it produces a conditional on the
        whole topological prefix for each member of the district. `_narrow` cuts each
        conditioning set to what the graph says it depends on and `_fold` recombines the
        pieces into the joint kernel they came from, so a district produced by one
        mechanism comes back as one factor rather than as its chain rule.
        """
        factors: list[Expression] = []
        for position, name in enumerate(order):
            if name in district:
                factors.append(self.conditional(name, order[:position], graph))
        return _fold(factors)


# --- graph helpers ----------------------------------------------------------------


def _fold(factors: Sequence[Expression]) -> tuple[Expression, ...]:
    """Run the chain rule backwards: `P(a | W) * P(b | W, a)` is `P(a, b | W)`.

    Tian's `Q[S]` is written variable by variable, but a district's members are produced
    together and generally share a conditioning set, so the pieces recombine into the joint
    kernel they came from. Exact, and worth doing: `P(C, D | A, B)` is one cell of a
    four-way table where `P(C | A, B) * P(D | A, B, C)` needs a five-way one, and the
    estimator has to find rows in whichever is asked for.

    This is also what makes the output comparable to the mechanism-level answer, where a
    mechanism's outputs are jointly produced and its factor is a single joint by
    construction.
    """
    folded: list[Expression] = []
    for factor in factors:
        previous = folded[-1] if folded else None
        if (
            isinstance(factor, Probability)
            and isinstance(previous, Probability)
            and set(factor.given) == set(previous.given) | set(previous.variables)
        ):
            folded[-1] = Probability(
                previous.variables + factor.variables, given=previous.given
            )
            continue
        folded.append(factor)
    return tuple(folded)


def _narrow(graph: ADMG, variable: str, prefix: Sequence[str]) -> tuple[str, ...]:
    """Drop the conditioning variables `variable` is independent of anyway.

    Tian's `Q[S]` conditions each variable on its whole topological prefix. That is
    correct and it is ruinous: the prefix is the entire ancestry, so a kernel that depends
    on two parents would be written as a conditional on two hundred variables, and no
    dataset has a stratum for that. The graph licenses dropping the ones `variable` is
    m-separated from, which is not an approximation -- `P(v | S, W) = P(v | S)` holds in
    every model of the graph when `v` is separated from `W` by `S`.

    Removal is greedy and checks the whole removed set each time, so the result really does
    satisfy joint independence of the dropped variables, not only the pairwise version.
    Sorted iteration keeps the choice deterministic between runs.

    Applied only where the carried law is the observational joint, since that is where the
    graph's independencies are the law's. Deeper in the recursion the law is a `Q[S]` whose
    independence structure is not read off this graph, and the full prefix is kept.
    """
    kept = list(prefix)
    for candidate in sorted(prefix):
        trial = [name for name in kept if name != candidate]
        removed = [name for name in prefix if name not in trial]
        if all(graph.m_separated(variable, name, trial) for name in removed):
            kept = trial
    return tuple(sorted(kept))


def _cut_incoming(graph: ADMG, nodes: frozenset[str]) -> ADMG:
    """`G` with every edge *into* `nodes` removed, bidirected edges included."""
    return ADMG(
        nodes=graph.nodes,
        directed_edges=[e for e in graph.directed_edges if e[1] not in nodes],
        bidirected_edges=[
            e for e in graph.bidirected_edges if e[0] not in nodes and e[1] not in nodes
        ],
    )


def _districts(graph: ADMG, nodes: frozenset[str]) -> list[frozenset[str]]:
    return [frozenset(d) for d in graph.induced(nodes).districts()]


# --- the recursion ----------------------------------------------------------------


@dataclass
class _Run:
    aliases: dict[str, str] = field(default_factory=dict)
    counter: int = 0

    def copy_name(self, base: str) -> str:
        self.counter += 1
        suffix = "_prime" if self.counter == 1 else f"_prime{self.counter}"
        name = f"{base}{suffix}"
        self.aliases[name] = base
        return name


def _rename(expression: Expression, mapping: Mapping[str, str]) -> Expression:
    """Rewrite variable names throughout an expression."""
    if not mapping:
        return expression
    swap = lambda name: mapping.get(name, name)  # noqa: E731
    if isinstance(expression, Probability):
        return Probability(
            tuple(swap(n) for n in expression.variables),
            given=tuple(swap(n) for n in expression.given),
        )
    if isinstance(expression, Product):
        return Product([_rename(f, mapping) for f in expression.factors])
    if isinstance(expression, SumOut):
        return SumOut(
            tuple(swap(n) for n in expression.variables),
            _rename(expression.expression, mapping),
        )
    if isinstance(expression, Quotient):
        return Quotient(
            _rename(expression.numerator, mapping), _rename(expression.denominator, mapping)
        )
    return expression


def _id(
    outcomes: frozenset[str],
    interventions: frozenset[str],
    law: _Law,
    graph: ADMG,
    run: _Run,
) -> _Law:
    nodes = frozenset(graph.nodes)

    if not interventions:
        _fire("1")
        return law.marginal(outcomes)

    ancestors = frozenset(graph.ancestors(outcomes))
    if nodes - ancestors:
        _fire("2")
        restricted = law.marginal(ancestors)
        # A do-variable that is not an ancestor of the outcome is marginalized here while
        # still bound outside. Same name, two quantities: rename before it captures.
        clash = {
            name: run.copy_name(name)
            for name in sorted(interventions - ancestors)
            if name in restricted.summed
        }
        if clash:
            restricted = _Law(
                variables=restricted.variables,
                summed=tuple(sorted(clash.get(n, n) for n in restricted.summed)),
                factors=tuple(_rename(f, clash) for f in restricted.factors),
            )
        return _id(
            outcomes, interventions & ancestors, restricted, graph.induced(ancestors), run
        )

    without = nodes - interventions
    reachable = frozenset(_cut_incoming(graph, interventions).ancestors(outcomes))
    extra = without - reachable
    if extra:
        _fire("3")
        return _id(outcomes, interventions | extra, law, graph, run)

    components = _districts(graph, without)
    if len(components) > 1:
        _fire("4")
        parts = [_id(part, nodes - part, law, graph, run) for part in components]
        factors: list[Expression] = []
        summed: set[str] = set()
        for part in parts:
            factors.extend(part.factors)
            summed.update(part.summed)
        summed.update(without - outcomes)
        return _Law(
            variables=outcomes,
            summed=tuple(sorted(summed)),
            factors=tuple(sorted(factors, key=lambda f: f.render())),
        )

    (component,) = components
    whole = [frozenset(d) for d in graph.districts()]
    if nodes in whole:
        _fire("5")
        raise HedgeFound(
            Hedge(
                forest=tuple(sorted(nodes)),
                subforest=tuple(sorted(component)),
                outcomes=tuple(sorted(outcomes)),
                interventions=tuple(sorted(interventions)),
            )
        )

    order = graph.topological_order()
    if component in whole:
        _fire("6")
        return _Law(
            variables=outcomes,
            summed=tuple(sorted(component - outcomes)),
            factors=law.factorize(component, order, graph),
        )

    _fire("7")
    (enclosing,) = [d for d in whole if component < d]
    return _id(
        outcomes,
        interventions & enclosing,
        _Law(variables=enclosing, factors=law.factorize(enclosing, order, graph)),
        graph.induced(enclosing),
        run,
    )


def identify_admg_effect(
    graph: ADMG, outcomes: Sequence[str], interventions: Sequence[str]
) -> tuple[Expression, Mapping[str, str]] | Hedge:
    """`P(outcomes | do(interventions))` in `graph`, or the hedge that forbids it."""
    run = _Run()
    law = _Law(variables=frozenset(graph.nodes), factors=(Probability(graph.nodes),))
    try:
        result = _id(frozenset(outcomes), frozenset(interventions), law, graph, run)
    except HedgeFound as found:
        return found.hedge

    expression = result.expression()
    return _close(expression, outcomes, interventions, run), dict(run.aliases)


def _close(
    expression: Expression, outcomes: Sequence[str], interventions: Sequence[str], run: _Run
) -> Expression:
    """Bind the variables line 3 added to the do-set.

    Line 3 replaces `P(y | do(x))` by `P(y | do(x, w))`, which is the same quantity for
    *every* value of `w` -- that is what makes the substitution legal. The expression the
    recursion returns is therefore a function of `w` that does not vary with it, but `w` is
    still a free name, and an estimand nobody can bind is not an answer.

    Averaging it away is exact rather than approximate: `sum_w P(w) f(w) = f` when `f` does
    not depend on `w`. It also happens to be the right thing under estimation error, where
    `f` is only nearly constant and picking one arbitrary value of `w` would throw away the
    rest of the sample.
    """
    expected = frozenset(outcomes) | frozenset(interventions)
    extra = tuple(
        sorted(
            name
            for name in expression.scope() - expected
            if name not in run.aliases
        )
    )
    if not extra:
        return expression
    return SumOut(extra, Product([Probability(extra), expression]))
