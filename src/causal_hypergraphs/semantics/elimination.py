"""Variable elimination: pay the treewidth of a query, not the size of its ancestry.

The marginal reduction shrank an estimand to the ancestral closure of its outcome. That is
exact and it is necessary, but it only changes the exponent: two hops into a sparse
20,000-gene network the closure already holds ~56 variables, and `2**56` assignments are as
unenumerable as `2**20000`. Something has to remove the exponent, not shrink it.

Summation distributes over a product, so a factor that does not mention the variable being
summed leaves the sum:

    sum_{a,b,c} f(a) g(a,b) h(b,c)  =  sum_c ( sum_b h(b,c) ( sum_a f(a) g(a,b) ) )

Computing each inner sum once and *keeping the table* turns the cost from exponential in
the number of summed variables into exponential in the largest **bucket** -- the set of
variables that meet at one elimination step. That maximum, minus one, is the induced width
of the elimination order; for a sparse regulatory chain it is a small constant, and the
query becomes affordable no matter how long the chain is.

Design commitments
------------------
**This is a strategy, not a second semantics.** `evaluate` in `discrete.py` remains the
reference -- it is what the conformance sweep checked against exact interventional laws --
and `eliminate` is verified against *it*. Nothing here may be believed on its own.

**The same kernel cells are read.** Enumeration evaluates every factor at every cell of its
own scope; so does elimination, exactly once each. That is not an incidental property: the
estimation path defines its positivity certificates as "whatever the evaluator touched", so
a strategy that touched a different set would silently change which certificates come due.

**Cost is reported before it is paid, and refused rather than absorbed.** `plan_elimination`
answers "what will this cost?" without evaluating anything, and `eliminate` raises
`IntractableQuery` naming the offending bucket rather than exhausting memory. A tool that
hangs for an hour tells the user less than one that says which variables met and how wide.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any

from causal_hypergraphs.expression import Expression, Product, SumOut

from .discrete import Assignment, Model, SemanticsError, evaluate

DEFAULT_MAX_ENTRIES = 2**20
"""Largest intermediate table `eliminate` will build without being told to.

A safety bound, not a performance target: it exists so that an unaffordable query fails by
name in a second rather than by exhausting memory in an hour. Raise it deliberately when a
wide query is genuinely worth paying for.
"""


class IntractableQuery(SemanticsError):
    """Elimination would need a table larger than the caller allowed.

    Carries `bucket` -- the variables that had to be tabulated together -- because the
    actionable response is structural, not a matter of waiting longer. Widening the bound
    helps only if the width is nearly affordable; otherwise the answer is to ask about a
    different outcome, or to measure a variable that breaks the bucket apart. The message
    distinguishes the two causes: variables that *met* at an elimination step, which the
    order influences, and a single indivisible factor that is itself that wide, which no
    order can help.
    """

    def __init__(
        self,
        message: str,
        *,
        bucket: Sequence[str],
        entries: int,
        limit: int,
    ) -> None:
        super().__init__(message)
        self.bucket = tuple(bucket)
        self.entries = entries
        self.limit = limit


@dataclass(frozen=True)
class EliminationPlan:
    """What evaluating an estimand by elimination will cost, computed without evaluating it.

    Attributes
    ----------
    order:
        The order the summed variables will be eliminated in.
    buckets:
        The variables multiplied together at each step, aligned with `order`.
    entries:
        The size of the table built at each step, aligned with `order`.
    leaf_entries:
        The size of each factor's own table. A factor is read once per cell of its scope,
        so these are part of the high-water mark even though no step produced them.
    naive_entries:
        Assignments enumeration would visit, per point of the estimand's scope. This is the
        number the reduction alone leaves behind, and the one elimination is here to avoid.
    """

    order: tuple[str, ...]
    buckets: tuple[tuple[str, ...], ...]
    entries: tuple[int, ...]
    leaf_entries: tuple[int, ...]
    naive_entries: int

    @property
    def max_entries(self) -> int:
        """The largest table elimination will materialize, in entries.

        The real cost driver, and the quantity `eliminate`'s bound is compared against.
        """
        return max((*self.entries, *self.leaf_entries, 1))

    @property
    def induced_width(self) -> int:
        """Largest bucket size minus one -- the standard induced-width convention.

        Reported alongside `max_entries` rather than instead of it: width is comparable
        across problems, but entries is what actually has to fit in memory, and the two
        come apart as soon as variables have different domain sizes.
        """
        if not self.buckets:
            return 0
        return max(len(bucket) for bucket in self.buckets) - 1

    def summary(self) -> str:
        if not self.order:
            return "nothing to eliminate: the estimand has no summed variables"
        return (
            f"eliminate {len(self.order)} variable(s) at induced width "
            f"{self.induced_width}; largest table {self.max_entries} entries, against "
            f"{self.naive_entries} assignments for enumeration"
        )


@dataclass(frozen=True)
class _Leaf:
    """One factor of the sum-product, with the variables it is a function of."""

    variables: tuple[str, ...]
    expression: Expression


@dataclass(frozen=True)
class _Table:
    """A materialized factor: a value for every cell of its variables."""

    variables: tuple[str, ...]
    values: Mapping[tuple[Any, ...], float]


def _split(
    expression: Expression, bound: frozenset[str]
) -> tuple[frozenset[str], tuple[_Leaf, ...]]:
    """Peel the sum-product spine into (summed variables, factors).

    Only the spine. A `SumOut` nested *inside* a `Product` is left as one opaque factor:
    lifting its variables into the outer sum would be wrong whenever a sibling factor
    mentions them, since `(sum_v f(v)) * g(v)` is not `sum_v f(v) g(v)`. An opaque factor is
    still a function of its free variables, so it eliminates correctly -- just without the
    saving that flattening would have bought.
    """
    if isinstance(expression, SumOut):
        summed = frozenset(expression.variables)
        clash = sorted(summed & bound)
        if clash:
            raise SemanticsError(
                f"The assignment binds {clash}, which the estimand also sums over. "
                "A bound variable and a summed variable of the same name are two different "
                "quantities, and evaluating them as one would answer a different query."
            )
        inner, leaves = _split(expression.expression, bound)
        return summed | inner, leaves
    if isinstance(expression, Product):
        return frozenset(), tuple(_leaf(factor, bound) for factor in expression.factors)
    return frozenset(), (_leaf(expression, bound),)


def _leaf(expression: Expression, bound: frozenset[str]) -> _Leaf:
    return _Leaf(tuple(sorted(expression.scope() - bound)), expression)


def _sizes(
    names: Iterable[str], domains: Mapping[str, tuple[Any, ...]]
) -> dict[str, int]:
    resolved: dict[str, int] = {}
    for name in names:
        try:
            resolved[name] = len(domains[name])
        except KeyError as exc:
            raise SemanticsError(
                f"The estimand ranges over {name!r}, for which the model supplies no "
                "domain. A summed variable needs a domain even though no caller binds it."
            ) from exc
    return resolved


def _min_fill(summed: frozenset[str], leaves: Sequence[_Leaf]) -> tuple[str, ...]:
    """A min-fill elimination order over the estimand's interaction graph.

    Each factor makes its own variables a clique -- they meet whenever any one of them is
    eliminated -- and the heuristic repeatedly removes the variable whose removal adds the
    fewest new edges. Min-fill is not optimal (finding the optimal order is NP-hard) but it
    is the standard choice and it recovers the obvious orders on the structures that matter
    here: it eliminates a chain end-to-end at width 1.

    Ties break on degree and then on name, so the order is deterministic. Reproducibility
    is not cosmetic: the plan reports a cost, and a cost that varied between runs of the
    same query could not be checked against anything.

    Costs O(n^2 d^2) in the number of summed variables and their degree, so the *ordering*
    becomes the bottleneck long before the elimination does on a very wide closure. Pass an
    explicit `order` when that matters.
    """
    neighbours: dict[str, set[str]] = {name: set() for name in summed}
    for leaf in leaves:
        for name in leaf.variables:
            neighbours[name].update(other for other in leaf.variables if other != name)

    remaining = set(summed)
    order: list[str] = []
    while remaining:
        best_key: tuple[int, int, str] | None = None
        best: str = ""
        for name in sorted(remaining):
            adjacent = sorted(neighbours[name] & remaining)
            fill = sum(
                1
                for first, second in itertools.combinations(adjacent, 2)
                if second not in neighbours[first]
            )
            key = (fill, len(adjacent), name)
            if best_key is None or key < best_key:
                best_key, best = key, name
        adjacent = sorted(neighbours[best] & remaining)
        for first, second in itertools.combinations(adjacent, 2):
            neighbours[first].add(second)
            neighbours[second].add(first)
        remaining.discard(best)
        order.append(best)
    return tuple(order)


def plan_elimination(
    expression: Expression,
    domains: Mapping[str, tuple[Any, ...]],
    *,
    bound: Iterable[str] | None = None,
    order: Sequence[str] | None = None,
) -> EliminationPlan:
    """What eliminating this estimand will cost, without evaluating anything.

    `bound` names the variables an assignment will supply; it defaults to the estimand's
    own scope, which is what a caller evaluating one point of the answer will bind. Every
    other variable in the footprint is summed, and those are what get eliminated.
    """
    resolved_bound = frozenset(expression.scope()) if bound is None else frozenset(bound)
    summed, leaves = _split(expression, resolved_bound)

    free: set[str] = set()
    for leaf in leaves:
        free.update(leaf.variables)
    unbound = sorted(free - summed)
    if unbound:
        raise SemanticsError(
            f"{unbound} are free in the estimand but neither bound by the assignment nor "
            "summed over it. Treating them as summed would answer a different query, and "
            "treating them as bound would need values nobody supplied."
        )

    sizes = _sizes(summed, domains)
    chosen = _min_fill(summed, leaves) if order is None else tuple(str(v) for v in order)
    if set(chosen) != set(summed) or len(set(chosen)) != len(chosen):
        raise SemanticsError(
            f"Elimination order {list(chosen)} is not a permutation of the summed "
            f"variables {sorted(summed)}."
        )

    live = [set(leaf.variables) for leaf in leaves]
    buckets: list[tuple[str, ...]] = []
    entries: list[int] = []
    for variable in chosen:
        bucket: set[str] = set()
        rest: list[set[str]] = []
        for scope in live:
            if variable in scope:
                bucket |= scope
            else:
                rest.append(scope)
        if not bucket:
            # No factor mentions it, so summing it multiplies by the size of its domain.
            buckets.append((variable,))
            entries.append(sizes[variable])
            live = rest
            continue
        buckets.append(tuple(sorted(bucket)))
        entries.append(prod(sizes[name] for name in bucket))
        residue = bucket - {variable}
        if residue:
            rest.append(residue)
        live = rest

    return EliminationPlan(
        order=chosen,
        buckets=tuple(buckets),
        entries=tuple(entries),
        leaf_entries=tuple(
            prod(sizes[name] for name in leaf.variables) for leaf in leaves
        ),
        naive_entries=prod(sizes[name] for name in sorted(summed)) if summed else 1,
    )


def _check(names: Sequence[str], entries: int, limit: int, *, factor: Expression | None) -> None:
    """Refuse a table larger than the caller allowed, and say what would help.

    The remedy differs by cause, so the message does too. A wide *bucket* is a property of
    the elimination order and of how the mechanisms interlock, and can sometimes be broken
    by asking about a different outcome. A wide *single factor* cannot: an indivisible
    kernel over that many variables -- which is what the hidden-variable identifier's
    quotient is -- has no order that helps, and saying "these variables meet at one step"
    would point at a fix that does not exist.
    """
    if entries <= limit:
        return
    shape = (
        f"The factor {factor} is a single kernel over {len(names)} variable(s) "
        f"{list(names)}; no elimination order splits one kernel, so this is the "
        "identifier's own width and not the order's."
        if factor is not None
        else f"The variables {list(names)} meet at one elimination step, so widening the "
        "bound helps only if the width is nearly affordable; otherwise ask about a "
        "narrower outcome, or measure a variable that splits the bucket."
    )
    raise IntractableQuery(
        f"Elimination needs a table of {entries} entries ({len(names)} variable(s), "
        f"induced width {max(len(names) - 1, 0)}), above the {limit}-entry bound. {shape}",
        bucket=names,
        entries=entries,
        limit=limit,
    )


def _leaf_value(expression: Expression, model: Model, assignment: Assignment) -> float:
    if isinstance(expression, SumOut):
        # An opaque nested sum: eliminate it too rather than enumerating it.
        return eliminate(expression, model, assignment)
    return evaluate(expression, model, assignment)


def eliminate(
    expression: Expression,
    model: Model,
    assignment: Assignment,
    *,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    order: Sequence[str] | None = None,
) -> float:
    """Evaluate an estimand by variable elimination.

    Returns exactly what `evaluate` returns -- the same number, by a cheaper route. Use it
    whenever the estimand sums over more than a handful of variables; `evaluate` remains
    the reference implementation and the thing to check against.

    Raises `IntractableQuery` if any intermediate table would exceed `max_entries`, and
    `UndefinedEstimand` at the same strata `evaluate` would, since the same kernel cells
    are read.
    """
    bound = frozenset(assignment)
    plan = plan_elimination(expression, model.domains, bound=bound, order=order)
    _, leaves = _split(expression, bound)
    domains = model.domains

    # Every size below is measured here rather than read back off the plan. The plan is a
    # *prediction*; if it and the evaluator drew their numbers from the same place, a plan
    # that reported the wrong cost would still agree with what was actually built, and the
    # test that pins the two together would be checking nothing.
    factors: list[_Table] = []
    for leaf in leaves:
        size = prod(len(domains[name]) for name in leaf.variables)
        _check(leaf.variables, size, max_entries, factor=leaf.expression)
        values: dict[tuple[Any, ...], float] = {}
        for combination in itertools.product(*(domains[name] for name in leaf.variables)):
            extended = dict(assignment)
            extended.update(zip(leaf.variables, combination, strict=True))
            values[combination] = _leaf_value(leaf.expression, model, extended)
        factors.append(_Table(leaf.variables, values))

    scalar = 1.0
    for variable in plan.order:
        bucket = [factor for factor in factors if variable in factor.variables]
        factors = [factor for factor in factors if variable not in factor.variables]
        if not bucket:
            scalar *= len(domains[variable])
            continue

        merged: set[str] = set()
        for factor in bucket:
            merged.update(factor.variables)
        _check(
            sorted(merged),
            prod(len(domains[name]) for name in merged),
            max_entries,
            factor=None,
        )

        remaining = tuple(name for name in sorted(merged) if name != variable)
        table: dict[tuple[Any, ...], float] = {}
        for combination in itertools.product(*(domains[name] for name in remaining)):
            binding = dict(zip(remaining, combination, strict=True))
            total = 0.0
            for value in domains[variable]:
                binding[variable] = value
                term = 1.0
                for factor in bucket:
                    term *= factor.values[tuple(binding[name] for name in factor.variables)]
                total += term
            table[combination] = total
        factors.append(_Table(remaining, table))

    result = scalar
    for factor in factors:
        result *= factor.values[()]
    return result
