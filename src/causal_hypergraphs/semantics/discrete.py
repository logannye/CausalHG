"""Finite-discrete semantics for the estimand AST.

The identification compiler emits expressions; this module gives them *meaning* by
evaluating them against a finite discrete probability model. That turns an estimand
from a string into a function, which is what makes the compiler's claims falsifiable:
an expression can now be checked against an interventional law rather than against a
hand-written expected string.

Design commitments
------------------
Evaluation is **total or loud**. Every AST node is either evaluated or raises a named
error; no node silently returns a default, and no undefined quantity is allowed to
become ``nan`` and propagate. This matters here specifically: the v1 identifiers are
density *quotients*, and a quotient whose denominator vanishes is exactly the case the
theory documents do not treat (``THEOREM_T2_T3.md`` concedes the mechanism factor can
be singular). Silent ``nan`` would let an undefined estimand score as agreement.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import singledispatch
from typing import Any

from causal_hypergraphs.expression import (
    Expression,
    Fallback,
    MechanismFactor,
    Probability,
    Product,
    Quotient,
    ReplacementFactor,
    SumOut,
)


class SemanticsError(Exception):
    """Base class for evaluation failures."""


class UndefinedEstimand(SemanticsError):
    """The estimand is undefined at this point of the sample space.

    Raised when a conditional or an explicit quotient divides by zero. This is a
    statement about the *expression*, not about the evaluator: an identifier that can
    reach this state has not established the positivity it silently assumes.
    """


class MissingKernel(SemanticsError):
    """The model does not supply a primitive the expression references."""


Assignment = Mapping[str, Any]


@dataclass(frozen=True)
class DiscreteModel:
    """A finite discrete probability model an estimand can be evaluated against.

    Parameters
    ----------
    domains:
        Finite domain for each variable. The model's variable order is
        ``tuple(sorted(domains))`` and ``joint`` keys must follow that order.
    joint:
        The observational law ``P(V)``, keyed by value tuples in the model's variable
        order. Need not be normalized for pointwise comparisons, but ``validate()``
        checks normalization because an unnormalized fixture makes a differential test
        vacuous.
    fallbacks:
        Per-variable fallback laws ``P0(v)``, as ``{variable: {value: probability}}``.
    replacements:
        Replacement mechanism kernels, as
        ``{mechanism_name: {(output_values, input_values): probability}}`` where both
        keys are value tuples in sorted-variable order.
    """

    domains: Mapping[str, tuple[Any, ...]]
    joint: Mapping[tuple[Any, ...], float]
    fallbacks: Mapping[str, Mapping[Any, float]] = field(default_factory=dict)
    replacements: Mapping[str, Mapping[tuple[Any, ...], float]] = field(default_factory=dict)

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(sorted(self.domains))

    def validate(self, *, tolerance: float = 1e-12) -> None:
        """Raise if the joint is not a probability distribution over the full domain."""
        expected = 1
        for variable in self.variables:
            expected *= len(self.domains[variable])
        if len(self.joint) != expected:
            raise SemanticsError(
                f"Joint has {len(self.joint)} entries; the domain product is {expected}. "
                "Every assignment must be present, including zero-probability ones."
            )
        if any(value < 0 for value in self.joint.values()):
            raise SemanticsError("Joint contains a negative probability.")
        total = sum(self.joint.values())
        if abs(total - 1.0) > tolerance:
            raise SemanticsError(f"Joint sums to {total!r}, not 1.")

    def marginal(self, variables: Iterable[str], assignment: Assignment) -> float:
        """``P(variables = assignment[variables])``, marginalizing everything else."""
        wanted = tuple(sorted(variables))
        missing = [v for v in wanted if v not in assignment]
        if missing:
            raise SemanticsError(f"Assignment does not bind {missing}.")
        order = self.variables
        index = {name: position for position, name in enumerate(order)}
        target = [(index[v], assignment[v]) for v in wanted]
        total = 0.0
        for key, probability in self.joint.items():
            if all(key[position] == value for position, value in target):
                total += probability
        return total

    def conditional(
        self, variables: Sequence[str], given: Sequence[str], assignment: Assignment
    ) -> float:
        """``P(variables | given)`` evaluated at ``assignment``."""
        numerator = self.marginal(tuple(variables) + tuple(given), assignment)
        if not given:
            return numerator
        denominator = self.marginal(given, assignment)
        if denominator == 0.0:
            raise UndefinedEstimand(
                f"P({','.join(variables)} | {','.join(given)}) is undefined: the conditioning "
                f"event has probability zero at {_restrict(assignment, given)!r}."
            )
        return numerator / denominator

    def fallback(self, variable: str, assignment: Assignment) -> float:
        try:
            table = self.fallbacks[variable]
        except KeyError as exc:
            raise MissingKernel(f"No fallback law P0({variable}) supplied.") from exc
        if variable not in assignment:
            raise SemanticsError(f"Assignment does not bind {variable!r}.")
        return table[assignment[variable]]

    def replacement(
        self, mechanism: str, variables: Sequence[str], given: Sequence[str], assignment: Assignment
    ) -> float:
        try:
            table = self.replacements[mechanism]
        except KeyError as exc:
            raise MissingKernel(f"No replacement kernel supplied for {mechanism!r}.") from exc
        key = (
            tuple(assignment[v] for v in sorted(variables)),
            tuple(assignment[v] for v in sorted(given)),
        )
        try:
            return table[key]
        except KeyError as exc:
            raise MissingKernel(
                f"Replacement kernel {mechanism!r} has no entry for {key!r}."
            ) from exc


def _restrict(assignment: Assignment, variables: Iterable[str]) -> dict[str, Any]:
    return {v: assignment[v] for v in sorted(variables) if v in assignment}


@singledispatch
def evaluate(expression: Expression, model: DiscreteModel, assignment: Assignment) -> float:
    """Evaluate an estimand at one point of the sample space.

    Dispatches on AST node type. An unregistered node type is an error rather than a
    default, so extending the AST cannot silently produce wrong numbers here.
    """
    raise SemanticsError(
        f"No discrete semantics for AST node {type(expression).__name__!r}. "
        "Register an `evaluate` implementation before compiling estimands that use it."
    )


@evaluate.register
def _(expression: Probability, model: DiscreteModel, assignment: Assignment) -> float:
    return model.conditional(expression.variables, expression.given, assignment)


@evaluate.register
def _(expression: Fallback, model: DiscreteModel, assignment: Assignment) -> float:
    return model.fallback(expression.variable, assignment)


@evaluate.register
def _(expression: ReplacementFactor, model: DiscreteModel, assignment: Assignment) -> float:
    return model.replacement(
        expression.mechanism, expression.variables, expression.given, assignment
    )


@evaluate.register
def _(expression: MechanismFactor, model: DiscreteModel, assignment: Assignment) -> float:
    # The observational mechanism factor P(out(m) | in(m)) is identified by the
    # corresponding observational conditional under C2 (THEOREM_T2_T3.md, Lemma 1.1).
    return model.conditional(expression.variables, expression.given, assignment)


@evaluate.register
def _(expression: Product, model: DiscreteModel, assignment: Assignment) -> float:
    total = 1.0
    for factor in expression.factors:
        total *= evaluate(factor, model, assignment)
    return total


@evaluate.register
def _(expression: Quotient, model: DiscreteModel, assignment: Assignment) -> float:
    denominator = evaluate(expression.denominator, model, assignment)
    if denominator == 0.0:
        raise UndefinedEstimand(
            f"Estimand divides by zero: {expression.denominator} vanishes at "
            f"{_restrict(assignment, expression.denominator.scope())!r}."
        )
    return evaluate(expression.numerator, model, assignment) / denominator


@evaluate.register
def _(expression: SumOut, model: DiscreteModel, assignment: Assignment) -> float:
    summed = expression.variables
    domains = [model.domains[v] for v in summed]
    total = 0.0
    for combination in itertools.product(*domains):
        extended = dict(assignment)
        extended.update(dict(zip(summed, combination)))
        total += evaluate(expression.expression, model, extended)
    return total
