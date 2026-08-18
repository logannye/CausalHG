"""Semantics for the estimand AST: give compiled expressions a value, not just a string.

`evaluate` is the reference: it says what an expression *means*, one point at a time, by
enumerating what the expression ranges over. `eliminate` returns the same number by a
cheaper route, and is verified against `evaluate` rather than trusted on its own.
"""
from .discrete import (
    Assignment,
    DiscreteModel,
    MissingKernel,
    Model,
    SemanticsError,
    UndefinedEstimand,
    evaluate,
)
from .elimination import (
    DEFAULT_MAX_ENTRIES,
    EliminationPlan,
    IntractableQuery,
    eliminate,
    plan_elimination,
)

__all__ = [
    "DEFAULT_MAX_ENTRIES",
    "Assignment",
    "DiscreteModel",
    "EliminationPlan",
    "IntractableQuery",
    "MissingKernel",
    "Model",
    "SemanticsError",
    "UndefinedEstimand",
    "eliminate",
    "evaluate",
    "plan_elimination",
]
