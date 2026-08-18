from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _items(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(sorted(str(v) for v in values))


def _join(values: tuple[str, ...]) -> str:
    return ",".join(values)


@dataclass(frozen=True)
class Kernel:
    """Machine-readable primitive distribution referenced by an expression."""

    kind: str
    label: str
    variables: tuple[str, ...]
    given: tuple[str, ...] = ()

    def __init__(
        self,
        kind: str,
        label: str,
        variables: object,
        given: object = (),
    ) -> None:
        object.__setattr__(self, "kind", str(kind))
        object.__setattr__(self, "label", str(label))
        object.__setattr__(self, "variables", _items(variables))
        object.__setattr__(self, "given", _items(given))


class Expression:
    """Base class for probability expression AST nodes."""

    def render(self) -> str:
        return str(self)

    def to_latex(self) -> str:
        return str(self)

    def scope(self) -> frozenset[str]:
        """The *free* variables: those an assignment must bind to evaluate this."""
        return frozenset()

    def footprint(self) -> frozenset[str]:
        """Every variable evaluating this ranges over, bound ones included.

        `scope()` says what a caller must supply; `footprint()` says what evaluation
        costs. They differ exactly at `SumOut`, which binds variables by enumerating
        their domains -- so the footprint, not the scope, is the exponent in the cost of
        a finite-discrete evaluation. A marginal query is worth computing precisely
        because it shrinks this set.
        """
        return self.scope()

    def conditioned_on(self) -> frozenset[str]:
        return frozenset()

    def kernels(self) -> tuple[Kernel, ...]:
        return ()

    def canonical_key(self) -> tuple[Any, ...]:
        return (type(self).__name__, str(self))

    def fingerprint(self) -> str:
        return repr(self.canonical_key())


@dataclass(frozen=True)
class Probability(Expression):
    variables: tuple[str, ...]
    given: tuple[str, ...] = ()

    def __init__(self, variables: object, given: object = ()) -> None:
        object.__setattr__(self, "variables", _items(variables))
        object.__setattr__(self, "given", _items(given))

    def __str__(self) -> str:
        if self.given:
            return f"P({_join(self.variables)} | {_join(self.given)})"
        return f"P({_join(self.variables)})"

    def to_latex(self) -> str:
        if self.given:
            return rf"P({_join(self.variables)} \mid {_join(self.given)})"
        return f"P({_join(self.variables)})"

    def scope(self) -> frozenset[str]:
        return frozenset(self.variables) | frozenset(self.given)

    def conditioned_on(self) -> frozenset[str]:
        return frozenset(self.given)

    def kernels(self) -> tuple[Kernel, ...]:
        return (Kernel("probability", "P", self.variables, self.given),)

    def canonical_key(self) -> tuple[Any, ...]:
        return ("probability", self.variables, self.given)


@dataclass(frozen=True)
class Fallback(Expression):
    """The intervention policy `P0^m(out(m))` installed when mechanism `m` is deleted.

    This is a *joint* kernel over all of `m`'s outputs, deliberately not a product of
    per-variable laws. Deleting a mechanism orphans its outputs simultaneously, and a
    product form would force them independent -- silently ruling out the case where a
    mechanism's outputs stay coupled after the mechanism is removed. A product policy is
    still expressible as a joint kernel that happens to factorize, so nothing is lost.

    There is no `given`: deletion removes the mechanism, so `in(m)` no longer influences
    `out(m)`. A policy that still reads the inputs is a *replacement*, not a deletion, and
    belongs in `ReplacementFactor`.
    """

    mechanism: str
    variables: tuple[str, ...]

    def __init__(self, mechanism: str, variables: object) -> None:
        object.__setattr__(self, "mechanism", str(mechanism))
        object.__setattr__(self, "variables", _items(variables))

    def __str__(self) -> str:
        return f"P0_{self.mechanism}({_join(self.variables)})"

    def to_latex(self) -> str:
        return rf"P_0^{{{self.mechanism}}}({_join(self.variables)})"

    def scope(self) -> frozenset[str]:
        return frozenset(self.variables)

    def kernels(self) -> tuple[Kernel, ...]:
        return (Kernel("fallback", f"P0_{self.mechanism}", self.variables),)

    def canonical_key(self) -> tuple[Any, ...]:
        return ("fallback", self.mechanism, self.variables)


@dataclass(frozen=True)
class MechanismFactor(Expression):
    mechanism: str
    variables: tuple[str, ...]
    given: tuple[str, ...] = ()

    def __init__(self, mechanism: str, variables: object, given: object = ()) -> None:
        object.__setattr__(self, "mechanism", str(mechanism))
        object.__setattr__(self, "variables", _items(variables))
        object.__setattr__(self, "given", _items(given))

    def __str__(self) -> str:
        suffix = f" | {_join(self.given)}" if self.given else ""
        return f"P_{self.mechanism}({_join(self.variables)}{suffix})"

    def to_latex(self) -> str:
        suffix = rf" \mid {_join(self.given)}" if self.given else ""
        return rf"P_{{{self.mechanism}}}({_join(self.variables)}{suffix})"

    def scope(self) -> frozenset[str]:
        return frozenset(self.variables) | frozenset(self.given)

    def conditioned_on(self) -> frozenset[str]:
        return frozenset(self.given)

    def kernels(self) -> tuple[Kernel, ...]:
        return (Kernel("mechanism", self.mechanism, self.variables, self.given),)

    def canonical_key(self) -> tuple[Any, ...]:
        return ("mechanism", self.mechanism, self.variables, self.given)


@dataclass(frozen=True)
class ReplacementFactor(MechanismFactor):
    def kernels(self) -> tuple[Kernel, ...]:
        return (Kernel("replacement", self.mechanism, self.variables, self.given),)

    def canonical_key(self) -> tuple[Any, ...]:
        return ("replacement", self.mechanism, self.variables, self.given)


@dataclass(frozen=True)
class Quotient(Expression):
    numerator: Expression
    denominator: Expression

    def __str__(self) -> str:
        return f"{self.numerator} / {self.denominator}"

    def to_latex(self) -> str:
        return rf"\frac{{{self.numerator.to_latex()}}}{{{self.denominator.to_latex()}}}"

    def scope(self) -> frozenset[str]:
        return self.numerator.scope() | self.denominator.scope()

    def footprint(self) -> frozenset[str]:
        return self.numerator.footprint() | self.denominator.footprint()

    def conditioned_on(self) -> frozenset[str]:
        return self.numerator.conditioned_on() | self.denominator.conditioned_on()

    def kernels(self) -> tuple[Kernel, ...]:
        return self.numerator.kernels() + self.denominator.kernels()

    def canonical_key(self) -> tuple[Any, ...]:
        return ("quotient", self.numerator.canonical_key(), self.denominator.canonical_key())


@dataclass(frozen=True)
class Product(Expression):
    factors: tuple[Expression, ...]

    def __init__(self, factors: object) -> None:
        flattened: list[Expression] = []
        for factor in factors:  # type: ignore[union-attr]
            if isinstance(factor, Product):
                flattened.extend(factor.factors)
            else:
                flattened.append(factor)
        object.__setattr__(self, "factors", tuple(sorted(flattened, key=lambda f: f.render())))

    def __str__(self) -> str:
        if not self.factors:
            return "1"
        return " * ".join(str(factor) for factor in self.factors)

    def to_latex(self) -> str:
        if not self.factors:
            return "1"
        return r" \cdot ".join(factor.to_latex() for factor in self.factors)

    def scope(self) -> frozenset[str]:
        scope: set[str] = set()
        for factor in self.factors:
            scope.update(factor.scope())
        return frozenset(scope)

    def footprint(self) -> frozenset[str]:
        touched: set[str] = set()
        for factor in self.factors:
            touched.update(factor.footprint())
        return frozenset(touched)

    def conditioned_on(self) -> frozenset[str]:
        conditioned: set[str] = set()
        for factor in self.factors:
            conditioned.update(factor.conditioned_on())
        return frozenset(conditioned)

    def kernels(self) -> tuple[Kernel, ...]:
        kernels: list[Kernel] = []
        for factor in self.factors:
            kernels.extend(factor.kernels())
        return tuple(kernels)

    def canonical_key(self) -> tuple[Any, ...]:
        return ("product", tuple(factor.canonical_key() for factor in self.factors))


@dataclass(frozen=True)
class SumOut(Expression):
    variables: tuple[str, ...]
    expression: Expression

    def __init__(self, variables: object, expression: Expression) -> None:
        object.__setattr__(self, "variables", _items(variables))
        object.__setattr__(self, "expression", expression)

    def __str__(self) -> str:
        if not self.variables:
            return str(self.expression)
        return f"sum_{{{_join(self.variables)}}} {self.expression}"

    def to_latex(self) -> str:
        if not self.variables:
            return self.expression.to_latex()
        return rf"\sum_{{{_join(self.variables)}}} {self.expression.to_latex()}"

    def scope(self) -> frozenset[str]:
        return self.expression.scope() - frozenset(self.variables)

    def footprint(self) -> frozenset[str]:
        # The summed variables are bound, not free -- but evaluation still enumerates
        # every one of their domains, so they are part of the cost.
        return self.expression.footprint() | frozenset(self.variables)

    def conditioned_on(self) -> frozenset[str]:
        return self.expression.conditioned_on() - frozenset(self.variables)

    def kernels(self) -> tuple[Kernel, ...]:
        return self.expression.kernels()

    def canonical_key(self) -> tuple[Any, ...]:
        return ("sumout", self.variables, self.expression.canonical_key())
