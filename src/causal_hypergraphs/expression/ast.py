from __future__ import annotations

from dataclasses import dataclass


def _items(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(sorted(str(v) for v in values))


def _join(values: tuple[str, ...]) -> str:
    return ",".join(values)


class Expression:
    """Base class for probability expression AST nodes."""

    def render(self) -> str:
        return str(self)

    def to_latex(self) -> str:
        return str(self)


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


@dataclass(frozen=True)
class Fallback(Expression):
    variable: str

    def __str__(self) -> str:
        return f"P0({self.variable})"

    def to_latex(self) -> str:
        return rf"P_0({self.variable})"


@dataclass(frozen=True)
class MechanismFactor(Expression):
    mechanism: str
    variables: tuple[str, ...]
    given: tuple[str, ...] = ()

    def __init__(self, mechanism: str, variables: object, given: object = ()) -> None:
        object.__setattr__(self, "mechanism", mechanism)
        object.__setattr__(self, "variables", _items(variables))
        object.__setattr__(self, "given", _items(given))

    def __str__(self) -> str:
        suffix = f" | {_join(self.given)}" if self.given else ""
        return f"P_{self.mechanism}({_join(self.variables)}{suffix})"

    def to_latex(self) -> str:
        suffix = rf" \mid {_join(self.given)}" if self.given else ""
        return rf"P_{{{self.mechanism}}}({_join(self.variables)}{suffix})"


@dataclass(frozen=True)
class ReplacementFactor(MechanismFactor):
    pass


@dataclass(frozen=True)
class Quotient(Expression):
    numerator: Expression
    denominator: Expression

    def __str__(self) -> str:
        return f"{self.numerator} / {self.denominator}"

    def to_latex(self) -> str:
        return rf"\frac{{{self.numerator.to_latex()}}}{{{self.denominator.to_latex()}}}"


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
        object.__setattr__(self, "factors", tuple(flattened))

    def __str__(self) -> str:
        if not self.factors:
            return "1"
        return " * ".join(str(factor) for factor in self.factors)

    def to_latex(self) -> str:
        if not self.factors:
            return "1"
        return r" \cdot ".join(factor.to_latex() for factor in self.factors)
