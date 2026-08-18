"""Give a copied variable a meaning.

Identification formulas routinely introduce a *copy* of a variable. The front-door
estimand is the familiar case:

    sum_z P(z | x) sum_{x'} P(x') P(y | x', z)

`x'` is a fresh name for `x`, needed because the inner sum must not capture the `x` held at
the do-value by the outer factor. The two occurrences are different quantities and must
have different names, or the expression means something else.

But a fresh name has no distribution. `P(X_prime)` is not a kernel any model supplies, and
until now the library emitted such expressions and could not evaluate them -- the front-door
estimand it ships raises `KeyError: 'X_prime'` the moment anyone asks it for a number, which
is why a test comparing its rendered text passed for so long.

An `AliasModel` supplies the missing meaning: a copy is looked up under its base, at the
copy's own value. The resolution happens **per kernel call**, and that is the whole
subtlety. Rebasing the variable once for the model would make the outer `P(z | x)` read the
copy's value too, and the estimand would come out wrong by a wide margin rather than by a
rounding error. Each kernel is renamed on its own, so a factor mentioning `x` and a factor
mentioning `x'` in the same product read different values.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .discrete import Assignment, Model, SemanticsError


class _Domains(Mapping[str, tuple[Any, ...]]):
    """The model's domains, answering for copies as well as for bases."""

    def __init__(
        self, inner: Mapping[str, tuple[Any, ...]], aliases: Mapping[str, str]
    ) -> None:
        self._inner = inner
        self._aliases = aliases

    def __getitem__(self, name: str) -> tuple[Any, ...]:
        return self._inner[self._aliases.get(name, name)]

    def __iter__(self):  # type: ignore[no-untyped-def]
        seen = list(self._inner)
        return iter(seen + [name for name in self._aliases if name not in self._inner])

    def __len__(self) -> int:
        return len(list(iter(self)))


class AliasModel:
    """A `Model` in which each copy resolves to its base, one kernel at a time."""

    def __init__(self, inner: Model, aliases: Mapping[str, str]) -> None:
        self._inner = inner
        self._aliases = dict(aliases)
        self._domains = _Domains(inner.domains, self._aliases)

    @property
    def domains(self) -> Mapping[str, tuple[Any, ...]]:
        return self._domains

    def _resolve(
        self, *groups: Sequence[str], assignment: Assignment
    ) -> tuple[list[tuple[str, ...]], dict[str, Any]]:
        """Rename one kernel's arguments to base names, carrying the copies' values."""
        renamed: list[tuple[str, ...]] = []
        values = dict(assignment)
        for group in groups:
            names: list[str] = []
            for name in group:
                base = self._aliases.get(name, name)
                if base != name:
                    if base in group:
                        raise SemanticsError(
                            f"Kernel references both {name!r} and its base {base!r}. A "
                            "copy and its original are different quantities and cannot be "
                            "arguments of the same kernel."
                        )
                    values[base] = assignment[name]
                names.append(base)
            renamed.append(tuple(names))
        return renamed, values

    def conditional(
        self, variables: Sequence[str], given: Sequence[str], assignment: Assignment
    ) -> float:
        (left, right), values = self._resolve(variables, given, assignment=assignment)
        return self._inner.conditional(left, right, values)

    def conditional_expectation(
        self, target: str, given: Sequence[str], assignment: Assignment
    ) -> float:
        (names,), values = self._resolve(given, assignment=assignment)
        base = self._aliases.get(target, target)
        return self._inner.conditional_expectation(base, names, values)

    def fallback(
        self,
        mechanism: str,
        variables: Sequence[str],
        assignment: Assignment,
        marginalized: Sequence[str] = (),
    ) -> float:
        (names, dropped), values = self._resolve(
            variables, marginalized, assignment=assignment
        )
        return self._inner.fallback(mechanism, names, values, dropped)

    def replacement(
        self,
        mechanism: str,
        variables: Sequence[str],
        given: Sequence[str],
        assignment: Assignment,
    ) -> float:
        (left, right), values = self._resolve(variables, given, assignment=assignment)
        return self._inner.replacement(mechanism, left, right, values)


def with_aliases(model: Model, aliases: Mapping[str, str]) -> Model:
    """`model`, able to answer for copied variables. Returns `model` when there are none."""
    if not aliases:
        return model
    return AliasModel(model, aliases)
