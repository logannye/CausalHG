"""An empirical model that is never assembled into a joint.

Variable elimination removes the exponential from *evaluating* an estimand, but only if
the model can hand it one factor at a time. The obvious way to build an empirical model --
tabulate the observational law over everything the estimand mentions, then read
conditionals off that table -- puts the same exponential back one layer down: a query over
a forty-link chain would need `2**40` cells before elimination got a chance to avoid them.

So this model computes each kernel from row counts over *that kernel's own variables*.
`P(out(m) | in(m))` costs a pass over the rows tallying `out(m) ∪ in(m)`, whatever else the
estimand happens to mention, and the tallies are cached per variable set because a factor
is asked for once per cell of its scope.

The numbers are identical to the ones `DiscreteModel` would give for the same data -- the
empirical joint marginalized to a factor's variables *is* the count of those variables over
the row total -- so this changes what is materialized, not what is estimated.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from causal_hypergraphs.semantics import (
    Assignment,
    MissingKernel,
    SemanticsError,
    UndefinedEstimand,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to a type checker
    from .dataset import Dataset, Point


class EmpiricalModel:
    """A `Model` backed by row counts, one factor at a time.

    Parameters
    ----------
    data:
        The rows. Conditional expectations are served straight from it, so a continuous
        measure is never discretized on the way through.
    variables:
        The variables the estimand ranges over -- its footprint, not its scope. Summed
        variables need domains even though no caller binds them.
    """

    def __init__(
        self,
        data: Dataset,
        variables: Sequence[str],
        *,
        fallbacks: Mapping[str, Mapping[Point, float]] | None = None,
        replacements: Mapping[str, Mapping[tuple[Point, Point], float]] | None = None,
    ) -> None:
        self._data = data
        self._domains = {name: data.domains[name] for name in variables}
        self._fallbacks = dict(fallbacks or {})
        self._replacements = dict(replacements or {})
        self._tallies: dict[tuple[str, ...], dict[Point, int]] = {}
        self._rows = float(data.n_rows)

    @property
    def domains(self) -> Mapping[str, tuple[Any, ...]]:
        return self._domains

    def counts(self, names: Sequence[str]) -> Mapping[Point, int]:
        """Row counts over `names`, tallied once and kept.

        Public because the caller that reports how thin the estimand's strata are needs
        exactly the tallies this model already built. Recomputing them would double a pass
        over the rows for every conditioning set the estimand touched.
        """
        key = tuple(names)
        tally = self._tallies.get(key)
        if tally is None:
            tally = self._data.counts(key)
            self._tallies[key] = tally
        return tally

    def _count(self, names: tuple[str, ...], assignment: Assignment) -> int:
        missing = [name for name in names if name not in assignment]
        if missing:
            raise SemanticsError(f"Assignment does not bind {missing}.")
        return self.counts(names).get(tuple(assignment[name] for name in names), 0)

    def conditional(
        self, variables: Sequence[str], given: Sequence[str], assignment: Assignment
    ) -> float:
        """``P(variables | given)`` as a ratio of two row counts."""
        joint = tuple(sorted(set(variables) | set(given)))
        numerator = self._count(joint, assignment)
        if not given:
            return numerator / self._rows
        conditioning = tuple(sorted(given))
        denominator = self._count(conditioning, assignment)
        if denominator == 0:
            stratum = {name: assignment[name] for name in conditioning}
            raise UndefinedEstimand(
                f"P({','.join(variables)} | {','.join(given)}) is undefined: the "
                f"conditioning event has probability zero at {stratum!r}.",
                kernel=f"P({','.join(variables)} | {','.join(given)})",
                stratum=stratum,
            )
        return numerator / denominator

    def conditional_expectation(
        self, target: str, given: Sequence[str], assignment: Assignment
    ) -> float:
        return self._data.conditional_expectation(target, given, assignment)

    def fallback(
        self,
        mechanism: str,
        variables: Sequence[str],
        assignment: Assignment,
        marginalized: Sequence[str] = (),
    ) -> float:
        """``P0^mechanism(variables)``: policy, supplied rather than estimated.

        A missing table or cell raises rather than defaulting to zero, which would install
        a different intervention than the caller declared.
        """
        try:
            table = self._fallbacks[mechanism]
        except KeyError as exc:
            raise MissingKernel(
                f"No fallback policy P0_{mechanism}({','.join(variables)}) supplied."
            ) from exc
        missing = [name for name in variables if name not in assignment]
        if missing:
            raise SemanticsError(f"Assignment does not bind {missing}.")
        if marginalized:
            outputs = sorted(tuple(variables) + tuple(marginalized))
            position = {name: index for index, name in enumerate(outputs)}
            wanted = [(position[name], assignment[name]) for name in variables]
            total = 0.0
            matched = False
            for key, probability in table.items():
                if all(key[index] == value for index, value in wanted):
                    total += probability
                    matched = True
            if not matched:
                raise MissingKernel(
                    f"Fallback policy P0_{mechanism} has no entry with "
                    f"{ {name: assignment[name] for name in variables} !r}."
                )
            return total
        key = tuple(assignment[name] for name in sorted(variables))
        try:
            return table[key]
        except KeyError as exc:
            raise MissingKernel(
                f"Fallback policy P0_{mechanism} has no entry for {key!r}."
            ) from exc

    def replacement(
        self,
        mechanism: str,
        variables: Sequence[str],
        given: Sequence[str],
        assignment: Assignment,
    ) -> float:
        try:
            table = self._replacements[mechanism]
        except KeyError as exc:
            raise MissingKernel(f"No replacement kernel supplied for {mechanism!r}.") from exc
        key = (
            tuple(assignment[name] for name in sorted(variables)),
            tuple(assignment[name] for name in sorted(given)),
        )
        try:
            return table[key]
        except KeyError as exc:
            raise MissingKernel(
                f"Replacement kernel {mechanism!r} has no entry for {key!r}."
            ) from exc
