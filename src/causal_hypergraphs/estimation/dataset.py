"""Observational data an estimand can be evaluated against.

A `Dataset` is a table of finite-valued observations plus one thing most tabular wrappers
leave implicit: **the unit of independence**. Rows are not generally exchangeable -- cells
come from donors, wells from plates, reads from libraries -- and resampling rows when the
independent unit is the donor produces an interval that is too narrow, often by a lot.
That parameter is therefore part of the type, and the estimate reports which unit it used.

Two kinds of column. **Variables** are finite-valued and form the sample space the
estimator enumerates. **Measures** are numeric and are never discretized: they can only be
reached through a conditional expectation, which integrates them inside the node rather
than enumerating them. That is how a continuous readout -- an expression level, a growth
rate -- is handled without binning, and binning is not a neutral preprocessing step: it
can create or destroy the very data support the estimator checks.

A variable that is genuinely continuous and appears as a *conditioning* variable still has
to be binned by the caller, deliberately and visibly.
"""
from __future__ import annotations

import itertools
import random
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from causal_hypergraphs.semantics import DiscreteModel, UndefinedEstimand

Point = tuple[Any, ...]


class DatasetError(Exception):
    """The records cannot be read as a table of finite-valued observations."""


@dataclass(frozen=True)
class Dataset:
    """A finite-valued observational table with a declared unit of independence.

    Attributes
    ----------
    variables:
        Modelled column names, sorted. The unit column is not among them.
    domains:
        The value set of each variable. Inferred from the data unless supplied; supply it
        when a level is possible but unobserved, since an inferred domain cannot know
        about a level that never appears.
    rows:
        One value tuple per observation, in `variables` order.
    units:
        The independent unit each row belongs to, positionally aligned with `rows`.
    unit_column:
        The column units were read from, or None when each row is its own unit.
    """

    variables: tuple[str, ...]
    domains: Mapping[str, tuple[Any, ...]]
    rows: tuple[Point, ...]
    units: tuple[Hashable, ...]
    unit_column: str | None = None
    measures: tuple[str, ...] = ()
    """Numeric columns kept as real values, never discretized.

    A measure can be the target of `E[Y | ...]` but never a coordinate of the sample
    space: the estimator integrates it inside the expectation instead of enumerating it.
    That is what lets a readout be an expression level or a growth rate without binning,
    and binning is not neutral -- it can create or destroy the data support the estimator
    checks.
    """
    measurements: Mapping[str, tuple[float, ...]] = field(default_factory=dict)
    """Each measure's values, positionally aligned with `rows`."""

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        domains: Mapping[str, Sequence[Any]] | None = None,
        unit: str | None = None,
        measures: Sequence[str] = (),
    ) -> Dataset:
        """Build a dataset from row dicts.

        `unit` names a column identifying the independent sampling unit -- donor, plate,
        library. Rows sharing a value are resampled together when bootstrapping. Omit it
        only when rows really are independent; the default treats each row as its own
        unit, which is the assumption that yields the narrowest interval.

        `measures` names numeric columns to keep as real values rather than discretize.
        They are excluded from `variables` and `domains`, and can only be reached through
        a conditional expectation.
        """
        materialized = [dict(record) for record in records]
        if not materialized:
            raise DatasetError("Cannot build a dataset from zero records.")

        columns = set(materialized[0])
        if unit is not None and unit not in columns:
            raise DatasetError(f"Unit column {unit!r} is not present in the records.")
        measure_names = tuple(sorted(measures))
        absent = [name for name in measure_names if name not in columns]
        if absent:
            raise DatasetError(f"Measure column(s) {absent} are not present in the records.")
        if unit is not None and unit in measure_names:
            raise DatasetError(f"{unit!r} cannot be both the unit column and a measure.")
        reserved = set(measure_names) | ({unit} if unit is not None else set())
        names = tuple(sorted(columns - reserved))
        if not names and not measure_names:
            raise DatasetError("Records contain no modelled variables.")

        for position, record in enumerate(materialized):
            missing = [name for name in names if name not in record]
            if missing:
                raise DatasetError(f"Record {position} is missing {missing}.")

        observed: dict[str, set[Any]] = {name: set() for name in names}
        for record in materialized:
            for name in names:
                observed[name].add(record[name])

        if domains is None:
            resolved = {name: tuple(sorted(values)) for name, values in observed.items()}
        else:
            missing_domains = [name for name in names if name not in domains]
            if missing_domains:
                raise DatasetError(f"No domain supplied for {missing_domains}.")
            resolved = {name: tuple(domains[name]) for name in names}
            for name in names:
                stray = observed[name] - set(resolved[name])
                if stray:
                    raise DatasetError(
                        f"Column {name!r} contains {sorted(stray)}, outside its declared "
                        f"domain {list(resolved[name])}."
                    )

        rows = tuple(tuple(record[name] for name in names) for record in materialized)
        units: tuple[Hashable, ...] = (
            tuple(record[unit] for record in materialized)
            if unit is not None
            else tuple(range(len(materialized)))
        )
        measurements = {
            name: tuple(float(record[name]) for record in materialized)
            for name in measure_names
        }
        return cls(
            variables=names,
            domains=resolved,
            rows=rows,
            units=units,
            unit_column=unit,
            measures=measure_names,
            measurements=measurements,
        )

    @classmethod
    def from_counts(
        cls,
        counts: Mapping[Point, int],
        variables: Sequence[str],
        *,
        domains: Mapping[str, Sequence[Any]] | None = None,
    ) -> Dataset:
        """Build a dataset from a contingency table keyed by value tuple.

        Equivalent to `from_records` on the expanded rows, but without materializing a
        dict per observation. Use it when the data arrive already tallied. Each row is its
        own unit here: a contingency table has discarded whatever grouping the rows had,
        so there is no honest way to reconstruct one.
        """
        names = tuple(variables)
        if len(set(names)) != len(names):
            raise DatasetError(f"Duplicate variable names: {list(names)}.")
        for key, count in counts.items():
            if len(key) != len(names):
                raise DatasetError(f"Count key {key!r} does not match variables {list(names)}.")
            if count < 0:
                raise DatasetError(f"Negative count {count} at {key!r}.")
        if not any(counts.values()):
            raise DatasetError("Cannot build a dataset from an all-zero contingency table.")

        observed: dict[str, set[Any]] = {name: set() for name in names}
        for key, count in counts.items():
            if count:
                for name, value in zip(names, key, strict=True):
                    observed[name].add(value)
        if domains is None:
            resolved = {name: tuple(sorted(values)) for name, values in observed.items()}
        else:
            resolved = {name: tuple(domains[name]) for name in names}

        order = {name: position for position, name in enumerate(names)}
        sorted_names = tuple(sorted(names))
        rows: list[Point] = []
        for key, count in counts.items():
            if not count:
                continue
            row = tuple(key[order[name]] for name in sorted_names)
            rows.extend([row] * count)
        return cls(
            variables=sorted_names,
            domains={name: resolved[name] for name in sorted_names},
            rows=tuple(rows),
            units=tuple(range(len(rows))),
            unit_column=None,
        )

    # -- shape -----------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_units(self) -> int:
        return len(set(self.units))

    @property
    def unit_description(self) -> str:
        if self.unit_column is None:
            return "row (rows assumed independent)"
        return f"{self.unit_column!r}"

    # -- laws ------------------------------------------------------------------

    def counts(self, variables: Sequence[str]) -> dict[Point, int]:
        """Row counts over `variables`, keyed by value tuple in the given order."""
        positions = self._positions(variables)
        tally: dict[Point, int] = {}
        for row in self.rows:
            key = tuple(row[position] for position in positions)
            tally[key] = tally.get(key, 0) + 1
        return tally

    def empirical_joint(self, variables: Sequence[str]) -> dict[Point, float]:
        """The empirical law over `variables`, with every domain cell present.

        Cells with no observations are present and zero rather than absent. That is what
        lets a downstream positivity check distinguish "this cell has no support" from
        "this cell was never mentioned", and it is why the estimator can name an empty
        stratum instead of returning a silent `nan`.
        """
        tally = self.counts(variables)
        total = float(self.n_rows)
        full: dict[Point, float] = {
            key: 0.0
            for key in itertools.product(*(self.domains[name] for name in variables))
        }
        for key, count in tally.items():
            full[key] = count / total
        return full

    def model(
        self,
        variables: Sequence[str],
        *,
        fallbacks: Mapping[str, Mapping[Point, float]] | None = None,
        replacements: Mapping[str, Mapping[tuple[Point, Point], float]] | None = None,
    ) -> DiscreteModel:
        """A `DiscreteModel` whose observational law is this dataset's empirical law.

        Restricted to `variables` -- normally the estimand's scope. Marginalization
        commutes with the evaluator's own marginalization, so restricting here is exact
        and keeps the materialized joint exponential in the estimand's scope rather than
        in the dataset's width.
        """
        ordered = tuple(variables)
        return DiscreteModel(
            domains={name: self.domains[name] for name in ordered},
            joint=self.empirical_joint(sorted(ordered)),
            fallbacks=dict(fallbacks or {}),
            replacements=dict(replacements or {}),
        )

    def conditional_expectation(
        self, target: str, given: Sequence[str], assignment: Mapping[str, Any]
    ) -> float:
        """``E[target | given]`` as the mean of `target` over the matching rows.

        This is where a continuous readout is actually handled: a group mean needs no
        domain for `target`, so nothing is binned. An empty cell raises rather than
        averaging over nothing, which routes it into the same certificate discharge as an
        empty conditioning cell in the density form -- a named stratum, not a `nan`.
        """
        if target not in self.measurements:
            raise DatasetError(
                f"{target!r} is not a measure column of this dataset. Declare it with "
                f"`measures=({target!r},)` so it is kept as a number rather than "
                "discretized. Measures present: "
                f"{list(self.measures)}."
            )
        values = self.measurements[target]
        if not given:
            return sum(values) / len(values)

        positions = self._positions(tuple(given))
        wanted = tuple(assignment[name] for name in given)
        total = 0.0
        count = 0
        for row, value in zip(self.rows, values, strict=True):
            if all(row[p] == v for p, v in zip(positions, wanted, strict=True)):
                total += value
                count += 1
        if count == 0:
            raise UndefinedEstimand(
                f"E[{target} | {','.join(given)}] is undefined: no rows at "
                f"{dict(zip(given, wanted, strict=True))!r}.",
                kernel=f"E[{target} | {','.join(given)}]",
                stratum=dict(zip(given, wanted, strict=True)),
            )
        return total / count

    # -- resampling ------------------------------------------------------------

    def resample(self, rng: random.Random) -> Dataset:
        """Draw a bootstrap replicate by sampling *units* with replacement.

        Resampling units rather than rows is what makes the resulting interval honest
        when rows within a unit are correlated. The replicate keeps this dataset's
        declared domains, so a level that vanishes under resampling still exists as a
        zero cell rather than silently leaving the model.
        """
        grouped: dict[Hashable, list[int]] = {}
        for position, unit in enumerate(self.units):
            grouped.setdefault(unit, []).append(position)

        labels = list(grouped)
        drawn = [labels[rng.randrange(len(labels))] for _ in labels]
        indices: list[int] = []
        units: list[Hashable] = []
        for replicate, label in enumerate(drawn):
            for position in grouped[label]:
                indices.append(position)
                units.append(replicate)
        return Dataset(
            variables=self.variables,
            domains=self.domains,
            rows=tuple(self.rows[i] for i in indices),
            units=tuple(units),
            unit_column=self.unit_column,
            measures=self.measures,
            measurements={
                name: tuple(values[i] for i in indices)
                for name, values in self.measurements.items()
            },
        )

    # -- internals -------------------------------------------------------------

    def _positions(self, variables: Sequence[str]) -> tuple[int, ...]:
        index = {name: position for position, name in enumerate(self.variables)}
        missing = [name for name in variables if name not in index]
        if missing:
            raise DatasetError(f"Dataset has no column(s) {missing}.")
        return tuple(index[name] for name in variables)
