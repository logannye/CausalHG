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
import math
import random
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Complex, Integral, Number, Rational, Real
from types import MappingProxyType
from typing import Any, cast

from causal_hypergraphs.semantics import DiscreteModel, UndefinedEstimand

Point = tuple[Any, ...]


class DatasetError(Exception):
    """The records cannot be read as a table of finite-valued observations."""


def _column_names(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, str | bytes | Mapping | set | frozenset):
        raise DatasetError(f"{label} must be an ordered sequence of column names.")
    try:
        names = tuple(values)
    except TypeError as exc:
        raise DatasetError(f"{label} must be an ordered sequence of column names.") from exc
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise DatasetError(f"{label} must contain nonempty string column names.")
    if len(set(names)) != len(names):
        raise DatasetError(f"{label} contains duplicate column names.")
    return names


def _category(value: Any, label: str) -> Any:
    """Reject missing/nonfinite/unstable labels without turning them into categories."""
    if value is None:
        raise DatasetError(
            f"{label} is missing (None); handle missingness explicitly before ingestion."
        )
    try:
        hash(value)
    except TypeError as exc:
        raise DatasetError(f"{label} must be a hashable categorical value.") from exc
    try:
        present = bool(value == value)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise DatasetError(f"{label} has missing or ambiguous equality semantics.") from exc
    if not present:
        raise DatasetError(f"{label} is missing or nonfinite; handle missingness before ingestion.")
    if isinstance(value, tuple | frozenset):
        for member in value:
            _category(member, label)
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise DatasetError(f"{label} must be finite.")
    elif isinstance(value, Number) and not isinstance(value, Rational):
        try:
            finite = math.isfinite(cast(Any, getattr(value, "real", value))) and math.isfinite(
                getattr(value, "imag", 0)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise DatasetError(f"{label} must be a finite numeric category.") from exc
        if not finite:
            raise DatasetError(f"{label} must be finite.")
    return value


def _measure(value: Any, label: str) -> float:
    if not isinstance(value, Number | Decimal) or (
        isinstance(value, Complex) and not isinstance(value, Real)
    ):
        raise DatasetError(
            f"{label} must be a finite numeric measure, not missing or textual data."
        )
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise DatasetError(f"{label} must be a finite numeric measure.") from exc
    if not math.isfinite(number):
        raise DatasetError(f"{label} must be a finite numeric measure.")
    return number


def _domains(
    domains: Mapping[str, Sequence[Any]],
    variables: tuple[str, ...],
) -> dict[str, tuple[Any, ...]]:
    if not isinstance(domains, Mapping) or set(domains) != set(variables):
        raise DatasetError("Declared domains must cover exactly the categorical variable columns.")
    resolved = {}
    for name in variables:
        values = domains[name]
        if isinstance(values, str | bytes | Mapping | set | frozenset):
            raise DatasetError(f"Domain for {name!r} must be an ordered sequence of levels.")
        try:
            levels = tuple(_category(value, f"Domain {name!r}") for value in values)
        except TypeError as exc:
            raise DatasetError(
                f"Domain for {name!r} must be an ordered sequence of levels."
            ) from exc
        if not levels or len(set(levels)) != len(levels):
            raise DatasetError(f"Domain for {name!r} must be nonempty with unique levels.")
        resolved[name] = levels
    return resolved


def _inferred_domain(values: set[Any], name: str) -> tuple[Any, ...]:
    try:
        return tuple(sorted(values))
    except TypeError as exc:
        raise DatasetError(
            f"Column {name!r} has incomparable categorical labels; "
            "declare an explicit domain order."
        ) from exc


@dataclass(frozen=True)
class Dataset:
    """A finite-valued observational table with a declared unit of independence.

    Containers are copied into immutable snapshots. Categorical and unit labels
    must themselves have stable hashing and equality throughout the dataset's use.

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

    def __post_init__(self) -> None:
        """Validate direct and factory construction equally, and snapshot their containers."""
        original_names = _column_names(self.variables, "Variables")
        measure_names = _column_names(self.measures, "Measures")
        if not original_names and not measure_names:
            raise DatasetError("A dataset must contain at least one variable or measure column.")
        if set(original_names) & set(measure_names):
            raise DatasetError("Categorical variables and numeric measures must be disjoint.")
        if self.unit_column is not None:
            _column_names((self.unit_column,), "Unit column")
            if self.unit_column in set(original_names) | set(measure_names):
                raise DatasetError("The unit column cannot also be a modelled variable or measure.")
        names = tuple(sorted(original_names))
        domains = _domains(self.domains, names)
        positions = tuple(original_names.index(name) for name in names)
        if isinstance(self.rows, str | bytes | Mapping):
            raise DatasetError("Rows must be a sequence of value tuples.")
        try:
            raw_rows = tuple(self.rows)
        except TypeError as exc:
            raise DatasetError("Rows must be a sequence of value tuples.") from exc
        if not raw_rows:
            raise DatasetError("Cannot build a dataset with zero rows.")
        rows = []
        domain_sets = {name: set(values) for name, values in domains.items()}
        for index, row in enumerate(raw_rows):
            if not isinstance(row, Sequence) or isinstance(row, str | bytes):
                raise DatasetError(f"Row {index} must be a sequence matching the variable columns.")
            if len(row) != len(original_names):
                raise DatasetError(f"Row {index} does not match the number of variable columns.")
            ordered = tuple(row[position] for position in positions)
            for name, value in zip(names, ordered, strict=True):
                _category(value, f"Row {index}, column {name!r}")
                if value not in domain_sets[name]:
                    raise DatasetError(
                        f"Row {index}, column {name!r} is outside its declared domain."
                    )
            rows.append(ordered)
        if isinstance(self.units, str | bytes | Mapping):
            raise DatasetError("Unit identifiers must be a sequence aligned with rows.")
        try:
            units = tuple(_category(value, "Unit identifier") for value in self.units)
        except TypeError as exc:
            raise DatasetError("Unit identifiers must be a sequence aligned with rows.") from exc
        if len(units) != len(rows):
            raise DatasetError("Unit identifiers must have exactly one entry per row.")
        if self.unit_column is None and len(set(units)) != len(units):
            raise DatasetError("Repeated unit identifiers require a named unit_column.")
        if not isinstance(self.measurements, Mapping) or set(self.measurements) != set(
            measure_names
        ):
            raise DatasetError("Measurements must cover exactly the declared measure columns.")
        measurements = {}
        for name in sorted(measure_names):
            values = self.measurements[name]
            if isinstance(values, str | bytes | Mapping):
                raise DatasetError(f"Measure {name!r} must be a sequence aligned with rows.")
            try:
                numeric = tuple(_measure(value, f"Measure {name!r}") for value in values)
            except TypeError as exc:
                raise DatasetError(
                    f"Measure {name!r} must be a sequence aligned with rows."
                ) from exc
            if len(numeric) != len(rows):
                raise DatasetError(f"Measure {name!r} must have exactly one entry per row.")
            measurements[name] = numeric
        object.__setattr__(self, "variables", names)
        object.__setattr__(self, "domains", MappingProxyType(domains))
        object.__setattr__(self, "rows", tuple(rows))
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "measures", tuple(sorted(measure_names)))
        object.__setattr__(self, "measurements", MappingProxyType(measurements))

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

        Missing and nonfinite values are rejected in variables, units and measures.
        Every record must have the same columns. Handle missingness explicitly before
        ingestion; this constructor never invents a missing-value category or drops rows.
        """
        try:
            iterator = iter(records)
        except TypeError as exc:
            raise DatasetError("Records must be an iterable of row mappings.") from exc
        materialized = []
        for position, record in enumerate(iterator):
            if not isinstance(record, Mapping):
                raise DatasetError(
                    f"Record {position} must be a mapping of column names to values."
                )
            materialized.append(dict(record))
        if not materialized:
            raise DatasetError("Cannot build a dataset from zero records.")

        columns = set(materialized[0])
        _column_names(tuple(materialized[0]), "Record columns")
        if unit is not None:
            _column_names((unit,), "Unit column")
        if unit is not None and unit not in columns:
            raise DatasetError(f"Unit column {unit!r} is not present in the records.")
        measure_names = tuple(sorted(_column_names(measures, "Measures")))
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
            if set(record) != columns:
                missing = sorted(columns - set(record))
                extra = list(set(record) - columns)
                raise DatasetError(
                    f"Record {position} has a different schema: missing {missing}, extra {extra}."
                )

        observed: dict[str, set[Any]] = {name: set() for name in names}
        for position, record in enumerate(materialized):
            for name in names:
                observed[name].add(_category(record[name], f"Record {position}, column {name!r}"))

        if domains is None:
            resolved = {name: _inferred_domain(values, name) for name, values in observed.items()}
        else:
            resolved = _domains(domains, names)

        rows = tuple(tuple(record[name] for name in names) for record in materialized)
        units: tuple[Hashable, ...] = (
            tuple(record[unit] for record in materialized)
            if unit is not None
            else tuple(range(len(materialized)))
        )
        measurements = {
            name: tuple(_measure(record[name], f"Measure {name!r}") for record in materialized)
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
        names = _column_names(variables, "Variables")
        if not names:
            raise DatasetError("A contingency table must declare at least one variable.")
        if not isinstance(counts, Mapping):
            raise DatasetError("Counts must be a mapping from value tuples to integer counts.")
        for key, count in counts.items():
            if not isinstance(key, tuple) or len(key) != len(names):
                raise DatasetError(f"Count key {key!r} does not match variables {list(names)}.")
            if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
                raise DatasetError(f"Count {count!r} at {key!r} must be a nonnegative integer.")
            for name, value in zip(names, key, strict=True):
                _category(value, f"Count coordinate {name!r}")
        if not any(counts.values()):
            raise DatasetError("Cannot build a dataset from an all-zero contingency table.")

        observed: dict[str, set[Any]] = {name: set() for name in names}
        for key, count in counts.items():
            if count:
                for name, value in zip(names, key, strict=True):
                    observed[name].add(value)
        if domains is None:
            resolved = {name: _inferred_domain(values, name) for name, values in observed.items()}
        else:
            resolved = _domains(domains, names)

        if domains is not None:
            for key in counts:
                if any(value not in resolved[name] for name, value in zip(names, key, strict=True)):
                    raise DatasetError(f"Count coordinate {key!r} is outside its declared domain.")

        order = {name: position for position, name in enumerate(names)}
        sorted_names = tuple(sorted(names))
        rows: list[Point] = []
        for key, count in counts.items():
            if not count:
                continue
            row = tuple(key[order[name]] for name in sorted_names)
            rows.extend([row] * int(count))
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
            key: 0.0 for key in itertools.product(*(self.domains[name] for name in variables))
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

        Restricted to `variables`. Marginalization commutes with the evaluator's own, so
        restricting here is exact and keeps the materialized joint exponential in the
        estimand's footprint rather than in the dataset's width.

        Exponential in the footprint is still exponential, which is why `estimate` does
        **not** use this: it builds an `EmpiricalModel`, which counts each factor over that
        factor's own variables and never assembles a joint at all. Use this one for exact
        work on a handful of variables, where having the whole law in hand is the point.
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
