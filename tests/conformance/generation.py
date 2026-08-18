"""Deterministic random generation of finite hypergraph SCMs satisfying C1-C4.

The generator produces a `RandomModel`, which carries both the typed incidence the
compiler sees and the numerical kernels the compiler does *not* see. Ground truth is
computed here from the kernels, so a conformance check compares the compiler's
symbolic answer against a law it had no access to.

Everything is seeded from a single integer and uses `random.Random`, so a failing case
is reproducible from its seed alone and CI cannot flake.

Three kernel shapes matter, because the two defects found by hand both lived in the
non-positive ones:

``positive``
    Every output tuple has positive probability. The comfortable case.
``coupled``
    Supported only on tuples where all outputs are equal -- the discrete image of
    stoichiometric coupling, and a *singular* mechanism factor. Declared via
    ``output_equalities`` so the determination closure can see it.
``sparse``
    Structural zeros without any declared determination. Exercises positivity
    handling where the compiler has not been told why the zeros are there.
"""
from __future__ import annotations

import itertools
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from causal_hypergraphs.graph import Mechanism, MechanismGraph

BINARY: tuple[int, ...] = (0, 1)

# Kernel entries below this are treated as structurally absent when building supports.
MIN_WEIGHT = 0.15

Assignment = dict[str, int]
Point = tuple[int, ...]
Kernel = dict[Point, dict[Point, float]]  # inputs -> outputs -> probability


@dataclass(frozen=True)
class MechanismSpec:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    latent: bool
    shape: str  # "positive" | "coupled" | "sparse"
    emit_equalities: bool
    """Whether to declare an `output_equalities` group over all outputs.

    Normally set only when the kernel really is coupled, so the declaration is *valid*
    in the sense of THEOREM_T1.md 1.1. The generator can also set it on a positive
    kernel, producing a **mendacious** declaration: an equality the model does not
    satisfy. That is the input class that makes T1's validity hypothesis load-bearing.
    """

    def as_mechanism(self) -> Mechanism:
        equalities = (
            (tuple(self.outputs),) if self.emit_equalities and len(self.outputs) > 1 else ()
        )
        return Mechanism(
            self.name,
            inputs=self.inputs,
            outputs=self.outputs,
            latent=self.latent,
            output_equalities=equalities,
        )


@dataclass(frozen=True)
class RandomModel:
    """A finite hypergraph SCM with its numerical kernels."""

    seed: int
    variables: tuple[str, ...]
    observed: tuple[str, ...]
    mechanisms: tuple[MechanismSpec, ...]
    exogenous_laws: Mapping[str, Mapping[int, float]]
    kernels: Mapping[str, Kernel]
    fallbacks: Mapping[str, Mapping[Point, float]]
    """Joint deletion policies ``P0^m(out(m))``, keyed by mechanism then by output tuple.

    Deliberately *not* per-variable: a product policy forces the orphaned outputs
    independent, which is the case the framework used to be unable to state. Some of these
    are generated non-factorizing on purpose -- see `non_factorizing_fallbacks`.
    """
    replacements: Mapping[str, Kernel]

    # -- structure -------------------------------------------------------------

    @property
    def domains(self) -> dict[str, tuple[int, ...]]:
        return {v: BINARY for v in self.variables}

    @property
    def observed_domains(self) -> dict[str, tuple[int, ...]]:
        return {v: BINARY for v in self.observed}

    @property
    def produced(self) -> frozenset[str]:
        return frozenset(v for spec in self.mechanisms for v in spec.outputs)

    @property
    def exogenous(self) -> tuple[str, ...]:
        return tuple(v for v in self.variables if v not in self.produced)

    def spec(self, name: str) -> MechanismSpec:
        for candidate in self.mechanisms:
            if candidate.name == name:
                return candidate
        raise KeyError(name)

    def graph(self) -> MechanismGraph:
        return MechanismGraph(
            variables=set(self.variables),
            mechanisms={spec.name: spec.as_mechanism() for spec in self.mechanisms},
            observed_variables=set(self.observed),
        )

    # -- exact laws ------------------------------------------------------------

    def assignments(self) -> list[Assignment]:
        return [
            dict(zip(self.variables, combo, strict=True))
            for combo in itertools.product(*(BINARY for _ in self.variables))
        ]

    def _factor(self, spec: MechanismSpec, x: Assignment) -> float:
        inputs = tuple(x[v] for v in spec.inputs)
        outputs = tuple(x[v] for v in spec.outputs)
        return self.kernels[spec.name][inputs][outputs]

    def _law(self, x: Assignment, override: Mapping[str, float]) -> float:
        value = 1.0
        for v in self.exogenous:
            value *= self.exogenous_laws[v][x[v]]
        for spec in self.mechanisms:
            value *= override[spec.name] if spec.name in override else self._factor(spec, x)
        return value

    def joint(self) -> dict[Point, float]:
        """The observational law P(V), keyed in `self.variables` order."""
        return {
            tuple(x[v] for v in self.variables): self._law(x, {}) for x in self.assignments()
        }

    def interventional_delete(self, target: str) -> dict[Point, float]:
        """P(V | delete(target)): the target factor becomes the joint fallback policy."""
        spec = self.spec(target)
        table = self.fallbacks[target]
        result: dict[Point, float] = {}
        for x in self.assignments():
            fallback = table[tuple(x[v] for v in sorted(spec.outputs))]
            result[tuple(x[v] for v in self.variables)] = self._law(x, {target: fallback})
        return result

    def sample_counts(
        self, law: Mapping[Point, float], n_rows: int, rng: random.Random
    ) -> dict[Point, int]:
        """Draw `n_rows` observations from `law`, returned as a contingency table.

        Counts rather than rows because the estimator accepts a contingency table
        directly, and because a sweep that materializes hundreds of thousands of row dicts
        spends its time in allocation rather than in the property under test.
        """
        cells = sorted(law)
        weights = [law[cell] for cell in cells]
        drawn = rng.choices(cells, weights=weights, k=n_rows)
        tally: dict[Point, int] = {}
        for cell in drawn:
            tally[cell] = tally.get(cell, 0) + 1
        return tally

    def non_factorizing_fallbacks(self) -> tuple[str, ...]:
        """Mechanisms whose deletion policy is not a product of its own marginals.

        A sweep that only ever saw product policies would pass just as well against the
        old per-variable type, so it could not detect a regression to it.
        """
        found: list[str] = []
        for spec in self.mechanisms:
            if len(spec.outputs) < 2:
                continue
            table = self.fallbacks[spec.name]
            marginals = [
                {
                    value: sum(p for point, p in table.items() if point[position] == value)
                    for value in BINARY
                }
                for position in range(len(spec.outputs))
            ]
            for point, probability in table.items():
                product = 1.0
                for position, value in enumerate(point):
                    product *= marginals[position][value]
                if abs(probability - product) > 1e-9:
                    found.append(spec.name)
                    break
        return tuple(found)

    def interventional_replace(self, target: str) -> dict[Point, float]:
        """P(V | replace(target, target')) using the generated replacement kernel."""
        spec = self.spec(target)
        kernel = self.replacements[target]
        result: dict[Point, float] = {}
        for x in self.assignments():
            inputs = tuple(x[v] for v in spec.inputs)
            outputs = tuple(x[v] for v in spec.outputs)
            result[tuple(x[v] for v in self.variables)] = self._law(
                x, {target: kernel[inputs][outputs]}
            )
        return result

    def marginalize_to(
        self, law: Mapping[Point, float], names: Sequence[str]
    ) -> dict[Point, float]:
        """Sum a law over every variable outside `names`, re-keyed in `names` order.

        `law` is keyed in `self.variables` order. Marginal conformance needs an arbitrary
        subset rather than just the observed set, so the observed case delegates here
        instead of carrying a second copy of the same sum.
        """
        index = {name: position for position, name in enumerate(self.variables)}
        positions = [index[name] for name in names]
        result: dict[Point, float] = {
            combo: 0.0 for combo in itertools.product(*(BINARY for _ in names))
        }
        for key, value in law.items():
            result[tuple(key[p] for p in positions)] += value
        return result

    def marginalize_to_observed(self, law: Mapping[Point, float]) -> dict[Point, float]:
        """Sum a law over the hidden variables, re-keyed in `self.observed` order."""
        return self.marginalize_to(law, self.observed)

    def replacement_table(self, target: str) -> dict[tuple[Point, Point], float]:
        """The replacement kernel keyed as `DiscreteModel.replacements` expects.

        Both halves of the key are value tuples in sorted-variable order, which matches
        the AST's normalization of `ReplacementFactor.variables` and `.given`.
        """
        return {
            (outputs, inputs): probability
            for inputs, row in self.replacements[target].items()
            for outputs, probability in row.items()
        }


# --- generation ---------------------------------------------------------------


def _distribution(rng: random.Random, support: Sequence[Point]) -> dict[Point, float]:
    weights = {point: rng.uniform(MIN_WEIGHT, 1.0) for point in support}
    total = sum(weights.values())
    return {point: weight / total for point, weight in weights.items()}


def _scalar_law(rng: random.Random) -> dict[int, float]:
    p = rng.uniform(0.2, 0.8)
    return {0: 1.0 - p, 1: p}


def _fallback_policy(rng: random.Random, outputs: tuple[str, ...]) -> dict[Point, float]:
    """A joint deletion policy ``P0^m(out(m))`` over the outputs' value tuples.

    For a multi-output mechanism this draws either a free joint (generically
    non-factorizing) or a diagonal one supported on all-equal tuples -- the discrete image
    of coupling that survives the mechanism's removal. Both are unreachable by a product
    of per-variable laws, which is the point.
    """
    points = list(itertools.product(*(BINARY for _ in outputs)))
    if len(outputs) > 1 and rng.random() < 0.3:
        points = [point for point in points if len(set(point)) == 1]
    weights = {point: rng.uniform(MIN_WEIGHT, 1.0) for point in points}
    total = sum(weights.values())
    full = dict.fromkeys(itertools.product(*(BINARY for _ in outputs)), 0.0)
    full.update({point: weight / total for point, weight in weights.items()})
    return full


def _kernel(
    rng: random.Random, inputs: tuple[str, ...], outputs: tuple[str, ...], shape: str
) -> Kernel:
    """Build P(out | in) for every input configuration, in the requested shape."""
    out_points = list(itertools.product(*(BINARY for _ in outputs)))
    in_points = list(itertools.product(*(BINARY for _ in inputs)))

    if shape == "coupled" and len(outputs) > 1:
        support = [point for point in out_points if len(set(point)) == 1]
    elif shape == "sparse" and len(out_points) > 2:
        # Zero out a strict, non-empty subset while leaving at least two survivors, so
        # the kernel is neither degenerate nor positive.
        keep = rng.sample(out_points, rng.randint(2, len(out_points) - 1))
        support = keep
    else:
        support = out_points

    kernel: Kernel = {}
    for in_point in in_points:
        row = dict.fromkeys(out_points, 0.0)
        row.update(_distribution(rng, support))
        kernel[in_point] = row
    return kernel


def generate_model(
    seed: int,
    *,
    allow_hidden: bool = True,
    allow_coupled: bool = True,
    declare_equalities: bool = True,
    n_variables: int | None = None,
    shapes: Sequence[str] | None = None,
    mendacious: bool = False,
) -> RandomModel:
    """Generate one model satisfying C1-C4 by construction.

    C4 holds because each variable is drawn from an unproduced pool at most once. C1
    holds because a mechanism's inputs are drawn only from variables already realized,
    so the mechanism dependency graph is acyclic by construction. Inputs and outputs
    are disjoint for the same reason.
    """
    rng = random.Random(seed)
    count = n_variables if n_variables is not None else rng.randint(4, 6)
    variables = tuple(f"v{i}" for i in range(count))

    n_exogenous = rng.randint(1, max(1, count - 2))
    available = list(variables[:n_exogenous])
    pool = list(variables[n_exogenous:])

    specs: list[MechanismSpec] = []
    index = 0
    while pool and index < 4:
        n_out = rng.randint(1, min(2, len(pool)))
        outputs = tuple(sorted(rng.sample(pool, n_out)))
        n_in = rng.randint(1, min(2, len(available)))
        inputs = tuple(sorted(rng.sample(available, n_in)))

        if shapes is not None:
            choices = list(shapes)
        else:
            choices = ["positive", "sparse"]
            if allow_coupled and len(outputs) > 1:
                choices += ["coupled", "coupled"]  # weight coupling up; it is the hard case
        shape = rng.choice(choices)

        if mendacious:
            # Declare an equality the kernel does not satisfy. Invalid by construction.
            emit = shape == "positive" and len(outputs) > 1
        else:
            emit = declare_equalities and shape == "coupled" and len(outputs) > 1

        specs.append(
            MechanismSpec(
                name=f"m{index}",
                inputs=inputs,
                outputs=outputs,
                latent=rng.random() < 0.25,
                shape=shape,
                emit_equalities=emit,
            )
        )
        for v in outputs:
            pool.remove(v)
            available.append(v)
        index += 1

    produced = {v for spec in specs for v in spec.outputs}
    exogenous = [v for v in variables if v not in produced]

    observed = list(variables)
    if allow_hidden and rng.random() < 0.35:
        # Hide at most one variable, so the observed marginal stays informative.
        hideable = [v for v in variables if v not in exogenous] or list(variables)
        observed.remove(rng.choice(hideable))

    return RandomModel(
        seed=seed,
        variables=variables,
        observed=tuple(sorted(observed)),
        mechanisms=tuple(specs),
        exogenous_laws={v: _scalar_law(rng) for v in exogenous},
        kernels={
            spec.name: _kernel(rng, spec.inputs, spec.outputs, spec.shape) for spec in specs
        },
        fallbacks={spec.name: _fallback_policy(rng, spec.outputs) for spec in specs},
        replacements={
            spec.name: _kernel(rng, spec.inputs, spec.outputs, "positive") for spec in specs
        },
    )
