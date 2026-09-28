"""Compile and evaluate immutable causal queries on explicitly supported model classes.

The mechanism compiler supports fully observed acyclic independent-noise models.
Observed variable interventions on hidden-variable graphs use the existing Pearl-ID
backend. Its failures do not prove failure of a mechanism-model or policy-mixture
query. Finite evaluation keeps policy axes and intervention semantics explicit.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from causal_hypergraphs.estimation import Dataset, Estimate, NotIdentified, estimate
from causal_hypergraphs.expression import (
    Expression,
    Fallback,
    Kernel,
    Probability,
    Product,
    Quotient,
    ReplacementFactor,
    SumOut,
)
from causal_hypergraphs.graph import MechanismGraph
from causal_hypergraphs.identification import (
    ADMG,
    Assumption,
    IdentificationResult,
    Identified,
    ProofStep,
    Unidentified,
    Unknown,
    identify_effect,
    latent_project_to_variable_admg,
)
from causal_hypergraphs.identification.api import CORE_ASSUMPTIONS
from causal_hypergraphs.queries import (
    AtomicIntervention,
    CausalQuery,
    Delete,
    EffectContrast,
    HardIntervention,
    JointPolicy,
    Replace,
    intervention_operations,
    validate_intervention,
)
from causal_hypergraphs.semantics import (
    DEFAULT_MAX_ENTRIES,
    Assignment,
    IntractableQuery,
    Model,
    SemanticsError,
    evaluate,
    with_aliases,
)

Point = tuple[Any, ...]


@dataclass(frozen=True)
class _Coordinate(Expression):
    variable: str

    def __str__(self) -> str:
        return self.variable

    def scope(self) -> frozenset[str]:
        return frozenset((self.variable,))

    def canonical_key(self) -> tuple[Any, ...]:
        return ("numeric-coordinate", self.variable)


@dataclass(frozen=True)
class _Difference(Expression):
    left: Expression
    right: Expression

    def __str__(self) -> str:
        return f"({self.left}) - ({self.right})"

    def to_latex(self) -> str:
        return rf"\left({self.left.to_latex()}\right)-\left({self.right.to_latex()}\right)"

    def scope(self) -> frozenset[str]:
        return self.left.scope() | self.right.scope()

    def footprint(self) -> frozenset[str]:
        return self.left.footprint() | self.right.footprint()

    def conditioned_on(self) -> frozenset[str]:
        return self.left.conditioned_on() | self.right.conditioned_on()

    def kernels(self) -> tuple[Kernel, ...]:
        return self.left.kernels() + self.right.kernels()

    def canonical_key(self) -> tuple[Any, ...]:
        return ("difference", self.left.canonical_key(), self.right.canonical_key())


@evaluate.register
def _evaluate_coordinate(expression: _Coordinate, model: Model, assignment: Assignment) -> float:
    value = assignment[expression.variable]
    if not isinstance(value, int | float) or not math.isfinite(value):
        raise SemanticsError("Finite expectation outcomes must have finite numeric values.")
    return float(value)


@evaluate.register
def _evaluate_difference(expression: _Difference, model: Model, assignment: Assignment) -> float:
    return evaluate(expression.left, model, assignment) - evaluate(
        expression.right, model, assignment
    )


@dataclass(frozen=True)
class CompiledQuery:
    """An identifying expression/refusal, original quantity, and immutable policy bindings.

    ``variables`` declares the tuple order of returned values: outcomes then baseline
    variables for distributions, baseline variables alone for expectations. A contrast
    uses the same axes as either arm. Policies use internal collision-free identifiers;
    their public names and original axis orders remain in the JointPolicy objects.
    """

    query: CausalQuery | EffectContrast
    result: IdentificationResult
    policies: Mapping[str, JointPolicy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policies", MappingProxyType(dict(self.policies)))

    @property
    def variables(self) -> tuple[str, ...]:
        query = self.query.left if isinstance(self.query, EffectContrast) else self.query
        return (() if query.kind == "expectation" else query.outcomes) + query.given


@dataclass(frozen=True)
class _QueryEstimate(Estimate):
    query: CausalQuery | EffectContrast | None = None

    def summary(self) -> str:
        lines = super().summary().splitlines()
        if self.query is not None:
            contrast = isinstance(self.query, EffectContrast)
            query = self.query.left if isinstance(self.query, EffectContrast) else self.query
            assert isinstance(query, CausalQuery)
            quantity = "mean" if query.kind == "expectation" else "distribution"
            label = f"{quantity} of {','.join(query.outcomes)}"
            if query.given:
                label += f" given {','.join(query.given)}"
            if contrast:
                label = f"effect contrast (left - right) for {label}"
            lines[0] = f"Estimate of {label} via {self.theorem}"
        return "\n".join(lines)


def _unknown(reason: str, *suggestions: str) -> Unknown:
    return Unknown(reason=reason, next_algorithm="supported-model-profile", suggestions=suggestions)


def _policy_bindings(
    operations: tuple[AtomicIntervention, ...],
    prefix: str,
) -> tuple[dict[str, JointPolicy], set[str], dict[str, Replace]]:
    bindings: dict[str, JointPolicy] = {}
    targets: set[str] = set()
    replacements: dict[str, Replace] = {}
    for index, operation in enumerate(operations):
        if isinstance(operation, Replace):
            replacements[operation.target] = operation
            continue
        if isinstance(operation, HardIntervention):
            variables = tuple(sorted(operation.assignments))
            if not variables:
                continue
            policy = JointPolicy(
                variables,
                {tuple(operation.assignments[name] for name in variables): 1.0},
                "do",
            )
        else:
            policy = operation.policy if isinstance(operation, Delete) else operation
        bindings[f"{prefix}policy_{index}"] = policy
        targets.update(policy.variables)
    return bindings, targets, replacements


def _descendants(graph: MechanismGraph | ADMG, roots: set[str]) -> set[str]:
    edges = (
        tuple(
            (source, target)
            for mechanism in graph.mechanisms.values()
            for source in mechanism.inputs
            for target in mechanism.outputs
        )
        if isinstance(graph, MechanismGraph)
        else graph.directed_edges
    )
    reached = set(roots)
    while True:
        extra = {target for source, target in edges if source in reached} - reached
        if not extra:
            return reached
        reached.update(extra)


def _mechanism_law(
    graph: MechanismGraph,
    desired: set[str],
    targets: set[str],
    replacements: Mapping[str, Replace],
    bindings: Mapping[str, JointPolicy],
) -> Identified:
    # Each record is (joint outputs, inputs, expression). Intervening on T inside
    # O retains P(O\T | parents), not P(O\T | parents,T): the shared original
    # noise is not conditioned by the value externally assigned to T.
    factors: list[tuple[set[str], set[str], Expression]] = []
    for variable in sorted(graph.exogenous_variables - targets):
        factors.append(({variable}, set(), Probability((variable,))))
    for name, mechanism in graph.mechanisms.items():
        outputs = set(mechanism.outputs) - targets
        if not outputs:
            continue
        parents = set(mechanism.inputs)
        expression = (
            ReplacementFactor(replacements[name].replacement, outputs, parents)
            if name in replacements
            else Probability(outputs, parents)
        )
        factors.append((outputs, parents, expression))
    for name, policy in bindings.items():
        factors.append((set(policy.variables), set(), Fallback(name, policy.variables)))

    needed = set(desired)
    selected: set[int] = set()
    while True:
        additions = {
            index
            for index, (outputs, _, _) in enumerate(factors)
            if outputs & needed and index not in selected
        }
        if not additions:
            break
        selected.update(additions)
        for index in additions:
            outputs, parents, _ = factors[index]
            needed.update(outputs | parents)
    product = Product([factors[index][2] for index in sorted(selected)])
    expression = SumOut(product.scope() - desired, product)
    assumptions = CORE_ASSUMPTIONS + (
        Assumption("Downstream positivity", "Every observational conditional used is defined."),
    )
    if replacements:
        assumptions += (
            Assumption(
                "Replacement kernels",
                "Each supplied replacement kernel is normalized and has its target's "
                "input/output axes.",
            ),
        )
    return Identified(
        expression=expression,
        theorem="acyclic-mechanism-intervention",
        assumptions=assumptions,
        derivation=(
            ProofStep(
                "Validate profile", "Fully observed, acyclic, single-producer mechanism graph."
            ),
            ProofStep(
                "Intervene", "Marginalize replaced coordinates within each original joint kernel."
            ),
            ProofStep(
                "Install policies", "Joint policies are independent of original exogenous noise."
            ),
            ProofStep(
                "Marginalize", "Retain the post-intervention ancestral factors of the query."
            ),
        ),
    )


def _projected_law(
    graph: MechanismGraph | ADMG,
    query: CausalQuery,
    desired: set[str],
    targets: set[str],
    replacements: Mapping[str, Replace],
    bindings: Mapping[str, JointPolicy],
    operations: tuple[AtomicIntervention, ...],
) -> IdentificationResult:
    if replacements:
        return _unknown(
            "Replacement with hidden variables is outside this compiler's supported scope."
        )
    admg = latent_project_to_variable_admg(graph) if isinstance(graph, MechanismGraph) else graph
    if not targets <= admg.node_set or not desired <= admg.node_set:
        return _unknown(
            "Projected identification requires observed intervention and outcome variables."
        )
    if targets & desired:
        return _unknown(
            "The projected backend does not yet support outcomes overlapping interventions."
        )
    result = identify_effect(admg, desired, targets)
    if not isinstance(result, Identified):
        plain_hard = isinstance(graph, ADMG) and all(
            isinstance(operation, HardIntervention) for operation in operations
        )
        if isinstance(result, Unidentified) and plain_hard:
            return result
        return Unknown(
            reason="The projected variable-effect query failed; this does not refute the "
            "mechanism-model class or its policy mixture. " + getattr(result, "reason", ""),
            next_algorithm="mechanism-policy-identification",
            assumptions=getattr(result, "assumptions", ()),
            derivation=getattr(result, "derivation", ()),
        )
    product = Product(
        [
            result.expression,
            *(Fallback(name, policy.variables) for name, policy in bindings.items()),
        ]
    )
    return replace(
        result,
        expression=SumOut(targets, product),
        assumptions=result.assumptions
        + (
            Assumption(
                "Backend positivity",
                "All conditional kernels in the identifying formula are defined.",
            ),
        ),
        derivation=result.derivation
        + (
            ProofStep(
                "Policy mixture",
                "Integrate the identified variable intervention against joint policies.",
            ),
        ),
    )


def _compile(
    graph: MechanismGraph | ADMG,
    query: CausalQuery | EffectContrast,
    prefix: str,
) -> CompiledQuery:
    if isinstance(query, EffectContrast):
        left = _compile(graph, query.left, prefix + "left_")
        right = _compile(graph, query.right, prefix + "right_")
        if not isinstance(left.result, Identified) or not isinstance(right.result, Identified):
            return CompiledQuery(
                query,
                _unknown(
                    "Both contrast arms must be identified; failure of an arm alone does not prove "
                    "that their difference is unidentified.",
                    str(getattr(left.result, "reason", "Left arm identified.")),
                    str(getattr(right.result, "reason", "Right arm identified.")),
                ),
            )
        aliases = dict(left.result.aliases)
        if any(
            name in aliases and aliases[name] != base for name, base in right.result.aliases.items()
        ):
            return CompiledQuery(
                query, _unknown("Contrast arms introduce conflicting variable aliases.")
            )
        aliases.update(right.result.aliases)
        result = Identified(
            _Difference(left.result.expression, right.result.expression),
            "effect-contrast",
            tuple(dict.fromkeys(left.result.assumptions + right.result.assumptions)),
            left.result.derivation
            + right.result.derivation
            + (
                ProofStep(
                    "Contrast", "Subtract the two functionals on the same observational law."
                ),
            ),
            aliases=aliases,
        )
        return CompiledQuery(query, result, {**left.policies, **right.policies})
    if not isinstance(query, CausalQuery):
        raise TypeError("query must be a CausalQuery or EffectContrast.")
    variables = graph.variable_set if isinstance(graph, MechanismGraph) else graph.node_set
    desired = set(query.outcomes) | set(query.given)
    if not desired <= variables:
        raise ValueError(f"Query references unknown variables: {sorted(desired - variables)}.")
    if isinstance(graph, MechanismGraph):
        operations = validate_intervention(graph, query.intervention)
    else:
        operations = intervention_operations(query.intervention)
        if any(isinstance(operation, Delete | Replace) for operation in operations):
            return CompiledQuery(query, _unknown("Mechanism operations require a MechanismGraph."))
    bindings, targets, replacements = _policy_bindings(operations, prefix)
    if not targets <= variables:
        raise ValueError(
            f"Intervention references unknown variables: {sorted(targets - variables)}."
        )
    affected = set(targets)
    if isinstance(graph, MechanismGraph):
        for name in replacements:
            affected.update(graph.get_mechanism(name).outputs)
    if set(query.given) & _descendants(graph, affected):
        return CompiledQuery(
            query,
            _unknown(
                "Baseline conditioning variables must not be intervened on or descendants of the "
                "intervention in the original graph.",
            ),
        )
    if isinstance(graph, MechanismGraph) and not graph.is_mechanism_acyclic():
        return CompiledQuery(
            query,
            _unknown(
                "Unified intervention compilation currently requires an acyclic mechanism graph.",
                "The legacy identify API supports some queries with an acyclic local closure.",
            ),
        )
    if isinstance(graph, MechanismGraph) and not graph.hidden_variables:
        result = _mechanism_law(graph, desired, targets, replacements, bindings)
    else:
        result = _projected_law(graph, query, desired, targets, replacements, bindings, operations)
    if not isinstance(result, Identified):
        return CompiledQuery(query, result)
    expression = result.expression
    if query.given:
        expression = Quotient(expression, SumOut(query.outcomes, expression))
        result = replace(
            result,
            assumptions=result.assumptions
            + (
                Assumption(
                    "Backend positivity",
                    "Each requested baseline conditioning event has positive mass.",
                ),
            ),
        )
    if query.kind == "expectation":
        outcome = query.outcomes[0]
        expression = SumOut((outcome,), Product((_Coordinate(outcome), expression)))
    result = replace(result, expression=expression)
    used = {
        kernel.label.removeprefix("P0_")
        for kernel in expression.kernels()
        if kernel.kind == "fallback"
    }
    return CompiledQuery(
        query, result, {name: policy for name, policy in bindings.items() if name in used}
    )


def compile_query(
    graph: MechanismGraph | ADMG,
    query: CausalQuery | EffectContrast,
) -> CompiledQuery:
    """Compile a supported distribution, finite mean, or difference of matching quantities.

    Unsupported semantic cases yield ``Unknown``. Malformed requests raise a named
    Python validation error. A plain ADMG hard intervention may return ``Unidentified``
    with a Pearl hedge; that verdict is not generalized to mechanism policy mixtures.
    """
    if not isinstance(graph, MechanismGraph | ADMG):
        raise TypeError("Compile a causal MechanismGraph or ADMG, not an untyped hypergraph.")
    return _compile(graph, query, "query_")


def _identified(compiled: CompiledQuery) -> Identified:
    if not isinstance(compiled.result, Identified):
        raise NotIdentified(getattr(compiled.result, "reason", "Query was not identified."))
    return compiled.result


def _validate_policy_domains(
    policies: Mapping[str, JointPolicy],
    domains: Mapping[str, tuple[Any, ...]],
) -> None:
    for policy in policies.values():
        missing = set(policy.variables) - set(domains)
        if missing:
            raise SemanticsError(
                f"No finite domain supplied for policy variables {sorted(missing)}."
            )
        for values, mass in policy.probabilities.items():
            if mass and any(
                value not in domains[variable]
                for variable, value in zip(policy.variables, values, strict=True)
            ):
                raise SemanticsError(
                    f"Policy {policy.name!r} puts positive mass outside declared finite domains. "
                    "Declare the domain explicitly; observation-free strata remain unsupported."
                )


def _policy_tables(
    policies: Mapping[str, JointPolicy],
    domains: Mapping[str, tuple[Any, ...]],
    limit: int,
) -> dict[str, dict[Point, float]]:
    _validate_policy_domains(policies, domains)
    tables = {}
    for name, policy in policies.items():
        axes = tuple(sorted(policy.variables))
        entries = math.prod(len(domains[axis]) for axis in axes)
        if entries > limit:
            raise IntractableQuery(
                f"Policy completion needs {entries} cells, above limit {limit}.",
                bucket=axes,
                entries=entries,
                limit=limit,
            )
        tables[name] = {
            values: policy.probability(dict(zip(axes, values, strict=True)))
            for values in itertools.product(*(domains[variable] for variable in axes))
        }
    return tables


class _PolicyModel:
    def __init__(self, inner: Model, policies: Mapping[str, JointPolicy]) -> None:
        self.inner = inner
        self.policies = policies
        self._validated_replacements: set[tuple] = set()
        _validate_policy_domains(policies, inner.domains)

    @property
    def domains(self) -> Mapping[str, tuple[Any, ...]]:
        return self.inner.domains

    def conditional(
        self, variables: Sequence[str], given: Sequence[str], assignment: Assignment
    ) -> float:
        return self.inner.conditional(variables, given, assignment)

    def conditional_expectation(
        self,
        target: str,
        given: Sequence[str],
        assignment: Assignment,
    ) -> float:
        return self.inner.conditional_expectation(target, given, assignment)

    def replacement(
        self,
        mechanism: str,
        variables: Sequence[str],
        given: Sequence[str],
        assignment: Assignment,
    ) -> float:
        key = (mechanism, tuple(variables), tuple(given), tuple(assignment[v] for v in given))
        if key not in self._validated_replacements:
            masses = []
            for values in itertools.product(*(self.domains[name] for name in variables)):
                extended = {**assignment, **dict(zip(variables, values, strict=True))}
                mass = self.inner.replacement(mechanism, variables, given, extended)
                if not math.isfinite(mass) or mass < 0:
                    raise SemanticsError("Replacement kernels must have finite nonnegative masses.")
                masses.append(mass)
            if not math.isclose(math.fsum(masses), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise SemanticsError("Replacement kernels must normalize for every input stratum.")
            self._validated_replacements.add(key)
        return self.inner.replacement(mechanism, variables, given, assignment)

    def fallback(
        self,
        mechanism: str,
        variables: Sequence[str],
        assignment: Assignment,
        marginalized: Sequence[str] = (),
    ) -> float:
        if mechanism not in self.policies:
            return self.inner.fallback(mechanism, variables, assignment, marginalized)
        policy = self.policies[mechanism]
        if set(variables) | set(marginalized) != set(policy.variables):
            raise SemanticsError("Compiled fallback axes do not match their bound joint policy.")
        return math.fsum(
            mass
            for values, mass in policy.probabilities.items()
            if all(values[policy.variables.index(name)] == assignment[name] for name in variables)
        )


def _finite_budget(expression: Expression, model: Model, limit: int) -> None:
    """Conservative reference-evaluation budget; never report an opaque contrast as width zero."""
    if limit < 1:
        raise ValueError("max_entries must be positive.")
    variables = tuple(sorted(expression.footprint()))
    try:
        entries = math.prod(len(model.domains[name]) for name in variables)
    except KeyError as exc:
        raise SemanticsError(f"Missing finite domain for {exc.args[0]!r}.") from exc
    if entries > limit:
        raise IntractableQuery(
            f"Finite query evaluation needs a {entries}-assignment footprint, above limit {limit}.",
            bucket=variables,
            entries=entries,
            limit=limit,
        )


def _has_opaque_integration(expression: Expression) -> bool:
    """Whether the elimination spine hides sums inside indivisible leaf factors.

    The legacy planner peels a leading SumOut and the outer Product only. A
    conditional ratio or a mean's inner nuisance sum is evaluated by recursive
    enumeration, so its free-variable table size does not bound its real work.
    """
    if isinstance(expression, SumOut):
        return _has_opaque_integration(expression.expression)
    if isinstance(expression, Product):
        return any(factor.footprint() != factor.scope() for factor in expression.factors)
    return expression.footprint() != expression.scope()


def evaluate_query(
    compiled: CompiledQuery,
    model: Model,
    *,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> dict[Point, float]:
    """Evaluate a finite query in declared axis order; means without baselines use key ``()``.

    This reference path enumerates finite domains. Continuous conditioning and continuous
    outcome integration require a different backend; no implicit binning occurs here.
    """
    result = _identified(compiled)
    bound = with_aliases(_PolicyModel(model, compiled.policies), result.aliases)
    _finite_budget(result.expression, bound, max_entries)
    return {
        values: evaluate(
            result.expression, bound, dict(zip(compiled.variables, values, strict=True))
        )
        for values in itertools.product(*(bound.domains[name] for name in compiled.variables))
    }


def estimate_query(compiled: CompiledQuery, data: Dataset, **kwargs: Any) -> Estimate:
    """Estimate the compiled functional with the existing empirical backend and unit bootstrap.

    Contrast arms are evaluated on the *same* bootstrap replicate, so intervals are for
    the difference itself. This is a finite-outcome mean backend. Sparse declared policies
    are completed by zero masses over the data's explicit domain, never by extrapolation.
    Returned values/intervals follow ``compiled.variables``; other diagnostics are retained.
    """
    result = _identified(compiled)
    if "fallbacks" in kwargs:
        raise ValueError(
            "Policies are bound by compile_query and cannot be overridden at estimation."
        )
    limit = kwargs.get("max_entries", DEFAULT_MAX_ENTRIES)
    if limit < 1:
        raise ValueError("max_entries must be positive.")
    tables = _policy_tables(compiled.policies, data.domains, limit)
    if kwargs.get("replacements") is not None:
        from causal_hypergraphs.estimation.empirical import EmpiricalModel

        checked = _PolicyModel(
            EmpiricalModel(data, data.variables, replacements=kwargs["replacements"]),
            {},
        )
        for kernel in result.expression.kernels():
            if kernel.kind != "replacement":
                continue
            axes = kernel.variables + kernel.given
            if any(name not in data.domains for name in axes):
                raise SemanticsError("Replacement axes require explicitly declared finite domains.")
            entries = math.prod(len(data.domains[name]) for name in axes)
            if entries > limit:
                raise IntractableQuery(
                    f"Replacement validation needs {entries} cells, above limit {limit}.",
                    bucket=axes,
                    entries=entries,
                    limit=limit,
                )
            outputs = {name: data.domains[name][0] for name in kernel.variables}
            for values in itertools.product(*(data.domains[name] for name in kernel.given)):
                assignment = {**outputs, **dict(zip(kernel.given, values, strict=True))}
                checked.replacement(kernel.label, kernel.variables, kernel.given, assignment)
    # Nested means, conditional ratios, and differences can be opaque to the old
    # elimination planner. Their leaf table size omits inner enumeration work.
    # Preserve efficient flat sum-product elimination, but bound reference work
    # before entering any opaque evaluator and suppress misleading plan metadata.
    opaque = _has_opaque_integration(result.expression)
    if isinstance(compiled.query, EffectContrast) or opaque:
        from causal_hypergraphs.estimation.empirical import EmpiricalModel

        inner = EmpiricalModel(data, data.variables)
        _finite_budget(
            result.expression,
            with_aliases(inner, result.aliases),
            kwargs.get("max_entries", DEFAULT_MAX_ENTRIES),
        )
    estimated = estimate(result, data, fallbacks=tables, **kwargs)
    positions = tuple(estimated.variables.index(name) for name in compiled.variables)
    ordered = replace(
        estimated,
        variables=compiled.variables,
        values={
            tuple(point[index] for index in positions): value
            for point, value in estimated.values.items()
        },
        interval={
            tuple(point[index] for index in positions): value
            for point, value in estimated.interval.items()
        },
        plan=None if isinstance(compiled.query, EffectContrast) or opaque else estimated.plan,
    )
    return _QueryEstimate(**vars(ordered), query=compiled.query)


__all__ = ["CompiledQuery", "compile_query", "evaluate_query", "estimate_query"]
