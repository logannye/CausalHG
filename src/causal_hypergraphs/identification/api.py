from __future__ import annotations

from causal_hypergraphs.expression import (
    Fallback,
    Probability,
    Product,
    Quotient,
    ReplacementFactor,
)
from causal_hypergraphs.graph import MechanismGraph

from .queries import DeleteMechanism, ReplaceMechanism
from .results import (
    Assumption,
    IdentificationResult,
    Identified,
    ProofStep,
    Unknown,
)

CORE_ASSUMPTIONS = (
    Assumption("C1", "Mechanism dependency graph is acyclic."),
    Assumption("C2", "Mechanisms have independent exogenous noise."),
    Assumption("C3", "Mechanisms use input/output role typing."),
    Assumption("C4", "Each variable has at most one producing mechanism."),
)


def identify(
    graph: MechanismGraph,
    query: DeleteMechanism | ReplaceMechanism,
    observed_variables: object | None = None,
) -> IdentificationResult:
    """Identify a mechanism-level query using the milestone-1 compiler.

    Supported complete paths:
    - T2/T3 when all variables are observed and no mechanism is marked latent.
    - T4/T4.1 when all variables are observed and latent mechanisms are present.
    - T6 when hidden variables exist but the target mechanism boundary is observed.

    Boundary-violating hidden-variable cases return Unknown with T7 guidance.
    """

    if isinstance(query, DeleteMechanism):
        return _identify_delete(graph, query, observed_variables)
    if isinstance(query, ReplaceMechanism):
        return _identify_replace(graph, query, observed_variables)
    raise TypeError(f"Unsupported query type: {type(query).__name__}")


def _observed(graph: MechanismGraph, observed_variables: object | None) -> frozenset[str]:
    if observed_variables is None:
        return graph.observed_set
    if isinstance(observed_variables, str):
        return frozenset({observed_variables})
    return frozenset(str(v) for v in observed_variables)  # type: ignore[union-attr]


def _theorem(graph: MechanismGraph, observed: frozenset[str], replacement: bool = False) -> str:
    all_observed = observed == graph.variable_set
    has_latent_mechanisms = bool(graph.latent_mechanism_names)
    if replacement:
        if all_observed and not has_latent_mechanisms:
            return "T3"
        if all_observed:
            return "T4.1"
        return "T6"
    if all_observed and not has_latent_mechanisms:
        return "T2"
    if all_observed:
        return "T4"
    return "T6"


def _unknown_boundary(graph: MechanismGraph, target: str, missing: tuple[str, ...]) -> Unknown:
    suggestions = tuple(f"Measure boundary variable {variable!r}." for variable in missing)
    return Unknown(
        reason="Target mechanism boundary contains hidden variables.",
        next_algorithm="T7 Pearl-ID reduction",
        suggestions=suggestions
        + ("Run the future T7 stochastic-intervention reduction on the bipartite ADMG.",),
        missing_variables=missing,
        assumptions=CORE_ASSUMPTIONS,
        derivation=(
            ProofStep(
                "Boundary check",
                f"{target!r} has hidden boundary variables {list(missing)}.",
            ),
            ProofStep(
                "Milestone-1 limit",
                "Only the observed-boundary T4/T6 compiler is implemented.",
            ),
        ),
    )


def _unknown_fallback(target: str, missing: tuple[str, ...]) -> Unknown:
    return Unknown(
        reason="Mechanism deletion would orphan outputs without a declared fallback policy.",
        suggestions=tuple(f"Declare fallback distribution P0({variable})." for variable in missing),
        missing_variables=missing,
        assumptions=CORE_ASSUMPTIONS,
        derivation=(ProofStep("Fallback check", f"{target!r} lacks P0 for {list(missing)}."),),
    )


def _identify_delete(
    graph: MechanismGraph,
    query: DeleteMechanism,
    observed_variables: object | None,
) -> IdentificationResult:
    target = graph.get_mechanism(query.target)
    observed = _observed(graph, observed_variables)
    missing_boundary = tuple(sorted(target.boundary - observed))
    if missing_boundary:
        return _unknown_boundary(graph, query.target, missing_boundary)

    missing_fallback = graph.missing_fallback_variables(query.target)
    if missing_fallback:
        return _unknown_fallback(query.target, missing_fallback)

    theorem = _theorem(graph, observed, replacement=False)
    numerator = Probability(tuple(sorted(observed)))
    denominator = Probability(target.outputs, given=target.inputs)
    expression = Product([Quotient(numerator, denominator), *[Fallback(v) for v in target.outputs]])
    assumptions = CORE_ASSUMPTIONS + (
        Assumption("P0", "Fallback distributions are specified for orphaned outputs."),
        Assumption("Observed boundary", "Target mechanism inputs and outputs are observed."),
    )
    derivation = (
        ProofStep("Validate graph", "C1-C4 passed during MechanismGraph construction."),
        ProofStep(
            "Read mechanism factor",
            f"P({','.join(target.outputs)} | {','.join(target.inputs)}) is observable.",
        ),
        ProofStep(
            "Replace factor",
            "Delete the target mechanism factor and multiply by fallback output factors.",
        ),
    )
    return Identified(
        expression=expression,
        theorem=theorem,
        assumptions=assumptions,
        derivation=derivation,
    )


def _identify_replace(
    graph: MechanismGraph,
    query: ReplaceMechanism,
    observed_variables: object | None,
) -> IdentificationResult:
    target = graph.get_mechanism(query.target)
    observed = _observed(graph, observed_variables)
    missing_boundary = tuple(sorted(target.boundary - observed))
    if missing_boundary:
        return _unknown_boundary(graph, query.target, missing_boundary)

    theorem = _theorem(graph, observed, replacement=True)
    numerator = Probability(tuple(sorted(observed)))
    denominator = Probability(target.outputs, given=target.inputs)
    replacement = ReplacementFactor(query.replacement, target.outputs, given=target.inputs)
    expression = Product([Quotient(numerator, denominator), replacement])
    assumptions = CORE_ASSUMPTIONS + (
        Assumption(
            "Replacement incidence",
            "Replacement mechanism has the same inputs and outputs.",
        ),
        Assumption("Observed boundary", "Target mechanism inputs and outputs are observed."),
    )
    derivation = (
        ProofStep("Validate graph", "C1-C4 passed during MechanismGraph construction."),
        ProofStep(
            "Read old factor",
            f"P({','.join(target.outputs)} | {','.join(target.inputs)}) is observable.",
        ),
        ProofStep("Swap factor", f"Replace old factor with P_{query.replacement}."),
    )
    return Identified(
        expression=expression,
        theorem=theorem,
        assumptions=assumptions,
        derivation=derivation,
    )
