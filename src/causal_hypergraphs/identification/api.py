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
from .t7 import identify_delete_via_t7

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
    allow_t7: bool = False,
) -> IdentificationResult:
    """Identify a mechanism-level query using the milestone-1 compiler.

    Supported complete paths:
    - T2/T3 when all variables are observed and no mechanism is marked latent.
    - T4/T4.1 when all variables are observed and latent mechanisms are present.
    - T6 when hidden variables exist but the target mechanism boundary is observed.

    Boundary-violating hidden-variable cases return Unknown with T7 guidance unless
    `allow_t7=True` and the experimental T7 vertical slice identifies the query.
    """

    if isinstance(query, DeleteMechanism):
        return _identify_delete(graph, query, observed_variables, allow_t7=allow_t7)
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


def _surviving_factors(graph: MechanismGraph, exclude: str) -> list[Probability]:
    """The mechanism-level chain-rule factors of P(V) that survive intervening on `exclude`.

    Lemma 1.1 factorizes P(V) into one marginal per exogenous variable and one joint
    conditional P(out(m) | in(m)) per mechanism. A mechanism-level intervention is a
    *local factor swap*: every factor except the target's is carried through unchanged.

    Returning the surviving factors explicitly — rather than dividing the full joint by
    the target factor — is what keeps the estimand well defined when the target factor is
    singular, which under C2 is the generic case for a mechanism whose noise carries fewer
    degrees of freedom than it has outputs.
    """
    factors = [Probability((variable,)) for variable in sorted(graph.exogenous_variables)]
    for name in sorted(graph.mechanisms):
        if name == exclude:
            continue
        mechanism = graph.get_mechanism(name)
        factors.append(Probability(mechanism.outputs, given=mechanism.inputs))
    return factors


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
    allow_t7: bool = False,
) -> IdentificationResult:
    target = graph.get_mechanism(query.target)
    observed = _observed(graph, observed_variables)
    missing_boundary = tuple(sorted(target.boundary - observed))
    if missing_boundary:
        if allow_t7:
            return identify_delete_via_t7(graph, query, observed_variables)
        return _unknown_boundary(graph, query.target, missing_boundary)

    missing_fallback = graph.missing_fallback_variables(query.target)
    if missing_fallback:
        return _unknown_fallback(query.target, missing_fallback)

    theorem = _theorem(graph, observed, replacement=False)
    fallbacks = [Fallback(v) for v in target.outputs]
    common = CORE_ASSUMPTIONS + (
        Assumption("P0", "Fallback distributions are specified for orphaned outputs."),
        Assumption("Observed boundary", "Target mechanism inputs and outputs are observed."),
    )
    validate_step = ProofStep("Validate graph", "C1-C4 passed during MechanismGraph construction.")
    factorize_step = ProofStep(
        "Factorize",
        "Lemma 1.1: P(V) is the product of exogenous marginals and one joint conditional "
        "P(out(m) | in(m)) per mechanism.",
    )

    if observed == graph.variable_set:
        # Every chain-rule factor is an observational quantity, so the target factor can be
        # *omitted* rather than divided out. This keeps the estimand defined where that
        # factor is singular -- the generic case under C2 for a mechanism whose noise carries
        # fewer degrees of freedom than it has outputs.
        expression = Product([*_surviving_factors(graph, exclude=query.target), *fallbacks])
        assumptions = common + (
            Assumption(
                "Downstream positivity",
                "Every surviving mechanism input configuration that the post-intervention law "
                "gives positive mass has positive observational probability, so each surviving "
                "factor P(out(m) | in(m)) is estimable there.",
            ),
        )
        derivation = (
            validate_step,
            factorize_step,
            ProofStep(
                "Omit target factor",
                f"Drop P({','.join(target.outputs)} | {','.join(target.inputs)}) from the "
                "product and multiply by the fallback output factors. No division by the "
                "target factor is performed.",
            ),
        )
        return Identified(
            expression=expression,
            theorem=theorem,
            assumptions=assumptions,
            derivation=derivation,
        )

    # Hidden variables are present. Surviving factors may reference them, so they are not
    # individually identified and the division-free form is unavailable. Because
    # boundary(m*) is observed, the target factor pulls out of the sum over hidden variables:
    #     P(O) = P(out(m*) | in(m*)) * R(O),  R(O) = sum_H prod_{m != m*} (...)
    # and P(O | delete(m*)) = prod P0(v) * R(O). R(O) is reachable only as the quotient, so
    # this route requires the target factor to be strictly positive -- an assumption the
    # full-observability branch above does not need.
    numerator = Probability(tuple(sorted(observed)))
    denominator = Probability(target.outputs, given=target.inputs)
    expression = Product([Quotient(numerator, denominator), *fallbacks])
    assumptions = common + (
        Assumption(
            "Target positivity",
            "P(out(m*) | in(m*)) > 0 wherever the post-intervention law puts mass. Not "
            "checkable from incidence; recorded as a certificate. It fails for a "
            "deterministic mechanism whose outputs are functionally coupled, in which case "
            "this quotient is 0/0 on exactly the region the intervention creates.",
        ),
    )
    derivation = (
        validate_step,
        factorize_step,
        ProofStep(
            "Pull out target factor",
            f"boundary(m*) is observed, so P({','.join(target.outputs)} | "
            f"{','.join(target.inputs)}) is constant in the hidden variables and factors out "
            "of the marginalization over them.",
        ),
        ProofStep(
            "Swap factor",
            "Divide it out of P(O) and multiply by the fallback output factors.",
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
    replacement = ReplacementFactor(query.replacement, target.outputs, given=target.inputs)
    common = CORE_ASSUMPTIONS + (
        Assumption(
            "Replacement incidence",
            "Replacement mechanism has the same inputs and outputs.",
        ),
        Assumption("Observed boundary", "Target mechanism inputs and outputs are observed."),
    )

    if observed == graph.variable_set:
        # Same argument as deletion: swap the target factor out of the chain-rule product
        # rather than dividing by it, so the estimand survives a singular target factor.
        # This is the case "replace a stoichiometrically coupled mechanism with a decoupled
        # one", where the replacement puts mass exactly where the old factor vanishes.
        expression = Product([*_surviving_factors(graph, exclude=query.target), replacement])
        assumptions = common + (
            Assumption(
                "Downstream positivity",
                "Every surviving mechanism input configuration that the post-intervention law "
                "gives positive mass has positive observational probability, so each surviving "
                "factor P(out(m) | in(m)) is estimable there.",
            ),
        )
        derivation = (
            ProofStep("Validate graph", "C1-C4 passed during MechanismGraph construction."),
            ProofStep(
                "Factorize",
                "Lemma 1.1: P(V) is the product of exogenous marginals and one joint "
                "conditional P(out(m) | in(m)) per mechanism.",
            ),
            ProofStep(
                "Swap target factor",
                f"Drop P({','.join(target.outputs)} | {','.join(target.inputs)}) from the "
                f"product and multiply by P_{query.replacement}. No division by the target "
                "factor is performed.",
            ),
        )
        return Identified(
            expression=expression,
            theorem=theorem,
            assumptions=assumptions,
            derivation=derivation,
        )

    numerator = Probability(tuple(sorted(observed)))
    denominator = Probability(target.outputs, given=target.inputs)
    expression = Product([Quotient(numerator, denominator), replacement])
    assumptions = common + (
        Assumption(
            "Target positivity",
            "P(out(m*) | in(m*)) > 0 wherever the post-intervention law puts mass. Not "
            "checkable from incidence; recorded as a certificate. It fails when the "
            "replaced mechanism is deterministic with functionally coupled outputs.",
        ),
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
