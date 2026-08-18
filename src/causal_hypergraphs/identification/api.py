from __future__ import annotations

from causal_hypergraphs.expression import (
    ConditionalExpectation,
    Expression,
    Fallback,
    Probability,
    Product,
    Quotient,
    ReplacementFactor,
    SumOut,
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


def _theorem(
    graph: MechanismGraph,
    observed: frozenset[str],
    replacement: bool = False,
    variables: frozenset[str] | None = None,
) -> str:
    all_observed = observed == (graph.variable_set if variables is None else variables)
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


def _validate_outcomes(graph: MechanismGraph, outcomes: tuple[str, ...]) -> None:
    unknown = tuple(sorted(set(outcomes) - graph.variable_set))
    if unknown:
        raise ValueError(
            f"Query outcomes are not variables of the graph: {list(unknown)}. "
            "A misspelled readout would otherwise be silently answered as a full-joint query."
        )


def ancestral_closure(graph: MechanismGraph, outcomes: object) -> frozenset[str]:
    """The variables `outcomes` can depend on: their ancestry in the mechanism graph.

    Walks backwards from each outcome through its producing mechanism, adding that
    mechanism's whole boundary -- all of `out(m)`, because the chain-rule factor is a
    *joint* over the outputs and cannot be split, and all of `in(m)`, because the factor
    conditions on them -- and recurses. Under C4 the producer is unique, so the walk is
    well defined; under C1 it terminates.

    Everything outside this set is irrelevant to `P(outcomes | do)` and can be dropped
    rather than summed, which is what makes a marginal query cheap.
    """
    wanted = _observed(graph, outcomes) if outcomes else frozenset()
    producer = {
        variable: name
        for name in graph.mechanisms
        for variable in graph.get_mechanism(name).outputs
    }
    needed: set[str] = set()
    frontier = list(wanted)
    while frontier:
        variable = frontier.pop()
        if variable in needed:
            continue
        needed.add(variable)
        name = producer.get(variable)
        if name is None:  # exogenous: nothing upstream of it
            continue
        mechanism = graph.get_mechanism(name)
        frontier.extend(set(mechanism.outputs) | set(mechanism.inputs))
    return frozenset(needed)


def _restrict_to_ancestry(
    expression: Product,
    graph: MechanismGraph,
    outcomes: tuple[str, ...],
) -> tuple[Expression, ProofStep | None]:
    """Reduce a truncated-factorization product to a marginal query on `outcomes`.

    Exact, by the ordinary ancestral argument: every retained factor is a conditional
    `P(out(m) | in(m))` that sums to one over `out(m)` at fixed `in(m)`, so summing the
    full product over the variables outside the ancestral closure collapses each of their
    factors to one and removes it. No factor outside the closure conditions on a variable
    inside it -- if it did, that variable would be in the closure -- so the elimination
    order is unconstrained.

    Only the factored form admits this. The hidden-variable branch is a quotient whose
    numerator is a single joint over all observed variables; it does not decompose, so
    the caller marginalizes it without reduction.
    """
    if not outcomes:
        return expression, None

    needed = ancestral_closure(graph, outcomes)
    retained = [
        factor for factor in expression.factors if factor.footprint() <= needed
    ]
    dropped = len(expression.factors) - len(retained)

    # If the intervention's own factor did not survive, the target mechanism cannot reach
    # the outcome at all. What remains is the ancestral factorization of the observational
    # law, which sums to exactly P(outcomes) -- so say that, rather than emitting a sum
    # over the ancestry that is guaranteed to reproduce it. The footprint drops from the
    # whole ancestry to the outcomes themselves, and the estimand states outright that the
    # answer cannot depend on what the intervention installs.
    intervenes = any(
        isinstance(factor, Fallback | ReplacementFactor) for factor in retained
    )
    if not intervenes:
        return Probability(outcomes), ProofStep(
            "Restrict to ancestry",
            f"The target mechanism is not an ancestor of {','.join(outcomes)}, so its "
            f"factor is outside the ancestral closure {sorted(needed)} and the "
            "intervention cannot reach the outcome. The remaining factors are the "
            f"ancestral factorization of the observational law, which sums to "
            f"P({','.join(outcomes)}).",
        )

    # Summed from what the retained factors actually mention, not from the closure. The
    # two normally agree, and where they differ the closure is wrong: a variable no factor
    # mentions contributes a bare `sum_v 1 = |domain(v)|`, which is a silent multiplicative
    # error and needs a domain for a variable the estimand does not otherwise use. That
    # happens exactly when an output was marginalized inside the policy factor.
    mentioned: set[str] = set()
    for factor in retained:
        mentioned.update(factor.footprint())
    summed = tuple(sorted(mentioned - set(outcomes)))
    reduced: Expression = Product(retained)
    if summed:
        reduced = SumOut(summed, reduced)

    step = ProofStep(
        "Restrict to ancestry",
        f"P({','.join(outcomes)} | do) depends only on the ancestral closure "
        f"{sorted(needed)}. Every factor outside it is a conditional summing to 1, so "
        f"{dropped} factor(s) were dropped rather than summed; "
        f"{len(summed)} variable(s) remain to marginalize.",
    )
    return reduced, step


def _contains_quotient(expression: Expression) -> bool:
    """Whether any node in the tree is a division.

    The factored identifiers are products of per-mechanism factors; the hidden-variable
    ones are quotients. Only the former can have one factor rewritten as an expectation,
    so this is the precise test -- narrower than "is the root a Product", which would also
    reject the `P(outcomes)` collapse that an unreachable intervention produces.
    """
    if isinstance(expression, Quotient):
        return True
    if isinstance(expression, Product):
        return any(_contains_quotient(factor) for factor in expression.factors)
    if isinstance(expression, SumOut):
        return _contains_quotient(expression.expression)
    return False


def _contains_intervention(expression: Expression) -> bool:
    """Whether the estimand still carries the factor the intervention installs.

    Its absence means the target mechanism cannot reach the outcome, which the ancestral
    reduction turns into a collapsed observational expression. Callers that rebuild from
    the body must branch on this rather than assume a factorized product is there.
    """
    if isinstance(expression, Fallback | ReplacementFactor):
        return True
    if isinstance(expression, Product):
        return any(_contains_intervention(factor) for factor in expression.factors)
    if isinstance(expression, SumOut):
        return _contains_intervention(expression.expression)
    if isinstance(expression, Quotient):
        return _contains_intervention(expression.numerator) or _contains_intervention(
            expression.denominator
        )
    return False


def identify_expectation(
    graph: MechanismGraph,
    query: DeleteMechanism | ReplaceMechanism,
    outcome: str,
    observed_variables: object | None = None,
) -> IdentificationResult:
    """Identify `E[outcome | do]` as a functional, without enumerating the outcome.

    The density form indexes its answer by the outcome's value, so the outcome must be a
    finite discrete variable. An expectation does not: in the truncated factorization the
    outcome appears in exactly one factor -- the one for its producing mechanism -- so

        E[Y | do] = sum over the ancestry of  (other factors) * E[Y | in(m_Y)]

    and `Y` enters only through a conditional mean. That is a regression, defined for a
    real-valued readout, and the outcome's domain is never touched.

    The co-outputs of `Y`'s mechanism drop out too. Not by assumption: a co-output that
    were an ancestor of `Y` would make `Y`'s own mechanism its own ancestor, which C1
    forbids. So no retained factor mentions one, and the joint factor marginalizes to
    `E[Y | in(m_Y)]` exactly.

    Refuses when `outcome` is produced by the target mechanism. Its post-intervention law
    is then the policy the caller supplied, so there would be nothing to estimate, and
    returning a number would misrepresent where it came from.
    """
    _validate_outcomes(graph, (outcome,))
    target = graph.get_mechanism(query.target)
    if outcome in target.outputs:
        raise ValueError(
            f"{outcome!r} is an output of the mechanism being intervened on, so its "
            f"post-intervention law is exactly the policy supplied for {query.target!r}. "
            "There is nothing to estimate from data; take the expectation of that policy "
            "directly, or ask about a downstream variable."
        )

    density = identify(graph, _with_outcomes(query, (outcome,)), observed_variables)
    if not isinstance(density, Identified):
        return density
    if _contains_quotient(density.expression):
        # A quotient does not decompose into per-mechanism factors, so there is no single
        # factor holding the outcome to fold into an expectation. This is exactly the
        # hidden-boundary case; the density form still answers it.
        raise ValueError(
            f"E[{outcome} | do] needs the factored identifier, and hidden variables forced "
            "the quotient form for this query, which has no per-mechanism factor to fold "
            "the outcome into. Use the density form instead."
        )

    if not _contains_intervention(density.expression):
        # The density form already collapsed to P(outcome): the target cannot reach it, so
        # the post-intervention expectation is the observational one. Rebuilding a weighted
        # sum from that collapsed body would silently drop the weights -- the sum over the
        # ancestry would still be there while the exogenous marginals that weight it were
        # not -- and produce an unweighted average of conditional means.
        return Identified(
            expression=ConditionalExpectation(outcome),
            theorem=density.theorem,
            assumptions=density.assumptions,
            derivation=density.derivation
            + (
                ProofStep(
                    "Take expectation",
                    f"The intervention cannot reach {outcome!r}, so its post-intervention "
                    f"law is the observational one and E[{outcome} | do] = E[{outcome}].",
                ),
            ),
        )

    producer = {
        variable: name
        for name in graph.mechanisms
        for variable in graph.get_mechanism(name).outputs
    }
    outcome_mechanism = producer.get(outcome)
    if outcome_mechanism is None:
        absorbed: tuple[str, ...] = (outcome,)
        expectation = ConditionalExpectation(outcome)
    else:
        produced_by = graph.get_mechanism(outcome_mechanism)
        absorbed = produced_by.outputs
        expectation = ConditionalExpectation(outcome, given=produced_by.inputs)

    needed = ancestral_closure(graph, (outcome,))
    body_expression = (
        density.expression.expression
        if isinstance(density.expression, SumOut)
        else density.expression
    )
    product = (
        body_expression
        if isinstance(body_expression, Product)
        else Product([body_expression])
    )
    retained = [
        factor
        for factor in product.factors
        if factor.footprint() <= (needed - set(absorbed))
    ]
    # Everything in the ancestry except the outcome's own output group. The variables the
    # expectation conditions on are summed here too: they are free *inside* the node and
    # bound by this sum, which is what makes the result a scalar.
    summed = tuple(sorted(needed - set(absorbed)))
    body: Expression = Product([*retained, expectation])
    if summed:
        body = SumOut(summed, body)

    return Identified(
        expression=body,
        theorem=density.theorem,
        assumptions=density.assumptions,
        derivation=density.derivation
        + (
            ProofStep(
                "Take expectation",
                f"{outcome!r} appears in exactly one chain-rule factor, so summing it "
                f"against that factor gives {expectation}. Its co-outputs "
                f"{sorted(set(absorbed) - {outcome})} cannot be its ancestors under C1, so "
                "they appear in no retained factor and marginalize away. The outcome's "
                "domain is never enumerated, so it may be continuous.",
            ),
        ),
    )


def _with_outcomes(
    query: DeleteMechanism | ReplaceMechanism, outcomes: tuple[str, ...]
) -> DeleteMechanism | ReplaceMechanism:
    if isinstance(query, DeleteMechanism):
        return DeleteMechanism(query.target, outcomes)
    return ReplaceMechanism(
        query.target, query.incidence or query.replacement, outcomes
    )


def _marginalize_quotient(
    expression: Product,
    observed: frozenset[str],
    outcomes: tuple[str, ...],
) -> tuple[Expression, ProofStep | None]:
    """Marginalize the hidden-variable estimand to `outcomes`, without reduction.

    The quotient form's numerator is `P(O)`, one joint over every observed variable. It
    does not factor, so there is no sub-product to keep and no factor to drop: the honest
    move is to sum over the rest and say so. The answer is correct and the *cost* is
    unchanged, which is a real limitation rather than a presentational one -- reducing it
    needs the quotient replaced by a factored identifier, which is the T7 work.
    """
    if not outcomes:
        return expression, None

    summed = tuple(sorted(observed - set(outcomes)))
    reduced: Expression = SumOut(summed, expression) if summed else expression
    step = ProofStep(
        "Restrict to ancestry",
        f"Marginalized to P({','.join(outcomes)} | do) by summing over {len(summed)} "
        "variable(s). No factor was dropped: the hidden-variable identifier is a quotient "
        "whose numerator is a single joint over all observed variables, so it admits no "
        "ancestral reduction. Evaluation cost is unchanged.",
    )
    return reduced, step


def _removal_step(removable: tuple[str, ...], target: str) -> ProofStep:
    """Record that a hidden output was summed out of the policy rather than estimated."""
    return ProofStep(
        "Marginalize the policy",
        f"{list(removable)} are outputs of {target!r} that no observable depends on, so "
        f"P0_{target} is summed over them inside the factor. The policy is declared over "
        "every output, so this is a sum over a supplied table and not an estimated "
        "quantity; the variables never enter the estimand's scope or its cost.",
    )


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
        suggestions=(
            f"Declare the joint fallback policy P0_{target}({','.join(missing)}) over the "
            "orphaned outputs.",
        ),
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
    _validate_outcomes(graph, query.outcomes)
    observed = _observed(graph, observed_variables)

    # A hidden output nothing consumes is not an obstruction. The caller supplies a joint
    # policy over every output, so its marginal over the observed ones is a sum over a
    # table already in hand -- the hidden coordinate's domain is part of the intervention,
    # not something the data has to provide. Removing it from the boundary check is what
    # turns the commonest hidden-boundary shape from a refusal into an answer.
    #
    # This is emphatically not true of a hidden output that *does* reach an observation:
    # that one is unidentifiable, and `identify_delete_via_t7` says so with a witness.
    removable = graph.removable_outputs(query.target, observed)
    effective_variables = graph.variable_set - frozenset(removable)
    if observed != effective_variables:
        # The relaxation is sound only on the division-free route. With other hidden
        # variables present the estimand becomes a quotient that *divides by* the target
        # factor P(out(m) | in(m)), and that factor is not observable when any output is
        # hidden -- removable or not. Omitting a factor and dividing by it are different
        # operations, and only the first one tolerates an unobservable output.
        removable = ()

    missing_boundary = tuple(sorted(target.boundary - observed - frozenset(removable)))
    if missing_boundary:
        if allow_t7:
            return identify_delete_via_t7(graph, query, observed_variables)
        return _unknown_boundary(graph, query.target, missing_boundary)

    missing_fallback = graph.missing_fallback_variables(query.target)
    if missing_fallback:
        return _unknown_fallback(query.target, missing_fallback)

    theorem = _theorem(graph, observed, replacement=False, variables=effective_variables)
    # One joint policy over all orphaned outputs, not one per variable: deletion orphans
    # `out(m*)` simultaneously, and a per-variable product would force them independent.
    # When every output is removable the policy is summed over its whole support, and a
    # policy is a distribution, so the factor is identically one. Dropping it rather than
    # emitting it is the same move the ancestral reduction makes on factors that sum to
    # one -- and it is what lets the estimand say outright that the deletion cannot reach
    # any observable, instead of multiplying by a 1 the reader has to verify.
    installed = tuple(sorted(set(target.outputs) - frozenset(removable)))
    fallbacks = [Fallback(query.target, installed, removable)] if installed else []
    common = CORE_ASSUMPTIONS + (
        Assumption(
            "P0",
            "A joint fallback policy P0^m(out(m)) is specified for the deleted "
            "mechanism's orphaned outputs.",
        ),
        Assumption("Observed boundary", "Target mechanism inputs and outputs are observed."),
    )
    validate_step = ProofStep("Validate graph", "C1-C4 passed during MechanismGraph construction.")
    factorize_step = ProofStep(
        "Factorize",
        "Lemma 1.1: P(V) is the product of exogenous marginals and one joint conditional "
        "P(out(m) | in(m)) per mechanism.",
    )

    if observed == effective_variables:
        # Every chain-rule factor is an observational quantity, so the target factor can be
        # *omitted* rather than divided out. This keeps the estimand defined where that
        # factor is singular -- the generic case under C2 for a mechanism whose noise carries
        # fewer degrees of freedom than it has outputs.
        product = Product([*_surviving_factors(graph, exclude=query.target), *fallbacks])
        expression, restrict_step = _restrict_to_ancestry(product, graph, query.outcomes)
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
                f"product and multiply by the joint fallback factor "
                f"P0_{query.target}({','.join(target.outputs)}). No division by the target "
                "factor is performed.",
            ),
        ) + ((restrict_step,) if restrict_step else ())
        return Identified(
            expression=expression,
            theorem=theorem,
            assumptions=assumptions,
            derivation=derivation
            + ((_removal_step(removable, query.target),) if removable else ()),
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
    expression, restrict_step = _marginalize_quotient(
        Product([Quotient(numerator, denominator), *fallbacks]), observed, query.outcomes
    )
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
            f"Divide it out of P(O) and multiply by the joint fallback factor "
            f"P0_{query.target}({','.join(target.outputs)}).",
        ),
    ) + ((restrict_step,) if restrict_step else ())
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
    _validate_outcomes(graph, query.outcomes)
    observed = _observed(graph, observed_variables)
    missing_boundary = tuple(sorted(target.boundary - observed))
    if missing_boundary:
        return _unknown_boundary(graph, query.target, missing_boundary)

    theorem = _theorem(graph, observed, replacement=True)
    replacement = ReplacementFactor(query.replacement, target.outputs, given=target.inputs)

    # do(m -> m') is defined only for rho(m') = rho(m). When the caller supplies the
    # replacement's incidence there is something to check, so check it and discharge the
    # certificate; with only a name there is not, and the assumption is the honest outcome.
    incidence_assumptions: tuple[Assumption, ...] = (
        Assumption(
            "Replacement incidence",
            "Replacement mechanism has the same inputs and outputs.",
        ),
    )
    incidence_steps: tuple[ProofStep, ...] = ()
    if query.incidence is not None:
        if (
            query.incidence.inputs != target.inputs
            or query.incidence.outputs != target.outputs
        ):
            raise ValueError(
                f"Replacement {query.replacement!r} does not preserve the typed incidence of "
                f"{query.target!r}: expected inputs {list(target.inputs)} and outputs "
                f"{list(target.outputs)}, got inputs {list(query.incidence.inputs)} and "
                f"outputs {list(query.incidence.outputs)}."
            )
        incidence_assumptions = ()
        incidence_steps = (
            ProofStep(
                "Verify replacement incidence",
                f"rho({query.replacement}) = rho({query.target}) = "
                f"({','.join(target.inputs)}) -> ({','.join(target.outputs)}).",
            ),
        )

    common = (
        CORE_ASSUMPTIONS
        + incidence_assumptions
        + (Assumption("Observed boundary", "Target mechanism inputs and outputs are observed."),)
    )

    if observed == graph.variable_set:
        # Same argument as deletion: swap the target factor out of the chain-rule product
        # rather than dividing by it, so the estimand survives a singular target factor.
        # This is the case "replace a stoichiometrically coupled mechanism with a decoupled
        # one", where the replacement puts mass exactly where the old factor vanishes.
        product = Product([*_surviving_factors(graph, exclude=query.target), replacement])
        expression, restrict_step = _restrict_to_ancestry(product, graph, query.outcomes)
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
            *incidence_steps,
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
        ) + ((restrict_step,) if restrict_step else ())
        return Identified(
            expression=expression,
            theorem=theorem,
            assumptions=assumptions,
            derivation=derivation,
        )

    numerator = Probability(tuple(sorted(observed)))
    denominator = Probability(target.outputs, given=target.inputs)
    expression, restrict_step = _marginalize_quotient(
        Product([Quotient(numerator, denominator), replacement]), observed, query.outcomes
    )
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
        *incidence_steps,
        ProofStep(
            "Read old factor",
            f"P({','.join(target.outputs)} | {','.join(target.inputs)}) is observable.",
        ),
        ProofStep("Swap factor", f"Replace old factor with P_{query.replacement}."),
    ) + ((restrict_step,) if restrict_step else ())
    return Identified(
        expression=expression,
        theorem=theorem,
        assumptions=assumptions,
        derivation=derivation,
    )
