from __future__ import annotations

from dataclasses import dataclass

from causal_hypergraphs.expression import Fallback, Probability, Product, SumOut
from causal_hypergraphs.graph import MechanismGraph

from .pearl_id import ADMG, PearlIDBackend, PearlIDQuery
from .queries import DeleteMechanism, ReplaceMechanism
from .results import (
    Assumption,
    IdentificationResult,
    Identified,
    ProofStep,
    Unidentified,
    Unknown,
)


def variable_node(variable: str) -> str:
    return f"V:{variable}"


def mechanism_node(mechanism: str) -> str:
    return f"M:{mechanism}"


@dataclass(frozen=True)
class BipartiteDAG:
    """Typed bipartite DAG before hidden-node projection."""

    observed_nodes: tuple[str, ...]
    hidden_nodes: tuple[str, ...]
    directed_edges: tuple[tuple[str, str], ...]

    def __init__(
        self,
        observed_nodes: object,
        hidden_nodes: object = (),
        directed_edges: object = (),
    ) -> None:
        object.__setattr__(self, "observed_nodes", _ordered(observed_nodes))
        object.__setattr__(self, "hidden_nodes", _ordered(hidden_nodes))
        edges = tuple(sorted((str(a), str(b)) for a, b in directed_edges))  # type: ignore[arg-type]
        object.__setattr__(self, "directed_edges", edges)

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset(self.observed_nodes) | frozenset(self.hidden_nodes)

    def children(self, node: str) -> frozenset[str]:
        return frozenset(target for source, target in self.directed_edges if source == node)


@dataclass(frozen=True)
class BipartiteADMG:
    """Interface placeholder for the latent-projected bipartite ADMG used by T7."""

    observed_nodes: tuple[str, ...] = ()
    hidden_nodes: tuple[str, ...] = ()
    directed_edges: tuple[tuple[str, str], ...] = ()
    bidirected_edges: tuple[tuple[str, str], ...] = ()

    def __init__(
        self,
        observed_nodes: object = (),
        hidden_nodes: object = (),
        directed_edges: object = (),
        bidirected_edges: object = (),
    ) -> None:
        object.__setattr__(self, "observed_nodes", _ordered(observed_nodes))
        object.__setattr__(self, "hidden_nodes", _ordered(hidden_nodes))
        directed = tuple(
            sorted((str(a), str(b)) for a, b in directed_edges)  # type: ignore[arg-type]
        )
        bidirected = tuple(
            sorted({_bidirected_edge(edge) for edge in bidirected_edges})  # type: ignore[arg-type]
        )
        object.__setattr__(self, "directed_edges", directed)
        object.__setattr__(self, "bidirected_edges", bidirected)

    def to_admg(self) -> ADMG:
        return ADMG(
            nodes=self.observed_nodes,
            directed_edges=self.directed_edges,
            bidirected_edges=self.bidirected_edges,
        )


@dataclass(frozen=True)
class LatentProjectionPlan:
    """Planned transform from a typed mechanism graph to a Pearl-style ADMG."""

    source_graph: object
    target_admg: BipartiteADMG | None = None
    status: str = "not_implemented"


@dataclass(frozen=True)
class StochasticInterventionReduction:
    """Planned reduction of mechanism deletion/replacement to Pearl ID inputs."""

    target_outputs: tuple[str, ...]
    conditioning_inputs: tuple[str, ...]
    target_mechanism: str = ""
    query_type: str = ""
    replacement_kernel: str | None = None
    admg: BipartiteADMG | None = None
    variable_admg: ADMG | None = None
    pearl_outcomes: tuple[str, ...] = ()
    pearl_interventions: tuple[str, ...] = ()
    status: str = "reduced"

    def __init__(
        self,
        target_outputs: object,
        conditioning_inputs: object,
        replacement_kernel: str | None = None,
        target_mechanism: str = "",
        query_type: str = "",
        admg: BipartiteADMG | None = None,
        variable_admg: ADMG | None = None,
        pearl_outcomes: object = (),
        pearl_interventions: object = (),
        status: str = "reduced",
    ) -> None:
        object.__setattr__(self, "target_mechanism", str(target_mechanism))
        object.__setattr__(self, "query_type", str(query_type))
        object.__setattr__(self, "target_outputs", _ordered(target_outputs))
        object.__setattr__(self, "conditioning_inputs", _ordered(conditioning_inputs))
        object.__setattr__(self, "replacement_kernel", replacement_kernel)
        object.__setattr__(self, "admg", admg)
        object.__setattr__(self, "variable_admg", variable_admg)
        object.__setattr__(self, "pearl_outcomes", _ordered(pearl_outcomes))
        object.__setattr__(self, "pearl_interventions", _ordered(pearl_interventions))
        object.__setattr__(self, "status", str(status))


@dataclass(frozen=True)
class RelabellingWitness:
    """Why deleting a mechanism with a hidden output identifies nothing.

    `delete(m)` installs a policy over the *values* of `out(m)`. When one of those is a
    variable the data never records, no observable pins down which value is which:
    permuting a hidden variable's labels leaves every observed distribution exactly as it
    was, and moves a policy defined on those labels. Two models compatible with the same
    graph then agree on everything measurable and disagree on the answer, which is what
    non-identifiability means.

    The witness carries the hidden outputs being permuted and the nearest observed
    variables they reach -- the reach is what makes the permutation observable *after* the
    intervention while remaining invisible before it. With no such variable the deletion
    cannot move any observable at all, and that case is identified rather than refused.
    """

    hidden_outputs: tuple[str, ...]
    observed_descendants: tuple[str, ...]
    explanation: str

    def __str__(self) -> str:
        return (
            f"relabelling {list(self.hidden_outputs)} preserves every observed "
            f"distribution and changes the post-deletion law of "
            f"{list(self.observed_descendants)}"
        )


@dataclass(frozen=True)
class HedgeWitness:
    """Pearl hedge witness returned by future non-identification paths."""

    districts: tuple[tuple[str, ...], ...] = ()
    explanation: str = ""


@dataclass(frozen=True)
class HyperHedgeWitness:
    """Mechanism-level lift of a Pearl hedge witness."""

    mechanisms: tuple[str, ...] = ()
    pearl_witness: HedgeWitness | None = None
    explanation: str = ""


@dataclass(frozen=True)
class T7ReductionPlaceholder:
    """Future hook for Pearl-ID reduction of boundary-violating mechanism queries."""

    reason: str = "T7 Pearl-ID reduction is not implemented in milestone 1."
    admg: BipartiteADMG | None = None
    stochastic_intervention: StochasticInterventionReduction | None = None


def _ordered(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(sorted(str(v) for v in values))


def _bidirected_edge(edge: tuple[str, str]) -> tuple[str, str]:
    left, right = edge
    if left == right:
        raise ValueError("Bidirected self-edges are not allowed.")
    return tuple(sorted((str(left), str(right))))  # type: ignore[return-value]


def build_bipartite_dag(graph: MechanismGraph) -> BipartiteDAG:
    hidden_nodes = {variable_node(v) for v in graph.hidden_variables}
    hidden_nodes.update(mechanism_node(m) for m in graph.latent_mechanism_names)

    observed_nodes = {variable_node(v) for v in graph.variable_set - graph.hidden_variables}
    observed_nodes.update(
        mechanism_node(name)
        for name in graph.mechanisms
        if name not in graph.latent_mechanism_names
    )

    directed_edges: set[tuple[str, str]] = set()
    for name, mechanism in graph.mechanisms.items():
        m_node = mechanism_node(name)
        for variable in mechanism.inputs:
            directed_edges.add((variable_node(variable), m_node))
        for variable in mechanism.outputs:
            directed_edges.add((m_node, variable_node(variable)))

    return BipartiteDAG(
        observed_nodes=observed_nodes,
        hidden_nodes=hidden_nodes,
        directed_edges=directed_edges,
    )


def _observed_reachable_through_hidden(
    dag: BipartiteDAG,
    source: str,
    observed: frozenset[str],
    hidden: frozenset[str],
) -> frozenset[str]:
    reached: set[str] = set()
    stack = list(dag.children(source))
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        if current in observed:
            reached.add(current)
            continue
        if current in hidden:
            stack.extend(dag.children(current))
    return frozenset(reached)


def latent_project(graph: MechanismGraph) -> BipartiteADMG:
    dag = build_bipartite_dag(graph)
    observed = frozenset(dag.observed_nodes)
    hidden = frozenset(dag.hidden_nodes)

    directed_edges: set[tuple[str, str]] = set()
    for source in observed:
        for target in _observed_reachable_through_hidden(dag, source, observed, hidden):
            if source != target:
                directed_edges.add((source, target))

    bidirected_edges: set[tuple[str, str]] = set()
    for hidden_node in hidden:
        descendants = sorted(_observed_reachable_through_hidden(dag, hidden_node, observed, hidden))
        for index, left in enumerate(descendants):
            for right in descendants[index + 1 :]:
                bidirected_edges.add(_bidirected_edge((left, right)))

    return BipartiteADMG(
        observed_nodes=observed,
        hidden_nodes=hidden,
        directed_edges=directed_edges,
        bidirected_edges=bidirected_edges,
    )


def latent_project_to_variable_admg(graph: MechanismGraph) -> ADMG:
    """Project a mechanism graph to a Pearl ADMG over its observed variables.

    This is the standard latent projection (Pearl 2009 section 3.7), applied to the
    bipartite blowup: over the observed variables, put `A -> B` when a directed path from
    `A` to `B` has only latent interior nodes, and `A <-> B` when a latent node is a common
    cause of both along paths with only latent interiors.

    What makes the hypergraph case its own thing is *which* nodes are latent. A mechanism
    is never observed -- data records variables, not the processes that produced them --
    so every mechanism's noise `u_m` is a latent common parent of all of `out(m)`. Its
    outputs are therefore confounded with each other. That is not an extra modelling
    assumption bolted on here; it is the content of "outputs are produced jointly from one
    shared noise", and it is what a variable-level DAG cannot express.

    `THEOREM_T4_T5.md` Proposition T4.0 is the specification and the test: with no hidden
    variables the districts of this graph must be exactly `{out(m)}` together with the
    exogenous singletons. Its proof is the algorithm --

        "Bidirected edges arise only by projecting out a mechanism noise `u_m`, whose
        children are exactly `out(m)`; projection therefore yields a complete bidirected
        component on `out(m)` and no bidirected edge with any endpoint outside it."

    -- so the bidirected cliques are one per mechanism, each over the observed closure of
    that mechanism's outputs, plus one per *exogenous* hidden variable, which has no
    producing mechanism and so is a latent source in its own right. A hidden variable that
    *is* produced needs no clique of its own: its noise is its producer's, and its closure
    is already inside that mechanism's clique.

    The `latent` flag on a mechanism plays no part. A mechanism node is unobserved either
    way; the flag records that its functional form is unknown, which is a question about
    estimation rather than about the graph.
    """
    observed = graph.observed_set
    produced = graph.produced_variables

    consumers = graph.consumers()

    directed_edges: set[tuple[str, str]] = set()
    for source in sorted(observed):
        for name in consumers.get(source, ()):
            outputs = graph.get_mechanism(name).outputs
            for target in graph.observed_closure(outputs):
                if target != source:
                    directed_edges.add((source, target))

    # One clique per latent source. Every mechanism is one, because `u_m` is never
    # observed; an exogenous hidden variable is one, because nothing produces it.
    cliques = [
        graph.observed_closure(graph.get_mechanism(name).outputs)
        for name in sorted(graph.mechanisms)
    ]
    cliques += [
        graph.observed_closure((variable,))
        for variable in sorted(graph.variable_set - observed - produced)
    ]

    bidirected_edges: set[tuple[str, str]] = set()
    for clique in cliques:
        members = sorted(clique)
        for index, left in enumerate(members):
            for right in members[index + 1 :]:
                bidirected_edges.add(_bidirected_edge((left, right)))

    return ADMG(
        nodes=observed,
        directed_edges=directed_edges,
        bidirected_edges=bidirected_edges,
    )


def reduce_mechanism_query_to_stochastic_intervention(
    graph: MechanismGraph,
    query: DeleteMechanism | ReplaceMechanism,
) -> StochasticInterventionReduction:
    target = graph.get_mechanism(query.target)
    replacement_kernel = None
    query_type = "delete"
    if isinstance(query, ReplaceMechanism):
        query_type = "replace"
        replacement_kernel = f"P_{query.replacement}"
    variable_admg = latent_project_to_variable_admg(graph)
    return StochasticInterventionReduction(
        target_mechanism=query.target,
        query_type=query_type,
        target_outputs=target.outputs,
        conditioning_inputs=target.inputs,
        replacement_kernel=replacement_kernel,
        admg=latent_project(graph),
        variable_admg=variable_admg,
        pearl_outcomes=query.outcomes if isinstance(query, DeleteMechanism) else (),
        pearl_interventions=target.outputs,
    )


CORE_T7_ASSUMPTIONS = (
    Assumption("C1", "Mechanism dependency graph is acyclic."),
    Assumption("C2", "Mechanisms have independent exogenous noise."),
    Assumption("C4", "Each variable has at most one producing mechanism."),
)

T7_ASSUMPTIONS = (
    Assumption("T7 reduction", "Boundary-violating mechanism query is reduced to Pearl ID."),
    Assumption("Stochastic deletion", "Mechanism deletion inserts fallback factors for outputs."),
)


def _hidden_output_verdict(
    graph: MechanismGraph,
    query: DeleteMechanism,
    hidden_outputs: tuple[str, ...],
    missing_boundary: tuple[str, ...],
) -> IdentificationResult:
    """Settle `delete(m)` when some of `out(m)` is never observed.

    Three outcomes, and none of them is "not implemented yet". Returning `Unknown` here --
    which is what the T7 path did before, on the occasions it did not raise outright --
    points the reader at an algorithm that cannot exist.

    Two different reach questions are involved, and using one for the other is a genuine
    error rather than a conservative approximation:

    - *which hidden outputs can move an observation* decides the refusal. Those are the
      ones whose relabelling is visible after the intervention and invisible before it, so
      they are what the witness rests on. A hidden output that reaches nothing observed
      cannot witness anything: permuting it changes no observable either way.
    - *what the deletion can move at all* ranges over the whole of `out(m)`. A mechanism's
      observed outputs are reset by the policy too, so a hidden dead-end sibling does not
      make the deletion invisible. Answering `P(outcomes)` there is not cautious but wrong:
      the observational law is precisely what the intervention changes.
    """
    target = graph.get_mechanism(query.target)

    # A hidden output witnesses non-identifiability only if relabelling it is a legal
    # transformation of a model over this graph. `output_equalities` declares that a
    # group of outputs are functionally equal, and a group containing an *observed*
    # member pins the hidden one's labels to it -- the permutation is then no longer
    # available, and a witness that cannot be constructed does not support the strongest
    # verdict the library has.
    pinned = frozenset(
        name
        for group in target.output_equalities
        if set(group) & graph.observed_set
        for name in group
    )
    witnessing = tuple(
        variable
        for variable in hidden_outputs
        if graph.observed_closure((variable,)) and variable not in pinned
    )
    withdrawn = tuple(
        variable
        for variable in hidden_outputs
        if graph.observed_closure((variable,)) and variable in pinned
    )
    reached = tuple(sorted(graph.observed_closure(witnessing)))
    moved = tuple(sorted(graph.observed_closure(target.outputs)))

    boundary_step = ProofStep(
        "Boundary check", f"Hidden boundary variables: {list(missing_boundary)}."
    )

    if witnessing:
        return Unidentified(
            reason=(
                f"Deleting {query.target!r} installs a policy over {list(witnessing)}, "
                "which are never observed. Relabelling them leaves every observed "
                "distribution unchanged and changes the policy, so no formula in the "
                "observed law can answer this."
            ),
            witness=RelabellingWitness(
                hidden_outputs=witnessing,
                observed_descendants=reached,
                explanation=(
                    "Two models agreeing on P(V_observed) and differing only by a "
                    f"permutation of {list(witnessing)} give different post-deletion laws "
                    f"for {list(reached)}. The argument needs the policy to distinguish "
                    "the labels it permutes: a permutation-invariant policy, a uniform one "
                    "for instance, is not covered, and `identify` never sees the policy's "
                    "values, so this answers for a general policy."
                ),
            ),
            assumptions=CORE_T7_ASSUMPTIONS,
            derivation=(
                boundary_step,
                ProofStep(
                    "Reach check",
                    f"Hidden output(s) {list(witnessing)} reach observed {list(reached)}.",
                ),
                ProofStep(
                    "Relabelling witness",
                    "A permutation of the hidden output is a symmetry of the observed law "
                    "and not of the intervention policy.",
                ),
            ),
        )

    if withdrawn:
        return Unknown(
            reason=(
                f"{list(withdrawn)} are hidden outputs of {query.target!r} that reach an "
                "observation, but they are declared functionally equal to an observed "
                "output, which pins their labels. The relabelling argument that refutes "
                "identifiability elsewhere is unavailable here, and this compiler has no "
                "other route, so the case is open rather than settled."
            ),
            next_algorithm="Identification under declared determinism.",
            suggestions=(
                f"Drop the equality group over {list(withdrawn)} if it was declared "
                "loosely; it is what makes this case neither refutable nor identified.",
            ),
            missing_variables=withdrawn,
            assumptions=CORE_T7_ASSUMPTIONS,
            derivation=(
                boundary_step,
                ProofStep(
                    "Witness check",
                    f"Relabelling {list(withdrawn)} is blocked by the declared equality "
                    f"group(s) {[list(g) for g in target.output_equalities]}.",
                ),
            ),
        )

    if not moved:
        return Identified(
            expression=Probability(query.outcomes),
            theorem="T7 (unreachable intervention)",
            assumptions=CORE_T7_ASSUMPTIONS,
            derivation=(
                boundary_step,
                ProofStep(
                    "Reach check",
                    f"No member of out({query.target}) = {list(target.outputs)} has an "
                    "observed descendant, so no observable can respond to the deletion.",
                ),
                ProofStep(
                    "Collapse",
                    f"P({','.join(query.outcomes)} | delete({query.target})) = "
                    f"P({','.join(query.outcomes)}). The estimand mentions neither the "
                    "mechanism nor its policy, so it cannot depend on either.",
                ),
            ),
        )

    return Unknown(
        reason=(
            f"The policy for {query.target!r} is a joint over {list(target.outputs)}, of "
            f"which {list(hidden_outputs)} are hidden dead ends. The answer needs that "
            "policy's marginal over the observed outputs, which the compiler cannot form "
            "without a domain for the hidden ones."
        ),
        next_algorithm="Marginalize the declared policy over its hidden outputs first.",
        suggestions=(
            f"Restate {query.target!r} with outputs "
            f"{sorted(set(target.outputs) - set(hidden_outputs))} and supply the "
            "already-marginalized policy.",
        ),
        missing_variables=hidden_outputs,
        assumptions=CORE_T7_ASSUMPTIONS,
        derivation=(
            boundary_step,
            ProofStep(
                "Reach check",
                f"Hidden output(s) {list(hidden_outputs)} reach nothing observed, but "
                f"out({query.target}) as a whole reaches {list(moved)}.",
            ),
        ),
    )


def identify_delete_via_t7(
    graph: MechanismGraph,
    query: DeleteMechanism,
    observed_variables: object | None = None,
) -> IdentificationResult:
    target = graph.get_mechanism(query.target)
    observed = (
        graph.observed_set
        if observed_variables is None
        else frozenset(_ordered(observed_variables))
    )
    missing_boundary = tuple(sorted(target.boundary - observed))
    if not missing_boundary:
        return Unknown(
            reason="T7 was requested, but the target boundary is already observed.",
            next_algorithm="Use T2/T4/T6 local factor replacement.",
        )
    if not query.outcomes:
        return Unknown(
            reason="T7 deletion queries require an explicit observed outcome set.",
            next_algorithm="Call DeleteMechanism(target, outcomes={...}).",
            missing_variables=missing_boundary,
        )
    overlap = tuple(sorted(set(query.outcomes) & set(target.outputs)))
    if overlap:
        return Unknown(
            reason=(
                f"Outcomes {list(overlap)} are outputs of the mechanism being deleted, so "
                "their post-intervention law is the policy supplied for it."
            ),
            next_algorithm="Ask about a downstream variable, or read the policy directly.",
            missing_variables=overlap,
        )

    hidden_outputs = tuple(sorted(set(target.outputs) - observed))
    if hidden_outputs:
        return _hidden_output_verdict(
            graph, query, hidden_outputs, missing_boundary
        )

    unknown_outcomes = set(query.outcomes) - observed
    if unknown_outcomes:
        return Unknown(
            reason="T7 outcomes must be observed variables.",
            suggestions=tuple(
                f"Observe outcome variable {variable!r}."
                for variable in sorted(unknown_outcomes)
            ),
            missing_variables=tuple(sorted(unknown_outcomes)),
        )
    missing_fallback = graph.missing_fallback_variables(query.target)
    if missing_fallback:
        return Unknown(
            reason="Mechanism deletion would orphan outputs without a declared fallback policy.",
            suggestions=(
                f"Declare the joint fallback policy P0_{query.target}"
                f"({','.join(missing_fallback)}) over the orphaned outputs.",
            ),
            missing_variables=missing_fallback,
        )

    reduction = reduce_mechanism_query_to_stochastic_intervention(graph, query)
    if reduction.variable_admg is None:
        return Unknown(
            reason="T7 reduction did not produce a Pearl ADMG.",
            next_algorithm="Build variable-level latent projection.",
            missing_variables=missing_boundary,
        )

    pearl_result = PearlIDBackend().identify(
        reduction.variable_admg,
        PearlIDQuery(reduction.pearl_outcomes, reduction.pearl_interventions),
    )
    if not isinstance(pearl_result, Identified):
        return Unknown(
            reason=(
                "The reduced Pearl query is not identifiable in the latent projection. "
                "That is reported as Unknown rather than Unidentified deliberately: a "
                "hedge refutes identifiability over ALL semi-Markovian models of the "
                "ADMG, and this projection's preimage is a strictly smaller class -- one "
                "noise per mechanism whose children are exactly out(m), single producers, "
                "declared output equalities. The mechanism query is also a MIXTURE against "
                "a policy the caller supplies, and each term failing does not make the "
                "mixture fail. Lifting the refutation across the projection is exactly "
                "conjecture H1+, which THEOREM_H1_PLUS.md marks open."
            ),
            next_algorithm="Prove H1+, or refute the mechanism query directly.",
            suggestions=(
                f"Pearl-level obstruction: {getattr(pearl_result, 'witness', None)}. "
                "That is evidence about the projected ADMG, not a proof about the "
                "mechanism query.",
            ),
            missing_variables=missing_boundary,
            derivation=(
                ProofStep(
                    "Boundary check",
                    f"Hidden boundary variables: {list(missing_boundary)}.",
                ),
                ProofStep("Pearl reduction", f"Backend result status: {pearl_result.status}."),
                ProofStep(
                    "Withhold the refutation",
                    "A hedge in the projection is not by itself a proof about the "
                    "mechanism query; see H1+.",
                ),
            ),
        )

    expression = SumOut(
        target.outputs,
        Product([Fallback(query.target, target.outputs), pearl_result.expression]),
    )
    return Identified(
        expression=expression,
        theorem="T7",
        # The Pearl sub-result may have introduced copied variables. Composing its
        # expression while dropping its aliases would hand the caller a formula naming
        # something nothing can resolve -- the same defect the aliases exist to fix.
        aliases=pearl_result.aliases,
        assumptions=T7_ASSUMPTIONS + pearl_result.assumptions,
        derivation=(
            ProofStep("Boundary check", f"Hidden boundary variables: {list(missing_boundary)}."),
            ProofStep(
                "Reduce to Pearl ID",
                f"delete({query.target}) installs an unconditional policy over "
                f"{list(target.outputs)}, so P(Y | delete) = sum_x P0(x) P(Y | do(x)) -- "
                "the policy does not read in(m), so it factors out of the truncated "
                "factorization and each term is a Pearl query on the latent projection.",
            ),
            ProofStep(
                "Pearl backend",
                f"Reduced effect identified by {pearl_result.theorem}.",
            ),
            ProofStep(
                "Compose deletion",
                f"Sum the Pearl term against P0_{query.target}"
                f"({','.join(target.outputs)}).",
            ),
        ),
    )


def identify_via_t7(
    graph: MechanismGraph | None = None,
    query: DeleteMechanism | ReplaceMechanism | None = None,
) -> Unknown:
    """Return an honest placeholder until ADMG latent projection and Pearl ID exist."""

    suggestions = ("Build bipartite ADMG, reduce to stochastic intervention, run Pearl ID.",)
    if graph is not None and query is not None:
        reduction = reduce_mechanism_query_to_stochastic_intervention(graph, query)
        suggestions = (
            f"Reduced {reduction.query_type}({reduction.target_mechanism}) to a "
            "stochastic intervention object.",
            *suggestions,
        )
    return Unknown(
        reason="T7 Pearl-ID reduction is not implemented.",
        next_algorithm="Build bipartite ADMG, reduce to stochastic intervention, run Pearl ID.",
        suggestions=suggestions,
    )
