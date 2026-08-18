"""Which covariates may be conditioned on around a mechanism-level query.

The compiler's own identifiers need no adjustment set -- the truncated factorization is a
complete answer, and none of the emitted estimands condition on a user-chosen covariate.
But nobody runs an estimator in isolation. Analysts stratify, filter to a subpopulation,
add a term to a regression; and *which variables may be conditioned on* is where analyses
break in practice. This module answers that question from the same graph, using the
`d_separated` oracle that until now was unreachable from the identification path.

The trap it exists for: conditioning on a marker that is downstream of the perturbation.
It looks like ordinary covariate control -- the marker is a real quantity that varies
across units -- and it removes part of the very effect being measured.

Two findings, with deliberately different evidential status:

**Post-treatment** is a structural fact. A covariate reachable from the intervened
mechanism is post-treatment in the graph, with no distributional assumption behind it.

**Path opening** is a warning. It is detected on the back-door graph -- the graph with the
target mechanism's outgoing edges severed, so that any remaining connection to the outcome
is non-causal -- by `d_separated` going from a separation verdict to a non-verdict when
the covariate is added to the conditioning set. `d_separated` is sound but complete only
under faithfulness, so a flagged covariate is one that *may* open a back-door path, not
one proven to. The report keeps the two apart rather than presenting a warning as a fact.
"""
from __future__ import annotations

from dataclasses import dataclass

from causal_hypergraphs.graph import Mechanism, MechanismGraph
from causal_hypergraphs.separation import d_separated

from .queries import DeleteMechanism, ReplaceMechanism


@dataclass(frozen=True)
class CovariateVerdict:
    """Whether one covariate may be conditioned on, and why not if not."""

    covariate: str
    admissible: bool
    post_treatment: bool
    opens_path: bool
    reason: str

    def __str__(self) -> str:
        mark = "ok " if self.admissible else "!! "
        return f"{mark}{self.covariate}: {self.reason}"


@dataclass(frozen=True)
class CovariateReport:
    """The verdicts for a set of candidate covariates, grouped by finding."""

    target: str
    outcome: str
    verdicts: tuple[CovariateVerdict, ...]

    @property
    def admissible(self) -> tuple[str, ...]:
        return tuple(v.covariate for v in self.verdicts if v.admissible)

    @property
    def post_treatment(self) -> tuple[str, ...]:
        return tuple(v.covariate for v in self.verdicts if v.post_treatment)

    @property
    def opens_path(self) -> tuple[str, ...]:
        return tuple(v.covariate for v in self.verdicts if v.opens_path)

    def summary(self) -> str:
        lines = [
            f"Conditioning around do({self.target}) with outcome {self.outcome!r}:",
            "",
            "  Structural -- post-treatment, no distributional assumption involved:",
        ]
        blocked = [v for v in self.verdicts if v.post_treatment or not v.admissible]
        structural = [v for v in blocked if v.post_treatment or not v.opens_path]
        lines.extend(f"    {v}" for v in structural) if structural else lines.append(
            "    (none)"
        )
        lines.append("")
        lines.append(
            "  Warning -- may open a back-door path; rests on faithfulness, so this is "
            "not a proof of harm:"
        )
        warnings = [v for v in self.verdicts if v.opens_path and not v.post_treatment]
        lines.extend(f"    {v}" for v in warnings) if warnings else lines.append("    (none)")
        lines.append("")
        lines.append(f"  Admissible: {list(self.admissible) or '(none)'}")
        return "\n".join(lines)


def _descendants(graph: MechanismGraph, start: str) -> frozenset[str]:
    """Everything reachable from `start` along directed edges of the bipartite blowup."""
    children: dict[str, set[str]] = {}
    for parent, child in graph.bipartite_edges():
        children.setdefault(parent, set()).add(child)
    seen: set[str] = set()
    frontier = [start]
    while frontier:
        node = frontier.pop()
        for child in children.get(node, ()):
            if child not in seen:
                seen.add(child)
                frontier.append(child)
    return frozenset(seen)


def _severed(graph: MechanismGraph, target: str) -> MechanismGraph:
    """`graph` with the target mechanism's outgoing edges removed.

    This is the back-door graph: with the intervention's own influence cut, any surviving
    connection between the mechanism and the outcome is non-causal. Testing there is what
    separates "conditioning opened a spurious path" from "there is a causal path", which a
    test on the full graph cannot do -- the causal path keeps them d-connected no matter
    what the covariate does.
    """
    mechanisms = dict(graph.mechanisms)
    original = graph.get_mechanism(target)
    mechanisms[target] = Mechanism(
        target, inputs=original.inputs, outputs=(), latent=original.latent
    )
    return MechanismGraph(
        variables=graph.variables,
        mechanisms=mechanisms,
        observed_variables=graph.observed_variables,
        fallback_variables=graph.fallback_variables,
    )


def check_covariates(
    graph: MechanismGraph,
    query: DeleteMechanism | ReplaceMechanism,
    outcome: str,
    covariates: object,
) -> CovariateReport:
    """Classify each covariate as admissible, post-treatment, or possibly path-opening.

    `admissible` means only that neither failure mode was detected: the covariate is not
    downstream of the intervention, and conditioning on it does not open a back-door path
    that was provably blocked. It is not a certificate that adjusting for it yields an
    unbiased estimate -- that depends on what else is being conditioned on and on
    assumptions no graph can supply.
    """
    names = (covariates,) if isinstance(covariates, str) else tuple(covariates)  # type: ignore[arg-type]
    unknown = sorted({outcome, *names} - graph.variable_set)
    if unknown:
        raise ValueError(
            f"Not variables of the graph: {unknown}. A misspelled covariate would "
            "otherwise be reported as admissible."
        )

    target = query.target
    graph.get_mechanism(target)
    if target in graph.variable_set:
        # The mechanism and a variable would be one node in the bipartite blowup, which
        # makes the separation queries below ill-posed. Raising beats degrading quietly:
        # skipping the path-opening test would leave a check that runs, passes, and cannot
        # report -- every covariate would come back admissible for the wrong reason.
        raise ValueError(
            f"{target!r} names both a mechanism and a variable. They are one node in the "
            "bipartite blowup, so separation queries about the mechanism are ambiguous. "
            "Rename one of them."
        )
    downstream = _descendants(graph, target)
    back_door = _severed(graph, target)
    separated_alone = d_separated(back_door, target, outcome)

    verdicts: list[CovariateVerdict] = []
    for name in names:
        if name == outcome:
            verdicts.append(
                CovariateVerdict(
                    covariate=name,
                    admissible=False,
                    post_treatment=name in downstream,
                    opens_path=False,
                    reason="this is the outcome itself; conditioning on it leaves nothing "
                    "to estimate.",
                )
            )
            continue
        if name in downstream:
            verdicts.append(
                CovariateVerdict(
                    covariate=name,
                    admissible=False,
                    post_treatment=True,
                    opens_path=False,
                    reason=(
                        f"post-treatment: reachable from {target!r} in the graph, so "
                        "conditioning on it removes part of the effect being measured. "
                        "Structural, not an assumption."
                    ),
                )
            )
            continue

        opens = separated_alone and not d_separated(back_door, target, outcome, (name,))
        verdicts.append(
            CovariateVerdict(
                covariate=name,
                admissible=not opens,
                post_treatment=False,
                opens_path=opens,
                reason=(
                    "may open a back-door path: conditioning on it d-connects "
                    f"{target!r} to {outcome!r} in the graph with {target!r}'s outgoing "
                    "edges severed, which is the signature of a collider. Rests on "
                    "faithfulness, so this is a warning rather than a proof."
                )
                if opens
                else (
                    "not downstream of the intervention, and conditioning on it opens no "
                    "back-door path that was blocked."
                ),
            )
        )

    return CovariateReport(target=target, outcome=outcome, verdicts=tuple(verdicts))
