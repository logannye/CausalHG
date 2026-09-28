"""Whole-set validation of the sufficient backdoor criterion for ordinary causal DAGs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from causal_hypergraphs.graph import MechanismGraph

from .pearl_id import ADMG
from .results import Assumption


@dataclass(frozen=True)
class AdjustmentResult:
    """A structural sufficient-criterion certificate, not an empirical guarantee.

    ``fail`` means this sufficient criterion fails; it does not prove that no valid
    adjustment formula exists, that a numerical estimate is biased in every model,
    or that the causal effect is unidentified. ``unsupported`` means the supplied
    model family is outside this validator's scope.
    """

    status: Literal["pass", "fail", "unsupported"]
    reason_code: str
    reason: str
    treatments: tuple[str, ...]
    outcomes: tuple[str, ...]
    covariates: tuple[str, ...]
    descendants_in_set: tuple[str, ...] = ()
    connected_pairs: tuple[tuple[str, str], ...] = ()
    criterion: str = "sufficient_dag_backdoor"
    assumptions: tuple[Assumption, ...] = ()


def _names(values: Iterable[str] | str, label: str) -> tuple[str, ...]:
    sequence = (values,) if isinstance(values, str) else tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in sequence):
        raise TypeError(f"{label} must contain nonempty string identifiers.")
    if len(sequence) != len(set(sequence)):
        raise ValueError(f"{label} must not contain duplicate identifiers.")
    return tuple(sorted(sequence))


def validate_adjustment_set(
    graph: ADMG | MechanismGraph,
    treatments: Iterable[str] | str,
    outcomes: Iterable[str] | str,
    covariates: Iterable[str] | str = (),
) -> AdjustmentResult:
    """Validate an entire proposed set using Pearl's sufficient backdoor criterion.

    Supported inputs are ADMG objects with no bidirected edges, or fully observed
    acyclic MechanismGraphs whose mechanisms each produce one variable. The set
    must contain no descendants of treatment in the original graph and must
    d-separate every treatment/outcome pair after outgoing treatment arrows are
    removed. The returned certificate does not check population positivity or
    whether the causal graph is correct, and is not a complete adjustment criterion.
    """
    if not isinstance(graph, ADMG | MechanismGraph):
        raise TypeError("Adjustment validation requires an ADMG or MechanismGraph.")
    x = _names(treatments, "Treatments")
    y = _names(outcomes, "Outcomes")
    z = _names(covariates, "Covariates")
    if not x or not y:
        raise ValueError("Treatment and outcome sets must both be nonempty.")
    if set(x) & set(y) or set(x) & set(z) or set(y) & set(z):
        raise ValueError("Treatments, outcomes, and covariates must be pairwise disjoint.")
    nodes = graph.node_set if isinstance(graph, ADMG) else graph.variable_set
    unknown = set(x + y + z) - nodes
    if unknown:
        raise ValueError(f"Unknown adjustment-query variables: {sorted(unknown)}.")

    def result(
        status: Literal["pass", "fail", "unsupported"],
        code: str,
        reason: str,
        *,
        descendants: tuple[str, ...] = (),
        connected: tuple[tuple[str, str], ...] = (),
    ) -> AdjustmentResult:
        assumptions = (
            Assumption("Causal DAG", "The declared graph correctly represents causal structure."),
            Assumption(
                "Population positivity",
                "Required treatment/covariate strata have positive population support.",
                state="unresolved",
            ),
        )
        if status == "pass":
            assumptions += (
                Assumption(
                    "Backdoor criterion",
                    "The complete proposed set passes the DAG criterion.",
                    state="structurally_checked",
                ),
            )
        return AdjustmentResult(
            status, code, reason, x, y, z, descendants, connected, assumptions=assumptions
        )

    if isinstance(graph, ADMG):
        if graph.bidirected_edges:
            return result(
                "unsupported",
                "bidirected_model",
                "This validator supports ordinary DAGs; bidirected-edge adjustment is unsupported.",
            )
        dag = graph
    else:
        if graph.hidden_variables:
            return result(
                "unsupported",
                "hidden_variables",
                "This validator requires every mechanism-graph variable to be observed.",
            )
        if any(len(mechanism.outputs) != 1 for mechanism in graph.mechanisms.values()):
            return result(
                "unsupported",
                "joint_output_mechanism",
                "This validator supports single-output mechanism DAGs only.",
            )
        if not graph.is_mechanism_acyclic():
            return result(
                "unsupported",
                "cyclic_model",
                "This validator requires an acyclic mechanism graph.",
            )
        dag = ADMG(
            graph.variables,
            (
                (parent, mechanism.outputs[0])
                for mechanism in graph.mechanisms.values()
                for parent in mechanism.inputs
            ),
        )

    descendants = set(x)
    pending = list(x)
    while pending:
        source = pending.pop()
        for child in dag.children(source):
            if child not in descendants:
                descendants.add(child)
                pending.append(child)
    forbidden = tuple(sorted(set(z) & descendants))
    if forbidden:
        return result(
            "fail",
            "treatment_descendant",
            "The proposed set includes a treatment descendant in the original graph.",
            descendants=forbidden,
        )

    backdoor = ADMG(dag.nodes, ((a, b) for a, b in dag.directed_edges if a not in x))
    connected = tuple((a, b) for a in x for b in y if not backdoor.m_separated(a, b, z))
    if connected:
        return result(
            "fail",
            "open_backdoor_path",
            "The complete proposed set leaves a treatment/outcome pair d-connected in the "
            "graph with outgoing treatment arrows removed.",
            connected=connected,
        )
    return result(
        "pass",
        "backdoor_criterion_satisfied",
        "The complete proposed set blocks all backdoor paths and contains no treatment "
        "descendants. Adjustment is justified under the declared causal model and positivity.",
    )


__all__ = ["AdjustmentResult", "validate_adjustment_set"]
