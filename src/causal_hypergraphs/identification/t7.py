from __future__ import annotations

from dataclasses import dataclass

from .results import Unknown


@dataclass(frozen=True)
class BipartiteADMG:
    """Interface placeholder for the latent-projected bipartite ADMG used by T7."""

    observed_nodes: tuple[str, ...] = ()
    hidden_nodes: tuple[str, ...] = ()
    directed_edges: tuple[tuple[str, str], ...] = ()
    bidirected_edges: tuple[tuple[str, str], ...] = ()


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
    replacement_kernel: str | None = None
    status: str = "not_implemented"


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


def identify_via_t7() -> Unknown:
    """Return an honest placeholder until ADMG latent projection and Pearl ID exist."""

    return Unknown(
        reason="T7 Pearl-ID reduction is not implemented.",
        next_algorithm="Build bipartite ADMG, reduce to stochastic intervention, run Pearl ID.",
    )
