"""Build a concrete SCM compatible with an ADMG, and compute its laws exactly."""
from __future__ import annotations

import itertools
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from causal_hypergraphs.identification import ADMG

BINARY = (0, 1)
Point = tuple[int, ...]


@dataclass(frozen=True)
class ADMGModel:
    """A binary SCM realizing an ADMG: one latent per bidirected edge, one noise per node.

    Every node is a random function of its observed parents and of the latents on the
    bidirected edges it touches. That construction is exactly what a bidirected edge
    *means* -- an unobserved common cause of its two endpoints -- so the observed law this
    produces is a law the ADMG can generate, and any estimand claiming to identify an
    effect in that ADMG must reproduce the effect here.
    """

    admg: ADMG
    latents: tuple[tuple[str, str], ...]
    latent_priors: Mapping[tuple[str, str], float]
    kernels: Mapping[str, Mapping[Point, float]]
    """`P(node = 1 | its observed parents, then its latents)`, keyed in that order."""

    @property
    def nodes(self) -> tuple[str, ...]:
        return self.admg.nodes

    def _parents(self, node: str) -> tuple[str, ...]:
        return tuple(sorted(self.admg.parents(node)))

    def _touching(self, node: str) -> tuple[tuple[str, str], ...]:
        return tuple(edge for edge in self.latents if node in edge)

    def _probability(
        self, node: str, world: Mapping[str, int], latent: Mapping[tuple[str, str], int]
    ) -> float:
        key = tuple(world[p] for p in self._parents(node)) + tuple(
            latent[edge] for edge in self._touching(node)
        )
        return self.kernels[node][key]

    def _worlds(self) -> list[dict[tuple[str, str], int]]:
        return [
            dict(zip(self.latents, combination, strict=True))
            for combination in itertools.product(BINARY, repeat=len(self.latents))
        ]

    def _latent_weight(self, latent: Mapping[tuple[str, str], int]) -> float:
        weight = 1.0
        for edge, value in latent.items():
            prior = self.latent_priors[edge]
            weight *= prior if value == 1 else 1.0 - prior
        return weight

    def joint(self) -> dict[Point, float]:
        """`P(V)` over the observed nodes, exactly, by summing the latents away."""
        law: dict[Point, float] = {}
        for combination in itertools.product(BINARY, repeat=len(self.nodes)):
            world = dict(zip(self.nodes, combination, strict=True))
            total = 0.0
            for latent in self._worlds():
                weight = self._latent_weight(latent)
                for node in self.nodes:
                    one = self._probability(node, world, latent)
                    weight *= one if world[node] == 1 else 1.0 - one
                total += weight
            law[combination] = total
        return law

    def interventional(
        self, outcomes: Sequence[str], interventions: Mapping[str, int]
    ) -> dict[Point, float]:
        """`P(outcomes | do(interventions))` by truncated factorization, exactly."""
        free = [node for node in self.nodes if node not in interventions]
        fixed: dict[str, int] = dict(interventions)
        law: dict[Point, float] = {
            point: 0.0 for point in itertools.product(BINARY, repeat=len(outcomes))
        }
        for combination in itertools.product(BINARY, repeat=len(free)):
            world: dict[str, int] = dict(zip(free, combination, strict=True))
            world.update(fixed)
            total = 0.0
            for latent in self._worlds():
                weight = self._latent_weight(latent)
                for node in free:  # intervened nodes contribute no factor
                    one = self._probability(node, world, latent)
                    weight *= one if world[node] == 1 else 1.0 - one
                total += weight
            law[tuple(world[name] for name in outcomes)] += total
        return law


def random_scm(admg: ADMG, seed: int, *, floor: float = 0.12) -> ADMGModel:
    """A random model of `admg`, kept away from the boundary of the simplex.

    `floor` keeps every conditional probability inside `[floor, 1 - floor]`. Positivity is
    not a nicety here: an identifying formula is generally a ratio, and a model that puts
    a conditioning event at zero would make the estimand undefined rather than wrong,
    which tests nothing.
    """
    rng = random.Random(seed)
    latents = tuple(admg.bidirected_edges)
    kernels: dict[str, dict[Point, float]] = {}
    for node in admg.nodes:
        arity = len(admg.parents(node)) + sum(1 for edge in latents if node in edge)
        kernels[node] = {
            key: rng.uniform(floor, 1.0 - floor)
            for key in itertools.product(BINARY, repeat=arity)
        }
    return ADMGModel(
        admg=admg,
        latents=latents,
        latent_priors={edge: rng.uniform(0.3, 0.7) for edge in latents},
        kernels=kernels,
    )
