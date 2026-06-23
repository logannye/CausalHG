from __future__ import annotations

from dataclasses import dataclass

from causal_hypergraphs.expression import Expression


@dataclass(frozen=True)
class Assumption:
    code: str
    description: str

    def __str__(self) -> str:
        return f"{self.code}: {self.description}"


@dataclass(frozen=True)
class ProofStep:
    label: str
    detail: str

    def __str__(self) -> str:
        return f"{self.label}: {self.detail}"


class IdentificationResult:
    status: str


@dataclass(frozen=True)
class Identified(IdentificationResult):
    expression: Expression
    theorem: str
    assumptions: tuple[Assumption, ...]
    derivation: tuple[ProofStep, ...]
    status: str = "identified"


@dataclass(frozen=True)
class Unknown(IdentificationResult):
    reason: str
    next_algorithm: str | None = None
    suggestions: tuple[str, ...] = ()
    missing_variables: tuple[str, ...] = ()
    assumptions: tuple[Assumption, ...] = ()
    derivation: tuple[ProofStep, ...] = ()
    status: str = "unknown"


@dataclass(frozen=True)
class Unidentified(IdentificationResult):
    reason: str
    witness: object | None = None
    assumptions: tuple[Assumption, ...] = ()
    derivation: tuple[ProofStep, ...] = ()
    status: str = "unidentified"
