from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

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
    aliases: Mapping[str, str] = field(default_factory=dict)
    """Copied variables the expression introduced, mapped to what they are copies of.

    An identifying formula sometimes needs a second, independent name for a variable --
    the `x'` in the front-door estimand, which must not be captured by the `x` held at the
    do-value outside it. A fresh name has no distribution of its own, so the result carries
    what each copy stands for and `semantics.with_aliases` makes a model answer for it.
    Empty for every estimand that needs no copy, which is most of them.
    """


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
