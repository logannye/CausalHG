from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from causal_hypergraphs.graph import Mechanism

# The replacement name is rendered into the estimand and into its LaTeX, so it has to be a
# label rather than arbitrary text. Permissive enough for the "m'" convention.
_LABEL = re.compile(r"^[A-Za-z_][A-Za-z0-9_'-]*$")


def _ordered(values: Iterable[object] | str | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(sorted(str(v) for v in values))


@dataclass(frozen=True)
class DeleteMechanism:
    target: str
    outcomes: tuple[str, ...] = ()

    def __init__(self, target: str, outcomes: Iterable[object] | str | None = ()) -> None:
        object.__setattr__(self, "target", str(target))
        object.__setattr__(self, "outcomes", _ordered(outcomes))


@dataclass(frozen=True)
class ReplaceMechanism:
    """Replace mechanism `target` with a mechanism of the same typed incidence.

    `replacement` may be a bare name or a `Mechanism`. Given only a name, the compiler has
    nothing to check and records `rho(m') = rho(m)` as an assumption certificate. Given a
    `Mechanism`, it verifies the incidence and discharges that certificate into a proof
    step -- and rejects a mismatch, since `do(m -> m')` has no semantics when the
    replacement does not have the same input and output boundary.
    """

    target: str
    replacement: str = "m'"
    incidence: Mechanism | None = None

    def __init__(self, target: str, replacement: str | Mechanism = "m'") -> None:
        if isinstance(replacement, Mechanism):
            name, incidence = replacement.name, replacement
        else:
            name, incidence = str(replacement), None
        if not _LABEL.match(name):
            raise ValueError(
                f"Invalid replacement name {name!r}: the replacement name is rendered into "
                "the estimand, so it must match [A-Za-z_][A-Za-z0-9_'-]*."
            )
        object.__setattr__(self, "target", str(target))
        object.__setattr__(self, "replacement", name)
        object.__setattr__(self, "incidence", incidence)
