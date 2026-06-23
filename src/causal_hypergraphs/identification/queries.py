from __future__ import annotations

from dataclasses import dataclass


def _ordered(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(sorted(str(v) for v in values))


@dataclass(frozen=True)
class DeleteMechanism:
    target: str
    outcomes: tuple[str, ...] = ()

    def __init__(self, target: str, outcomes: object = ()) -> None:
        object.__setattr__(self, "target", str(target))
        object.__setattr__(self, "outcomes", _ordered(outcomes))


@dataclass(frozen=True)
class ReplaceMechanism:
    target: str
    replacement: str = "m'"

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", str(self.target))
        object.__setattr__(self, "replacement", str(self.replacement))
