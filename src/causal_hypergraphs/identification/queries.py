from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeleteMechanism:
    target: str


@dataclass(frozen=True)
class ReplaceMechanism:
    target: str
    replacement: str = "m'"
