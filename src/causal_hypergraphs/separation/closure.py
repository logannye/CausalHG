from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from causal_hypergraphs.graph import MechanismGraph


class DeterminationRule(Protocol):
    def closure_step(self, known: frozenset[str]) -> frozenset[str]:
        """Return variables newly determined by the known set."""


@dataclass(frozen=True)
class EqualityClosureRule:
    group: frozenset[str]

    def closure_step(self, known: frozenset[str]) -> frozenset[str]:
        if self.group & known:
            return self.group
        return frozenset()


def equality_rules_from_graph(graph: MechanismGraph) -> tuple[EqualityClosureRule, ...]:
    rules: list[EqualityClosureRule] = []
    for mechanism in graph.mechanisms.values():
        for group in mechanism.output_equalities:
            rules.append(EqualityClosureRule(frozenset(group)))
    return tuple(rules)


def deterministic_closure(
    graph: MechanismGraph,
    variables: object,
    rules: tuple[DeterminationRule, ...] | None = None,
) -> frozenset[str]:
    """Equality-based deterministic closure with an extension point for richer rules."""

    if isinstance(variables, str):
        closure = frozenset({variables})
    else:
        closure = frozenset(str(v) for v in variables)  # type: ignore[union-attr]
    active_rules = equality_rules_from_graph(graph) if rules is None else rules
    changed = True
    while changed:
        changed = False
        expanded = set(closure)
        for rule in active_rules:
            expanded.update(rule.closure_step(closure))
        new_closure = frozenset(expanded)
        if new_closure != closure:
            closure = new_closure
            changed = True
    return closure
