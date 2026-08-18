from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from causal_hypergraphs.graph import MechanismGraph


class DeterminationRule(Protocol):
    def closure_step(self, known: frozenset[str]) -> frozenset[str]:
        """Return variables newly determined by the known set."""
        ...


@dataclass(frozen=True)
class EqualityClosureRule:
    group: frozenset[str]

    def closure_step(self, known: frozenset[str]) -> frozenset[str]:
        if self.group & known:
            return self.group
        return frozenset()


def equality_rules_from_graph(graph: MechanismGraph) -> tuple[EqualityClosureRule, ...]:
    rules: list[EqualityClosureRule] = []
    for name in graph.mechanisms:
        for group in graph.get_mechanism(name).output_equalities:
            rules.append(EqualityClosureRule(frozenset(group)))
    return tuple(rules)


def deterministic_closure(
    graph: MechanismGraph,
    variables: object,
    rules: tuple[DeterminationRule, ...] | None = None,
) -> frozenset[str]:
    """Equality-based deterministic closure with an extension point for richer rules.

    This computes `Det_R` in the sense of THEOREM_T1.md 1.1 -- the closure under the
    *declared* rules -- and makes no attempt to compute `D_M`, the set of variables
    actually determined by the argument. The two are deliberately distinct:

    - `Det_R` subset of `D_M` (**validity**) is what T1's soundness proof needs, and it
      holds exactly when every declared `output_equalities` group really is a.s. equal.
      Validity cannot be checked here: this module sees typed incidence, never `F`.
      Declaring an equality the model does not satisfy yields unsound separations.
    - Equality of the two (**declaration completeness**) is what T1's completeness proof
      needs. It is an assumption, not a lemma. Determination arising from anything other
      than a declared equality group -- an injective `f_m`, structural zeros in a kernel
      -- is invisible here, and costs completeness only, never soundness.
    """

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
