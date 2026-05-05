"""Reference implementation of a Hypergraph SCM (v1 conventions).

See FOUNDATIONS.md for the formal definitions. This module is intended as a
faithful, minimal reflection of the formalism; readability is prioritized over
performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


NoiseRecord = dict  # {"mechanism": {name: noise}, "exogenous": {var: value}}


@dataclass(frozen=True)
class Factor:
    """A single factor in a chain-rule factorization of P(V).

    For an exogenous variable v: variables=(v,), conditional_on=(), source="exogenous".
    For a mechanism m: variables=tuple(out(m)), conditional_on=tuple(in(m)), source=m.name.
    For a fallback factor (post-deletion): variables=(v,), conditional_on=(), source="P_0".
    """

    variables: tuple[str, ...]
    conditional_on: tuple[str, ...]
    source: str

    def __repr__(self) -> str:
        var_str = ",".join(self.variables)
        cond = "" if not self.conditional_on else " | " + ",".join(self.conditional_on)
        return f"P({var_str}{cond}) [{self.source}]"


@dataclass(frozen=True)
class Mechanism:
    """A single hyperedge with a typed (in, out) incidence and a joint structural function.

    f: (input_values: dict[str, float], noise) -> output_values: dict[str, float]
    sample_noise: (rng) -> noise
    abduct: (input_values, output_values) -> noise   [optional, required for counterfactuals]
    output_equalities: tuples of outputs that are deterministically equal under f.
        Used by d*-separation's deterministic-relations augmentation (see THEOREM_T1.md §5.1).
        Example: (("C", "D"),) means C and D are always equal — knowing one determines the other.
    """

    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    f: Callable[[dict, object], dict]
    sample_noise: Callable[[np.random.Generator], object]
    abduct: Optional[Callable[[dict, dict], object]] = None
    output_equalities: tuple[tuple[str, ...], ...] = ()
    latent: bool = False  # If True, f is treated as unknown for identifiability purposes
    # but still used for ground-truth simulation (HADMG semantics, see THEOREM_T4_T5.md).


@dataclass
class HypergraphSCM:
    variables: tuple[str, ...]
    mechanisms: tuple[Mechanism, ...]
    exogenous: dict[str, Callable[[np.random.Generator], float]] = field(default_factory=dict)
    # Variables whose value is fixed by an intervention (do_node). Their cf-time value
    # comes from `exogenous` (the intervention's point mass), NOT from abducted noise.
    forced: frozenset[str] = field(default_factory=frozenset)

    def validate(self) -> None:
        """Check v1 conventions C1, C4 and incidence well-formedness.

        Raises ValueError on the first violation. C2 (deterministic functions with
        independent per-mechanism noise) is structural to the Mechanism dataclass.
        C3 (bipartite role typing — only an in/out partition) is structural to the
        dataclass's `inputs`/`outputs` fields. The in/out *disjointness* check below
        is a well-formedness requirement of the typed incidence (FOUNDATIONS.md §1),
        not C3 itself.
        """
        var_set = set(self.variables)
        # Incidence well-formedness: in(m) and out(m) must be subsets of V, disjoint.
        for m in self.mechanisms:
            for v in m.inputs:
                if v not in var_set:
                    raise ValueError(f"Mechanism {m.name!r} input {v!r} not in V.")
            for v in m.outputs:
                if v not in var_set:
                    raise ValueError(f"Mechanism {m.name!r} output {v!r} not in V.")
            if set(m.inputs) & set(m.outputs):
                raise ValueError(
                    f"Mechanism {m.name!r} has overlapping inputs and outputs "
                    "(typed-incidence well-formedness, FOUNDATIONS.md §1)."
                )
        # C4: single producer.
        producer_count: dict[str, list[str]] = {}
        for m in self.mechanisms:
            for v in m.outputs:
                producer_count.setdefault(v, []).append(m.name)
        for v, names in producer_count.items():
            if len(names) > 1:
                raise ValueError(
                    f"Variable {v!r} produced by multiple mechanisms {names} (C4 violation)."
                )
        # C1: mechanism-graph acyclicity.
        if not self.is_mechanism_acyclic():
            raise ValueError("Mechanism dependency graph is cyclic (C1 violation).")

    # -------- HADMG accessors --------

    def latent_mechanisms(self) -> tuple[Mechanism, ...]:
        return tuple(m for m in self.mechanisms if m.latent)

    def observed_mechanisms(self) -> tuple[Mechanism, ...]:
        return tuple(m for m in self.mechanisms if not m.latent)

    def is_hadmg(self) -> bool:
        """True iff at least one mechanism is marked latent."""
        return any(m.latent for m in self.mechanisms)

    def is_mechanism_deletion_identifiable(
        self, mechanism_name: str, observed_variables: Optional[set[str]] = None
    ) -> tuple[bool, str]:
        """T4: do(¬m*) is identifiable from P(V_obs) iff in(m*) ∪ out(m*) ⊆ V_obs.

        Returns (verdict, justification). Under v1's all-variables-observed convention
        (observed_variables=None means all of self.variables), T4 always returns True
        for any existing mechanism — the hypergraph framework's surprising positive
        result (THEOREM_T4_T5.md §1).
        """
        target = next((m for m in self.mechanisms if m.name == mechanism_name), None)
        if target is None:
            return False, f"No mechanism named {mechanism_name!r}."
        obs = set(self.variables) if observed_variables is None else set(observed_variables)
        boundary = set(target.inputs) | set(target.outputs)
        missing = boundary - obs
        if missing:
            return (
                False,
                f"T4 boundary condition fails: {sorted(missing)} ⊄ V_obs. "
                "Mechanism factor P(out(m) | in(m)) not directly observable as a conditional.",
            )
        return (
            True,
            f"T4 boundary condition holds: in(m) ∪ out(m) = {sorted(boundary)} ⊆ V_obs. "
            f"Identifying expression: P(V) / P(out(m) | in(m)) · ∏ P_0(v).",
        )

    def producers(self, var: str) -> list[Mechanism]:
        return [m for m in self.mechanisms if var in m.outputs]

    def is_exogenous(self, var: str) -> bool:
        return len(self.producers(var)) == 0

    def mechanism_dag(self) -> dict[str, set[str]]:
        edges: dict[str, set[str]] = {m.name: set() for m in self.mechanisms}
        for m in self.mechanisms:
            for out in m.outputs:
                for m2 in self.mechanisms:
                    if out in m2.inputs:
                        edges[m.name].add(m2.name)
        return edges

    def is_mechanism_acyclic(self) -> bool:
        edges = self.mechanism_dag()
        in_degree = {n: 0 for n in edges}
        for succs in edges.values():
            for s in succs:
                in_degree[s] += 1
        queue = [n for n, d in in_degree.items() if d == 0]
        visited = 0
        while queue:
            n = queue.pop()
            visited += 1
            for s in edges[n]:
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    queue.append(s)
        return visited == len(edges)

    def topological_order(self) -> list[Mechanism]:
        edges = self.mechanism_dag()
        in_degree = {m.name: 0 for m in self.mechanisms}
        for succs in edges.values():
            for s in succs:
                in_degree[s] += 1
        by_name = {m.name: m for m in self.mechanisms}
        queue = [by_name[n] for n, d in in_degree.items() if d == 0]
        order: list[Mechanism] = []
        while queue:
            m = queue.pop(0)
            order.append(m)
            for s in edges[m.name]:
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    queue.append(by_name[s])
        if len(order) != len(self.mechanisms):
            raise ValueError("Mechanism dependency graph is cyclic; v1 requires acyclicity (C1).")
        return order

    def bipartite_blowup(self) -> tuple[set[str], set[tuple[str, str]]]:
        nodes: set[str] = set(self.variables) | {m.name for m in self.mechanisms}
        edges: set[tuple[str, str]] = set()
        for m in self.mechanisms:
            for v in m.inputs:
                edges.add((v, m.name))
            for v in m.outputs:
                edges.add((m.name, v))
        return nodes, edges

    def sample(self, rng: Optional[np.random.Generator] = None) -> dict[str, float]:
        values, _ = self.sample_with_noise(rng)
        return values

    def sample_with_noise(
        self, rng: Optional[np.random.Generator] = None
    ) -> tuple[dict[str, float], NoiseRecord]:
        rng = rng if rng is not None else np.random.default_rng()
        values: dict[str, float] = {}
        exo_record: dict[str, float] = {}
        for v in self.variables:
            if self.is_exogenous(v):
                if v not in self.exogenous:
                    raise ValueError(
                        f"Variable {v!r} has no producing mechanism and no exogenous sampler."
                    )
                values[v] = self.exogenous[v](rng)
                exo_record[v] = values[v]
        mech_record: dict[str, object] = {}
        for m in self.topological_order():
            input_vals = {k: values[k] for k in m.inputs}
            noise = m.sample_noise(rng)
            mech_record[m.name] = noise
            output_vals = m.f(input_vals, noise)
            for out_name in m.outputs:
                values[out_name] = output_vals[out_name]
        return values, {"mechanism": mech_record, "exogenous": exo_record}

    def evaluate_with_noise(
        self, noise_record: NoiseRecord, rng: Optional[np.random.Generator] = None
    ) -> dict[str, float]:
        rng = rng if rng is not None else np.random.default_rng()
        values: dict[str, float] = {}
        exo = noise_record.get("exogenous", {})
        mech = noise_record.get("mechanism", {})
        for v in self.variables:
            if self.is_exogenous(v):
                if v in self.forced:
                    # Intervention's point mass overrides any abducted value.
                    values[v] = self.exogenous[v](rng)
                elif v in exo:
                    values[v] = exo[v]
                elif v in self.exogenous:
                    # Newly exogenous post-intervention via mechanism deletion: sample fresh from P_0.
                    values[v] = self.exogenous[v](rng)
                else:
                    raise ValueError(
                        f"No exogenous value or sampler available for {v!r}."
                    )
        for m in self.topological_order():
            input_vals = {k: values[k] for k in m.inputs}
            noise = mech.get(m.name, m.sample_noise(rng))
            output_vals = m.f(input_vals, noise)
            for out_name in m.outputs:
                values[out_name] = output_vals[out_name]
        return values

    # -------- Three do-operators (return new SCMs; original unchanged) --------

    def do_node(self, var: str, value: float) -> "HypergraphSCM":
        new_mechs = tuple(m for m in self.mechanisms if var not in m.outputs)
        new_exo = dict(self.exogenous)
        new_exo[var] = (lambda rng, v=value: v)
        return HypergraphSCM(
            variables=self.variables,
            mechanisms=new_mechs,
            exogenous=new_exo,
            forced=self.forced | {var},
        )

    def do_delete_mechanism(self, mechanism_name: str) -> "HypergraphSCM":
        new_mechs = tuple(m for m in self.mechanisms if m.name != mechanism_name)
        new_scm = HypergraphSCM(
            variables=self.variables,
            mechanisms=new_mechs,
            exogenous=dict(self.exogenous),
            forced=self.forced,
        )
        # Fail loudly if any orphaned variable lacks a fallback exogenous sampler.
        for v in self.variables:
            if new_scm.is_exogenous(v) and v not in new_scm.exogenous:
                raise ValueError(
                    f"Deleting {mechanism_name!r} orphans variable {v!r}; "
                    "provide P_0 (a fallback sampler in `exogenous`) for safe mechanism deletion."
                )
        return new_scm

    def do_replace_mechanism(self, name: str, new_mechanism: Mechanism) -> "HypergraphSCM":
        # Enforce same incidence (rho(m') = rho(m)).
        original = next((m for m in self.mechanisms if m.name == name), None)
        if original is None:
            raise ValueError(f"No mechanism named {name!r}.")
        if set(original.inputs) != set(new_mechanism.inputs) or set(original.outputs) != set(
            new_mechanism.outputs
        ):
            raise ValueError(
                f"Replacement mechanism must have the same typed incidence as {name!r}."
            )
        new_mechs = tuple(new_mechanism if m.name == name else m for m in self.mechanisms)
        return HypergraphSCM(
            variables=self.variables,
            mechanisms=new_mechs,
            exogenous=dict(self.exogenous),
            forced=self.forced,
        )

    # -------- Chain-rule factorization (Lemma 1.1, T2, T3) --------

    def factorize(self) -> list[Factor]:
        """Mechanism-level chain-rule factorization of P^M(V).

        Returns the factor list of Lemma 1.1 in THEOREM_T2_T3.md:
        one factor per exogenous variable, one factor per mechanism (joint conditional
        of out(m) given in(m)). Order respects the topological order of G_E for
        readability, but factor identity is order-independent.
        """
        factors: list[Factor] = []
        for v in self.variables:
            if self.is_exogenous(v) and v not in self.forced:
                factors.append(Factor(variables=(v,), conditional_on=(), source="exogenous"))
        for m in self.topological_order():
            factors.append(
                Factor(variables=tuple(m.outputs), conditional_on=tuple(m.inputs), source=m.name)
            )
        return factors

    def truncated_factorization(self, deleted_mechanism: str) -> list[Factor]:
        """Factor list of P^{M^{¬m}}(V) under T2.

        Surviving exogenous and non-deleted mechanism factors, plus one P_0 factor
        per orphaned output. Under C4, every output of the deleted mechanism is orphaned.
        """
        target = next((m for m in self.mechanisms if m.name == deleted_mechanism), None)
        if target is None:
            raise ValueError(f"No mechanism named {deleted_mechanism!r}.")
        # Confirm C4: deleted mechanism's outputs have no other producer.
        for v in target.outputs:
            other_producers = [m.name for m in self.mechanisms if m.name != deleted_mechanism and v in m.outputs]
            if other_producers:
                raise ValueError(
                    f"Variable {v!r} also produced by {other_producers}; "
                    "T2 truncated factorization assumes C4 (single producer)."
                )

        factors: list[Factor] = []
        for v in self.variables:
            if self.is_exogenous(v) and v not in self.forced:
                factors.append(Factor(variables=(v,), conditional_on=(), source="exogenous"))
        for v in target.outputs:
            factors.append(Factor(variables=(v,), conditional_on=(), source="P_0"))
        for m in self.mechanisms:
            if m.name == deleted_mechanism:
                continue
            factors.append(
                Factor(variables=tuple(m.outputs), conditional_on=tuple(m.inputs), source=m.name)
            )
        return factors

    # -------- Counterfactual (Pearl's three steps, generalized) --------

    def counterfactual(
        self,
        observed: dict[str, float],
        intervention: Callable[["HypergraphSCM"], "HypergraphSCM"],
        query: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict[str, float]:
        """Abduct, intervene, predict.

        `observed` must include all exogenous variables and all variables incident to
        any mechanism whose noise we need to abduct.
        """
        # Step 1: abduction.
        exo: dict[str, float] = {}
        for v in self.variables:
            if self.is_exogenous(v):
                if v not in observed:
                    raise ValueError(f"Exogenous variable {v!r} required in observation.")
                exo[v] = observed[v]
        mech: dict[str, object] = {}
        for m in self.mechanisms:
            if m.abduct is None:
                raise ValueError(
                    f"Mechanism {m.name!r} has no `abduct`; counterfactuals not supported."
                )
            try:
                input_vals = {k: observed[k] for k in m.inputs}
                output_vals = {k: observed[k] for k in m.outputs}
            except KeyError as e:
                raise ValueError(
                    f"Observation missing variable required for mechanism {m.name!r}: {e}"
                ) from e
            mech[m.name] = m.abduct(input_vals, output_vals)
        noise_record: NoiseRecord = {"mechanism": mech, "exogenous": exo}

        # Step 2: act.
        cf_scm = intervention(self)

        # Step 3: predict.
        cf_values = cf_scm.evaluate_with_noise(noise_record, rng=rng)
        return {q: cf_values[q] for q in query if q in cf_values}
