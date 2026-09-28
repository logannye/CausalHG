"""Reproducible width/cardinality benchmark with an independent analytic answer."""

from __future__ import annotations

import json
import platform
import time
import tracemalloc
from typing import Any

from causal_hypergraphs import CausalQuery, Composite, Identified, MechanismGraph, compile_query
from causal_hypergraphs.semantics import IntractableQuery, eliminate, evaluate, plan_elimination


class SymmetricChain:
    """Example third-party Model implementation: uniform stationary Markov chain."""

    def __init__(self, nodes: tuple[str, ...], cardinality: int):
        self.domains = dict.fromkeys(nodes, tuple(range(cardinality)))
        self.cardinality = cardinality

    def conditional(self, variables, given, assignment):
        assert len(variables) == 1
        if not given:
            return 1 / self.cardinality
        assert len(given) == 1
        return (
            0.8
            if assignment[variables[0]] == assignment[given[0]]
            else 0.2 / (self.cardinality - 1)
        )

    def conditional_expectation(self, target, given, assignment):
        return sum(
            value * self.conditional((target,), given, {**assignment, target: value})
            for value in self.domains[target]
        )

    def fallback(self, mechanism, variables, assignment, marginalized=()):
        raise AssertionError("No policy in this benchmark")

    def replacement(self, mechanism, variables, given, assignment):
        raise AssertionError("No replacement in this benchmark")


def run() -> dict[str, Any]:
    rows = []
    for size, cardinality in ((10, 2), (40, 2), (100, 2), (40, 4)):
        nodes = tuple(f"X{i:03}" for i in range(size))
        model = SymmetricChain(nodes, cardinality)
        tracemalloc.start()
        started = time.perf_counter()
        graph = MechanismGraph(
            nodes,
            {
                f"step{i}": {"inputs": (nodes[i - 1],), "outputs": (nodes[i],)}
                for i in range(1, size)
            },
        )
        compiled = compile_query(graph, CausalQuery((nodes[-1],), Composite(())))
        assert isinstance(compiled.result, Identified)
        expression = compiled.result.expression
        plan = plan_elimination(expression, model.domains)
        value = eliminate(expression, model, {nodes[-1]: 1}, max_entries=cardinality**2)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert abs(value - 1 / cardinality) < 1e-12
        assert plan.max_entries <= cardinality**2
        if size == 10:
            assert abs(evaluate(expression, model, {nodes[-1]: 1}) - value) < 1e-12
        rows.append(
            {
                "nodes": size,
                "hyperedges": size - 1,
                "max_arity": 2,
                "cardinality": cardinality,
                "induced_width": plan.induced_width,
                "max_entries": plan.max_entries,
                "enumeration_assignments": plan.naive_entries,
                "expression_characters": len(str(expression)),
                "seconds": elapsed,
                "peak_traced_bytes": peak,
                "answer": value,
            }
        )
    # The budget must refuse a wide factor before evaluating it.
    nodes = tuple(f"Y{i}" for i in range(20))
    graph = MechanismGraph(nodes, {"joint": {"inputs": (), "outputs": nodes}})
    compiled = compile_query(graph, CausalQuery(nodes[:1], Composite(())))
    assert isinstance(compiled.result, Identified)
    try:
        eliminate(
            compiled.result.expression, SymmetricChain(nodes, 2), {nodes[0]: 0}, max_entries=1024
        )
    except IntractableQuery:
        resource_guard = True
    else:
        raise AssertionError("Wide factor did not honor the resource limit")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "backend": "independent symmetric finite-chain Model",
        "cases": rows,
        "wide_factor_resource_guard": resource_guard,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
