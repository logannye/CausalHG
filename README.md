# CausalHG

A Python library for causal inference with mechanisms represented as **directed,
typed hyperedges**. A mechanism has several inputs and may produce several outputs
jointly. Name that mechanism, ask what happens when it is deleted or replaced, and
receive an identifying expression with its assumptions and derivation—or a
structured explanation of why the query cannot be answered.

The library is domain independent: mechanisms can represent software processing
steps, manufacturing operations, policy rules, or other joint causal processes.
An association hypergraph does not become a causal model merely by importing it;
the input/output roles and causal assumptions must be supplied.

**Status: research software, pre-1.0.** Read the [capability matrix](docs/capabilities.md)
before choosing an algorithm. The [roadmap](CAUSALHG_DEVELOPMENT_ROADMAP.md) describes
future work; it is not a list of currently available APIs.

## Install

From a checkout, with Python 3.11 or newer:

```bash
python -m pip install .
```

The shipped package currently uses only the Python standard library. For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

## Identify a mechanism intervention

Consider a processing step `m1` that consumes `A, B` and produces `C, D` together.
A second step consumes `C, E` and produces `F`. We want the distribution of `F`
after replacing `m1` with an input-independent joint policy over `C, D`.

```python
from causal_hypergraphs import DeleteMechanism, Identified, MechanismGraph, identify

graph = MechanismGraph(
    variables={"A", "B", "C", "D", "E", "F"},
    mechanisms={
        "m1": {"inputs": {"A", "B"}, "outputs": {"C", "D"}},
        "m2": {"inputs": {"C", "E"}, "outputs": {"F"}},
    },
    fallback_variables={"C", "D"},
)

result = identify(graph, DeleteMechanism("m1", outcomes={"F"}))
assert isinstance(result, Identified)
print(result.expression)
# sum_{C,D,E} P(E) * P(F | C,E) * P0_m1(C,D)

print(result.theorem)                 # T2
print([a.code for a in result.assumptions])
```

`fallback_variables` declares coverage for a symbolic policy; its numeric values
are supplied during evaluation. A deletion does not mean setting outputs to zero.
Its policy is **joint**, so it can preserve or change dependence between outputs.
`ReplaceMechanism` instead installs a kernel that can still read the mechanism's
inputs, and initially requires the same input/output boundary.

| Result | Meaning |
|---|---|
| `Identified` | An estimand is available under the attached assumptions. |
| `Unknown` | This compiler cannot justify an answer; inspect the reason and suggestions. |
| `Unidentified` | An applicable backend produced a nonidentification witness. |

The expression is an AST with scope, kernel introspection, marginalization, and
text/LaTeX rendering. The attached derivation is an inspectable record of the rules
used, not a proof checked by a formal theorem prover.

## Estimate from data

Continuing the example, construct synthetic frequency data with shared noise in
`m1`: `C` and `D` are equal in every observational record. The downstream process
has a separate error term. Repeated entries in the noise axes encode their weights.

```python
from itertools import product

from causal_hypergraphs import Dataset, estimate

records = []
for a, b, e, u, v in product(
    (0, 1), (0, 1), (0, 1), (0, 0, 0, 1), (0, 0, 0, 1)
):
    c = a ^ b ^ u
    records.append({"A": a, "B": b, "C": c, "D": c, "E": e, "F": (c & e) ^ v})

data = Dataset.from_records(records)
estimated = estimate(
    result,
    data,
    fallbacks={
        "m1": {(0, 0): 0.75, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.25}
    },
)

print(estimated.variables)            # ('F',)
print(dict(estimated.values))         # {(0,): 0.6875, (1,): 0.3125}
print(estimated.support.holds)        # True
```

Policy tuple axes follow the graph's sorted output names. Include zero-probability
entries explicitly; a missing policy entry is an error, not an implicit zero.

`estimate` evaluates the identified expression using empirical kernels and exact
finite-domain variable elimination. Numerical evaluation is exact for those fitted
kernels; that does not eliminate statistical error or validate causal assumptions.
Empty required strata are reported in `estimated.support.failures`, and affected
points are absent from `values` rather than represented as `nan`.

For sampled data, `Dataset.from_records(rows, unit="group_id")` declares the
independent sampling unit, and `estimate(..., bootstrap=200)` adds unit-bootstrap
intervals. Without `unit=`, each row is treated as independent. Support and
policy-weight diagnostics remain available without a bootstrap. Their empirical
checks do not prove population positivity; the policy diagnostic's configurable
count threshold is a heuristic.

## Supported scope

- Typed mechanism graphs with one producer per variable and independent mechanism
  noise as a declared assumption. Joint outputs may share noise.
- Mechanism deletion and boundary-preserving replacement; marginal queries;
  selected hidden-variable cases with explicit assumptions and refusals.
- A separate `ADMG` / `identify_effect` interface using the Shpitser–Pearl ID
  algorithm for ordinary variable-intervention queries.
- Finite-discrete evaluation, variable-elimination planning, empirical estimation,
  and continuous outcome expectations in supported factored queries.
- Graphical separation and covariate diagnostics. A diagnostic's “admissible”
  verdict is not a complete adjustment-set certificate.

Cyclic graphs can be represented, but identification requires that no kernel
needed by the query come from an observational cycle; successful local cases also
declare solvability assumptions. General cyclic inference is unsupported.
Continuous conditioning variables, general counterfactual identification, and a
public executable SCM runtime are not currently supported. See the
[capability matrix](docs/capabilities.md) for the exact boundaries.

The contribution is a mechanism-oriented interface and auditable computation.
The framework reduces to established structural causal models; it does not claim
greater expressive or identification power than Pearl models with suitable latents.

## Documentation

- [Current capabilities](docs/capabilities.md) and [migration notes](docs/migration.md).
- [Compiler specification](SPEC.md): objects, queries, dispatch, and refusals.
- [Foundations](FOUNDATIONS.md) and [whitepaper](whitepaper.md): definitions and research background.
- [Separation](THEOREM_T1.md), [factorization and interventions](THEOREM_T2_T3.md),
  [latent mechanisms](THEOREM_T4_T5.md), and [hidden-variable research](THEOREM_H1_PLUS.md).
- [Development roadmap](CAUSALHG_DEVELOPMENT_ROADMAP.md): proposed milestones and release gates.

Some research documents describe earlier reference implementations. The capability
matrix and public API describe the current package; unresolved theorem claims remain
research questions. `minimal_model/` is reference code, not the public runtime.

Licensed under [Apache 2.0](LICENSE). See [NOTICE](NOTICE).
