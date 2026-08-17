# Causal Hypergraphs

Causal Hypergraphs is a Python research library for **mechanism-level causal
identification**.

It extends Pearl-style structural causal modeling with first-class mechanisms:
typed hyperedges with multiple inputs and jointly produced outputs. The goal is
not to exceed Pearl SCMs in expressive power. Pearl with enough latent variables
is universal. The goal is to make mechanism-level interventions explicit,
inspectable, and identifiable when the graph structure supports them.

The current `minimal_model/` package is retained as a reference model while the
new `src/causal_hypergraphs/` package grows into the main API.

## Core Idea

Many real interventions act on mechanisms, not individual variables:

- delete a reaction,
- replace an enzyme,
- inhibit a pathway,
- swap an ETL step,
- remove an evidence-generating process,
- revise a bundled policy mechanism.

In a causal hypergraph, a mechanism `m` has inputs and outputs:

```text
m: in(m) -> out(m)
```

Under the v1 assumptions, the observational distribution factorizes by
mechanism:

```text
P(V) = product P(exogenous variables) * product P(out(m) | in(m))
```

That makes mechanism deletion a local factor replacement. When every variable is
observed, each chain-rule factor is itself an observational quantity, so the
target factor is *omitted* rather than divided out:

```text
P(V | delete(m)) =
  product_{v exogenous} P(v)
  * product_{m' != m} P(out(m') | in(m'))
  * product_{v in out(m)} P0(v)
```

and mechanism replacement swaps the factor rather than deleting it:

```text
P(V | replace(m, m_prime)) =
  product_{v exogenous} P(v)
  * product_{m' != m} P(out(m') | in(m'))
  * P_m_prime(out(m) | in(m))
```

Writing the estimand this way matters. The equivalent quotient form
`P(V) / P(out(m) | in(m)) * ...` is `0/0` whenever the target mechanism is
deterministic with functionally coupled outputs — and under C2 that is the
generic case, since a mechanism whose noise carries fewer degrees of freedom
than it has outputs induces a singular factor. Deleting such a mechanism moves
probability mass onto configurations the observational law never visits, which
is exactly where the quotient is undefined. Omitting the factor is defined
everywhere the intervention puts mass.

When hidden variables are present the surviving factors are not individually
identified, so `T6` must go through the quotient and therefore carries an
explicit `Target positivity` certificate.

The library compiles those queries into proof-carrying estimands when the
current theory identifies them.

## What This Library Does

Given a typed mechanism graph and a query, Causal Hypergraphs returns:

- an identified estimand,
- the theorem used,
- the assumptions required,
- a derivation certificate,
- or an honest `Unknown` / `Unidentified` result.

Example:

```python
from causal_hypergraphs import (
    DeleteMechanism,
    MechanismGraph,
    identify,
)

graph = MechanismGraph(
    variables={"A", "B", "C", "D", "E", "F"},
    mechanisms={
        "m1": {"inputs": {"A", "B"}, "outputs": {"C", "D"}},
        "m2": {"inputs": {"C", "E"}, "outputs": {"F"}},
    },
    observed_variables={"A", "B", "C", "D", "E", "F"},
)

result = identify(graph, DeleteMechanism("m1"))

print(result.status)
print(result.expression)
print(result.theorem)
```

Expected result:

```text
identified
P(A) * P(B) * P(E) * P(F | C,E) * P0(C) * P0(D)
T2
```

## Public API

```python
from causal_hypergraphs import (
    DeleteMechanism,
    Identified,
    Mechanism,
    MechanismGraph,
    ReplaceMechanism,
    Unknown,
    identify,
)
```

Primary query flow:

```python
result = identify(
    graph=graph,
    query=DeleteMechanism("m1"),
    observed_variables={"A", "B", "C", "D", "E", "F"},
)
```

The compiler returns structured results, not booleans:

```python
if isinstance(result, Identified):
    print(result.expression)
    print(result.theorem)
    print(result.assumptions)
    print(result.derivation)
elif isinstance(result, Unknown):
    print(result.reason)
    print(result.next_algorithm)
    print(result.suggestions)
```

## V1 Assumptions

The first version intentionally supports a narrow, defensible formal core:

- **C1:** the mechanism dependency graph is acyclic.
- **C2:** mechanisms have independent exogenous noise.
- **C3:** each mechanism has input/output role typing.
- **C4:** each variable has at most one producing mechanism.

These assumptions make the mechanism-level chain rule valid. If they fail, the
compiler refuses rather than returning an unsound estimand.

## Current Capabilities

- Standard `pyproject.toml` package layout.
- Formal compiler semantics in `SPEC.md`.
- Typed mechanism graph validation.
- Mechanism deletion and replacement queries.
- T2/T3 identifiers under full observability.
- T4/T6 identifiers under observed-boundary hidden-variable settings.
- Expression AST with canonical products, scope introspection, primitive kernel
  extraction, marginalization, and plain-text/LaTeX rendering.
- Proof-carrying result objects with assumptions and derivation steps.
- Equality-aware d*-separation for the bipartite blowup.
- T7 preparation objects: typed bipartite DAG construction, latent projection to
  a bipartite ADMG, and stochastic-intervention reduction records.
- Isolated Pearl-ID backend for observational marginals, Markovian truncated
  factorization, and the canonical front-door pattern.
- Opt-in T7 front-door vertical slice via
  `identify(..., DeleteMechanism("m_x", outcomes={...}), allow_t7=True)`.
- Canonical reaction-network examples.
- Compatibility test coverage for the existing `minimal_model/` reference code.

## Not Yet Supported

- Complete T7 Pearl-ID reduction beyond the current front-door vertical slice.
- Hyper-hedge completeness.
- Complete Pearl-ID support beyond the currently implemented backend cases.
- General functional-determination closure beyond declared equality rules.
- Cyclic mechanism graphs.
- Markov-kernel mechanisms.
- Transition/Petri-net semantics with multiple producers.
- Production-grade estimators over empirical data.

Unsupported cases return explicit `Unknown` results with assumptions and
suggested next steps.

## Why Not Just Use Pearl SCMs?

Pearl SCMs can represent the same distributions with enough latent variables.
The difference is intervention vocabulary.

In Pearl form, deleting a multi-output mechanism becomes a multi-variable
stochastic intervention or an intervention on an unnamed latent. In a causal
hypergraph, the mechanism is named directly:

```text
delete(m)
replace(m, m_prime)
```

That better matches scientific and engineering practice.

## Installation

```bash
pip install -e ".[dev]"
```

## Development

```bash
python -m pytest -q
ruff check .
pyright
```

## Repository Structure

```text
src/causal_hypergraphs/
  graph/              typed incidence and validation
  expression/         probability expression algebra
  identification/     T2/T3/T4/T6 compilers and result objects
  separation/         d*-separation and deterministic closure
  semantics/          optional simulation/counterfactual helpers, planned

minimal_model/        compatibility reference implementation

examples/             planned notebook/script examples
tests/                compiler and API tests
```

## Examples

The new package includes importable example graph builders:

- `reaction_graph()`: `m1: A,B -> C,D`; `m2: C,E -> F`.
- `latent_mechanism_graph()`: adds `m_lat: empty -> B,E`.
- `hidden_variable_graph()`: adds hidden `W`; T6 accepts `m1` and returns
  `Unknown(next_algorithm="T7 Pearl-ID reduction")` for the boundary-violating
  `m_2` query.

## Roadmap

1. Stabilize packaging and preserve the existing 60-test reference suite.
2. Build expression-returning T2/T3/T4/T6 identifiers.
3. Add proof-carrying result objects and LaTeX/text rendering.
4. Implement bipartite ADMG and Pearl-ID reduction for T7.
5. Return hedge and hyper-hedge witnesses for non-identification.
6. Add estimator compilation and experiment-design suggestions.

## Status

This project is research software. It is intended for causal-inference
researchers and technical teams exploring mechanism-level causal queries. It
should not be used for scientific or clinical claims without independent
validation of the graph, assumptions, and estimands.
