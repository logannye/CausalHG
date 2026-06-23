# Mechanism Intervention Compiler Specification

This document specifies the public semantics of the v1 mechanism intervention
compiler. It is intentionally narrower than the full research program: when the
current compiler cannot justify an estimand from the stated assumptions, it must
return a structured refusal instead of a plausible-looking formula.

## Objects

### Mechanism Graph

A `MechanismGraph` is a typed incidence structure:

```text
G = (V, M, in, out, V_obs, V_hidden, F0)
```

- `V` is the finite set of variable names.
- `M` is the finite set of mechanism names.
- `in(m) subset V` is the ordered input boundary of mechanism `m`.
- `out(m) subset V` is the ordered output boundary of mechanism `m`.
- `boundary(m) = in(m) union out(m)`.
- `V_obs` is the set of observed variables.
- `V_hidden = V \ V_obs`.
- `F0` is the set of variables with declared fallback distributions `P0(v)`.

The compiler treats mechanisms as named causal objects. Structural functions and
noise samplers may be attached by examples, but identification must depend only
on typed incidence, observability, and declared assumptions.

### V1 Assumptions

The v1 compiler relies on four structural assumptions:

- **C1:** the mechanism dependency graph is acyclic.
- **C2:** mechanisms have independent exogenous noise.
- **C3:** mechanisms use input/output role typing.
- **C4:** each variable has at most one producing mechanism.

Graph construction validates C1, C3, and C4 directly. C2 is recorded as an
assumption certificate because it is semantic rather than checkable from
incidence alone.

## Queries

### Mechanism Deletion

`DeleteMechanism(m)` denotes replacing mechanism `m` with fallback distributions
for its outputs:

```text
delete(m): remove P(out(m) | in(m)), insert product_{v in out(m)} P0(v)
```

Deletion is not defined unless every output in `out(m)` has a declared fallback
policy. If any output lacks `P0(v)`, the compiler returns `Unknown` with the
missing output variables.

### Mechanism Replacement

`ReplaceMechanism(m, m_prime)` denotes replacing the local factor for `m` with a
new mechanism factor with the same incidence:

```text
replace(m, m_prime): remove P(out(m) | in(m)), insert P_m_prime(out(m) | in(m))
```

The v1 API records the replacement mechanism by name and assumes incidence
compatibility as an explicit certificate. Later versions should validate
replacement incidence when the replacement mechanism object is available.

## Expression Language

Compiler outputs are expression ASTs, not strings. The v1 expression language
contains:

- `Probability(variables, given=...)` for observational kernels.
- `MechanismFactor(name, variables, given=...)` for named mechanism kernels.
- `ReplacementFactor(name, variables, given=...)` for replacement kernels.
- `Fallback(variable)` for fallback output factors.
- `Product([...])` for commutative products.
- `Quotient(numerator, denominator)` for local factor removal.
- `SumOut(variables, expression)` for marginalization.

Expressions must expose:

- a canonical key for stable equality and hashing,
- the variable scope required to evaluate the expression,
- the conditioning variables,
- the primitive kernels referenced by the expression,
- plain-text and LaTeX renderings.

String rendering is for human inspection only. Tests that assert compiler
semantics should prefer AST properties when possible.

## Theorem Dispatch

Let `O` be the observed-variable set used for identification. If the caller does
not override `observed_variables`, the graph's `observed_variables` field is
used.

### T2: Full-Observation Deletion

If all variables are observed and no mechanism is marked latent:

```text
P(O | delete(m)) =
  P(O) / P(out(m) | in(m)) * product_{v in out(m)} P0(v)
```

The result theorem is `T2`.

### T3: Full-Observation Replacement

If all variables are observed and no mechanism is marked latent:

```text
P(O | replace(m, m_prime)) =
  P(O) / P(out(m) | in(m)) * P_m_prime(out(m) | in(m))
```

The result theorem is `T3`.

### T4: All Variables Observed With Latent Mechanisms

If all variables are observed and at least one mechanism is marked latent, the
same local replacement formula is valid for deletion. The result theorem is
`T4`.

Replacement in this setting currently returns theorem label `T4.1`.

### T6: Hidden Variables With Observed Target Boundary

If hidden variables exist but the target boundary is fully observed:

```text
boundary(m) subset O
```

the compiler uses the same local factor replacement formula over `P(O)`:

```text
P(O | delete(m)) =
  P(O) / P(out(m) | in(m)) * product_{v in out(m)} P0(v)
```

and analogously for replacement. The result theorem is `T6`.

## Refusal Semantics

The compiler returns structured result objects:

- `Identified`: the current implementation has an estimand plus theorem,
  assumptions, and derivation steps.
- `Unknown`: the current implementation does not identify the query, but no
  non-identification theorem has been established.
- `Unidentified`: a backend has produced a non-identification witness such as a
  Pearl hedge.

Boundary-violating hidden-variable cases return this by default:

```text
Unknown(
  reason="Target mechanism boundary contains hidden variables.",
  next_algorithm="T7 Pearl-ID reduction",
  missing_variables=...
)
```

This is an honest implementation boundary, not a claim of impossibility.

Callers may opt into the experimental T7 vertical slice with
`identify(..., allow_t7=True)`. The current opt-in path supports single-output
mechanism deletion with explicit observed outcomes:

```python
identify(
    graph,
    DeleteMechanism("m_x", outcomes={"Y"}),
    allow_t7=True,
)
```

For this supported case, deletion of the target mechanism is compiled as a
stochastic intervention on its output:

```text
P(Y | delete(m_x)) = sum_x P0(x) * P(Y | do(x))
```

The Pearl backend must identify `P(Y | do(x))`; otherwise the mechanism compiler
returns `Unknown`.

## T7 Track

T7 is the planned reduction for boundary-violating hidden-variable cases:

```text
mechanism query
  -> bipartite DAG
  -> latent-projected bipartite ADMG
  -> stochastic intervention query
  -> Pearl ID backend
  -> mechanism-level expression or hedge witness
```

The current repository includes infrastructure for these intermediate objects,
a variable-level ADMG projection for the vertical slice, and a deliberately
isolated Pearl-ID backend. The top-level `identify(...)` function does not
silently route boundary-violating mechanism queries through partial T7 support;
the caller must pass `allow_t7=True`.
