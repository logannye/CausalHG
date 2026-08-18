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
- `F0` is the set of variables covered by a declared fallback policy. The policy
  itself is per-mechanism and joint: `P0^m(out(m))`, one kernel over all of `m`'s
  outputs, not a product of per-variable laws.

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

`DeleteMechanism(m)` denotes replacing mechanism `m` with a fallback policy over
its outputs:

```text
delete(m): remove P(out(m) | in(m)), insert P0^m(out(m))
```

`P0^m` is a single joint kernel over `out(m)`. It is deliberately not a product of
per-variable laws: deletion orphans all of `m`'s outputs simultaneously, and a product
would force them independent, which rules out removing a mechanism whose outputs stay
coupled afterwards. A product policy remains expressible as a joint kernel that happens
to factorize, so nothing is lost. `P0^m` does not condition on `in(m)` -- deletion
removes the mechanism, so its inputs no longer act; a policy that still reads the inputs
is a replacement, not a deletion.

Deletion is not defined unless every output in `out(m)` is covered by a declared
fallback policy. Otherwise the compiler returns `Unknown` with the missing output
variables.

### Mechanism Replacement

`ReplaceMechanism(m, m_prime)` denotes replacing the local factor for `m` with a
new mechanism factor with the same incidence:

```text
replace(m, m_prime): remove P(out(m) | in(m)), insert P_m_prime(out(m) | in(m))
```

`replacement` may be a bare name or a `Mechanism`.

- Given a **name**, the compiler has nothing to check. It records
  `rho(m') = rho(m)` as an assumption certificate.
- Given a **`Mechanism`**, the compiler verifies the incidence and discharges
  that certificate into a `Verify replacement incidence` derivation step. A
  mismatch raises `ValueError`: `do(m -> m')` has no semantics when the
  replacement does not share the target's input and output boundary, so this is
  an ill-formed query rather than an unidentified one, and it is rejected the
  same way a C1/C4 violation is rejected at graph construction.

The replacement name is rendered into the estimand and its LaTeX, so it must
match `[A-Za-z_][A-Za-z0-9_'-]*`.

## Expression Language

Compiler outputs are expression ASTs, not strings. The v1 expression language
contains:

- `Probability(variables, given=...)` for observational kernels.
- `MechanismFactor(name, variables, given=...)` for named mechanism kernels.
- `ReplacementFactor(name, variables, given=...)` for replacement kernels.
- `Fallback(mechanism, variables, marginalized=())` for the joint deletion policy
  `P0^m(out(m))`. One factor over all of the deleted mechanism's outputs, not one per
  variable. `marginalized` names outputs summed out of the declared table *inside* the
  node, so they are neither free nor bound and require a domain from nobody -- the sum
  runs over the table's own keys. That is how a mechanism may have a hidden output that
  no observable depends on.
- `ConditionalExpectation(target, given=...)` for `E[target | given]`. The target is
  integrated inside the node, so it is neither free nor bound and its domain is never
  enumerated — which is what allows it to be continuous.
- `Product([...])` for commutative products.
- `Quotient(numerator, denominator)` for local factor removal.
- `SumOut(variables, expression)` for marginalization.

Expressions must expose:

- a canonical key for stable equality and hashing,
- `scope()` — the variables an assignment must bind to evaluate the expression,
- `footprint()` — every variable evaluation ranges over, bound ones included. The two
  differ exactly at `SumOut`, and conflating them is a bug: `scope` says what a caller
  supplies, `footprint` says what enumeration costs.
- the conditioning variables,
- the primitive kernels referenced by the expression,
- plain-text and LaTeX renderings.

An estimand may be evaluated by enumeration over its footprint or by variable
elimination; both must return the same value, and the same kernel cells must be read
either way, since the positivity certificates that come due are defined as the cells the
evaluator touched.

String rendering is for human inspection only. Tests that assert compiler
semantics should prefer AST properties when possible.

## Theorem Dispatch

Let `O` be the observed-variable set used for identification. If the caller does
not override `observed_variables`, the graph's `observed_variables` field is
used.

### Singular mechanism factors and the choice of estimand form

A mechanism intervention is a local factor swap in the Lemma 1.1 factorization.
There are two ways to write that swap, and they are **not** interchangeable.

```text
quotient form:     P(V) / P(out(m) | in(m)) * <new factor>
kernel form:       product_{v exogenous} P(v)
                   * product_{m' != m} P(out(m') | in(m'))
                   * <new factor>
```

They agree wherever both are defined. They differ on their domains.

C2 posits deterministic structural functions driven by exogenous noise. When a
mechanism's noise carries fewer degrees of freedom than `|out(m)|`, its factor
`P(out(m) | in(m))` is **singular**: it is supported on a proper subset of
`Dom(out(m))`. This is not a corner case — it is the framework's motivating
case, since stoichiometric coupling among jointly produced outputs is exactly a
functional dependence among them.

Intervening on such a mechanism moves probability mass onto configurations the
observational law never visits. The quotient form is `0/0` on precisely that
region: the region the intervention creates. The kernel form never divides by
the target factor and is defined wherever the post-intervention law puts mass.

The compiler therefore emits the **kernel form whenever it can**, and records
the weaker positivity condition it still needs (`Downstream positivity`). The
quotient form is used only when hidden variables make the kernel form
unavailable, and that path carries an explicit `Target positivity` certificate.

### T2: Full-Observation Deletion

If all variables are observed and no mechanism is marked latent:

```text
P(V | delete(m)) =
  product_{v exogenous} P(v)
  * product_{m' != m} P(out(m') | in(m'))
  * P0^m(out(m))
```

The result theorem is `T2`. Every factor is an observational quantity because
every variable is observed, which is what makes the kernel form available here.

### T3: Full-Observation Replacement

If all variables are observed and no mechanism is marked latent:

```text
P(V | replace(m, m_prime)) =
  product_{v exogenous} P(v)
  * product_{m' != m} P(out(m') | in(m'))
  * P_m_prime(out(m) | in(m))
```

The result theorem is `T3`.

### T4: All Variables Observed With Latent Mechanisms

If all variables are observed and at least one mechanism is marked latent, the
same kernel form is valid for deletion. A latent mechanism has an unknown `f`,
not hidden variables, so its chain-rule factor is still an observational
conditional over observed variables. The result theorem is `T4`.

Replacement in this setting currently returns theorem label `T4.1`.

### T6: Hidden Variables With Observed Target Boundary

If hidden variables exist but the target boundary is fully observed:

```text
boundary(m) subset O
```

then surviving factors may reference hidden variables and are not individually
identified, so the kernel form is unavailable. Because `boundary(m)` is
observed, the target factor does not depend on the hidden variables and factors
out of the marginalization over them:

```text
P(O) = P(out(m) | in(m)) * R(O),
  where R(O) = sum_H product_{m' != m} P(out(m') | in(m')) * product_exo P(v)
```

Since `P0^m` and the replacement factor also do not depend on `H`:

```text
P(O | delete(m))          = R(O) * P0^m(out(m))
P(O | replace(m, m'))     = R(O) * P_m_prime(out(m) | in(m))
```

`R(O)` is reachable only as `P(O) / P(out(m) | in(m))`, so this route genuinely
requires the target factor to be strictly positive wherever the
post-intervention law puts mass. That is recorded as the `Target positivity`
assumption. It is a semantic condition, not checkable from incidence, and it
**fails** for a deterministic mechanism with functionally coupled outputs. The
result theorem is `T6`.

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
P(Y | delete(m_x)) = sum_x P0^m_x(x) * P(Y | do(x))
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
