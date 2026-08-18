# Causal Hypergraphs

A Python research library for **mechanism-level causal identification**.

Most causal tooling asks what happens when you set a *variable* to a value. A lot of
real interventions do not work that way. You knock out a reaction, swap an enzyme,
inhibit a pathway, replace an ETL step, change a policy rule. The thing you
manipulate is a **mechanism** — a process with several inputs and several jointly
produced outputs — and the variables move as a consequence.

Causal Hypergraphs makes that mechanism a first-class object. You describe a system
as typed hyperedges, ask a mechanism-level question, and get back either an
identified estimand with the theorem and assumptions it rests on, or an explicit
refusal that tells you what is missing.

> **Status: research software, pre-1.0.** The formal core is deliberately narrow and
> the compiler refuses outside it. See [Status and known gaps](#status-and-known-gaps)
> before relying on it. Not for scientific or clinical claims without independent
> validation of the graph, the assumptions, and the estimand.

---

## The idea

A mechanism `m` has an input boundary and an output boundary:

```text
m: in(m) -> out(m)
```

Its outputs are produced *jointly*, from shared noise. That is the part a
variable-level DAG cannot say. A reaction `A + B -> C + D` produces `C` and `D`
together with their rates linked by stoichiometry; a five-subunit complex binding is
not ten pairwise interactions; a drug ablating an enzyme acts on a *reaction*, not on
a *molecule*.

Under the v1 assumptions the observational law factorizes one factor per mechanism:

```text
P(V) = prod_{v exogenous} P(v) * prod_{m} P(out(m) | in(m))
```

and a mechanism intervention is a **local factor swap** — delete one factor and put
something else in its place. Two operations are supported:

- `delete(m)` — remove the mechanism; its orphaned outputs fall back to declared
  laws `P0(v)`.
- `replace(m, m')` — keep the wiring, change the function. Requires
  `rho(m') = rho(m)`: same inputs, same outputs.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+. The only runtime dependency is NumPy.

## Quickstart

```python
from causal_hypergraphs import DeleteMechanism, Identified, MechanismGraph, identify

graph = MechanismGraph(
    variables={"A", "B", "C", "D", "E", "F"},
    mechanisms={
        "m1": {"inputs": {"A", "B"}, "outputs": {"C", "D"}},
        "m2": {"inputs": {"C", "E"}, "outputs": {"F"}},
    },
)

result = identify(graph, DeleteMechanism("m1"))

if isinstance(result, Identified):
    print(result.theorem)      # T2
    print(result.expression)   # P(A) * P(B) * P(E) * P(F | C,E) * P0(C) * P0(D)
    for assumption in result.assumptions:
        print(assumption.code) # C1 C2 C3 C4 P0 Observed boundary Downstream positivity
    for step in result.derivation:
        print(step.label)      # Validate graph / Factorize / Omit target factor
```

The estimand is an AST, not a string. It renders to text and LaTeX, exposes its
scope and the primitive kernels it references, and has a canonical key for
comparison.

### Refusal is a first-class outcome

The compiler will not return a plausible-looking formula it cannot justify. Declaring
fallback laws for everything *except* `m1`'s outputs turns the same query into a
refusal that names the gap:

```python
graph = MechanismGraph(
    variables={"A", "B", "C", "D", "E", "F"},
    mechanisms={
        "m1": {"inputs": {"A", "B"}, "outputs": {"C", "D"}},
        "m2": {"inputs": {"C", "E"}, "outputs": {"F"}},
    },
    fallback_variables={"A", "B", "E", "F"},   # no P0 for C or D
)

result = identify(graph, DeleteMechanism("m1"))

result.status              # 'unknown'
result.reason              # 'Mechanism deletion would orphan outputs without a
                           #  declared fallback policy.'
result.missing_variables   # ('C', 'D')
result.suggestions[0]      # 'Declare fallback distribution P0(C).'
```

Three outcomes exist: `Identified`, `Unknown` (this compiler cannot do it, and here
is what would help), and `Unidentified` (a backend produced a non-identification
witness). Because `identify` returns the base type, the type checker will not let you
read an estimand before establishing that you have one.

### Evaluating an estimand

An estimand is only meaningful if it can be *computed*. `causal_hypergraphs.semantics`
evaluates one against a finite discrete model, which is what makes the compiler's
claims testable rather than merely well-formed:

```python
from causal_hypergraphs.semantics import DiscreteModel, evaluate

model = DiscreteModel(domains=..., joint=..., fallbacks={"C": ..., "D": ...})
value = evaluate(result.expression, model, {"A": 0, "B": 1, "C": 0, "D": 0, "E": 1, "F": 1})
```

Evaluation is total or loud: an undefined quantity raises `UndefinedEstimand` rather
than quietly becoming `nan`.

## Two forms of the same identifier

This is the one piece of internal detail worth knowing, because it determines when a
query works.

A factor swap can be written two ways, which agree wherever both are defined but do
**not** have the same domain:

```text
quotient form:  P(V) / P(out(m) | in(m)) * <new factor>
kernel form:    prod_exo P(v) * prod_{m' != m} P(out(m') | in(m')) * <new factor>
```

Assumption C2 gives mechanisms deterministic structural functions driven by exogenous
noise. When that noise carries fewer degrees of freedom than `|out(m)|` — which is
what stoichiometric coupling *is* — the factor `P(out(m) | in(m))` is **singular**.
Deleting such a mechanism moves probability onto configurations the observational law
never visits, so the quotient is `0/0` on exactly the region the intervention creates.

The compiler therefore emits the kernel form whenever every chain-rule factor is
observational, and falls back to the quotient only when hidden variables leave no
alternative — recording a `Target positivity` certificate when it does.

## Assumptions

The formal core is narrow on purpose. If these fail, the compiler refuses rather than
returning an unsound estimand.

| | |
|---|---|
| **C1** | The mechanism dependency graph is acyclic. |
| **C2** | Mechanisms have independent exogenous noise. |
| **C3** | Mechanisms have input/output role typing. |
| **C4** | Each variable has at most one producing mechanism. |

C1, C3 and C4 are checked when the graph is constructed. C2 is semantic rather than
structural, so it is recorded as an assumption certificate on every result.

## How this relates to Pearl

**It is not more expressive.** Pearl with enough latent variables is universal, and
this framework reduces to a Pearl DAG through a bipartite blowup — mechanisms become
nodes. The documents prove that reduction and then use it.

More precisely: under C1–C4, the districts (c-components) of the latent-projected
bipartite ADMG are exactly the mechanism output sets `{out(m)}`
(`THEOREM_T4_T5.md`, Proposition T4.0). So the mechanism-level chain rule *is* the
Tian-Pearl c-component factorization for this graph class, and a mechanism
intervention is exactly one that replaces a complete district's kernel. That is why
mechanism deletion is always identifiable under full observability where a variable
intervention need not be: C4 aligns the intervention unit with the district, and
there is no such thing as a mechanism intervention on part of one.

Against the soft-intervention literature — Tian & Pearl (2001) on mechanism change,
Correa & Bareinboim (2020) on the sigma-calculus — this framework's intervention space
is a **subset**, not a superset: a sigma-intervention may change a mechanism's parent
set, while `replace(m, m')` pins the incidence.

What the library offers is therefore not new identification power. It is that the
well-behaved unit has a name in the object language, so a query is a single typed
operation rather than a multi-variable translation, and the answer arrives with its
theorem, its assumptions, and its derivation attached.

## What is implemented

- Typed mechanism graph with C1/C3/C4 validation at construction.
- `delete` and `replace` queries; `T2`/`T3` (full observation), `T4`/`T4.1` (latent
  mechanisms), `T6` (hidden variables, observed target boundary).
- Kernel-form identifiers where available; quotient plus an explicit positivity
  certificate where not.
- Expression AST: canonical products, scope and kernel introspection,
  marginalization, text and LaTeX rendering.
- Proof-carrying results with assumptions and derivation steps, and structured
  refusals with suggested next steps.
- Verified replacement incidence when a `Mechanism` is supplied, which discharges the
  `rho(m') = rho(m)` certificate into a derivation step.
- Finite-discrete semantics for evaluating estimands.
- `d*`-separation on the bipartite blowup with equality-based determination closure,
  by Bayes-Ball reachability in `O(V + E)`.
- A deliberately isolated Pearl-ID backend (observational marginals, Markovian
  truncated factorization, the canonical front-door pattern) and an opt-in `T7`
  front-door slice via `identify(..., allow_t7=True)`.
- `minimal_model/`: a NumPy reference implementation with executable semantics for
  all three do-operators and per-mechanism counterfactual abduction.

## Not yet supported

- Complete `T7` Pearl-ID reduction. The opt-in slice handles single-output mechanism
  deletion only — which is the Pearl-degenerate case, not the hypergraph one.
- Hyper-hedge completeness. Open conjecture.
- Complete Pearl-ID beyond the currently implemented backend cases.
- Cyclic mechanism graphs; Markov-kernel mechanisms; richer role typing
  (substrate / enzyme / product); mechanism-correlated noise.
- **Joint fallback kernels.** `P0` is a product of per-variable laws, so `delete(m)`
  necessarily renders `out(m)` mutually independent, and a post-deletion law that
  preserves coupling among the orphaned outputs cannot be expressed. The
  generalization to `P0^m(out(m))` is stated in `THEOREM_T2_T3.md` Remark T3.3 and
  every result goes through with it.
- Estimators. The discrete semantics evaluates an estimand against a model you
  supply; nothing estimates the primitive kernels from samples.

## Status and known gaps

The suite is `125 passed, 1 xfailed`, with ruff and CI on Python 3.11 and 3.13.

Correctness is established by a randomized differential harness (`tests/conformance/`)
rather than by comparing rendered strings. It generates models satisfying C1–C4 with
strictly positive, structurally sparse, and singular ("all outputs equal") kernels,
computes exact interventional laws the compiler never sees, and checks the compiler
against them. On the current sweep:

- **Identifiers.** 750 identified queries across all five theorem branches
  (`T2`/`T3`/`T4`/`T4.1`/`T6`); 716 verified pointwise against the exact interventional
  law, and 34 skipped because a positivity assumption the result *itself records* fails
  in that model. An estimand that cannot be evaluated while recording no such
  assumption is a failure, not a skip.
- **`d*`-separation.** 5,400 (X, Y, Z) triples, **zero unsound verdicts** — every
  claimed separation is an actual conditional independence in the model's own law. On
  models with strictly positive kernels, where faithfulness is generic, there were also
  zero *missed* independences over 5,400 triples, so the criterion is complete where its
  hypothesis holds.

The harness is itself tested against deliberately wrong estimands, an oracle that
separates everything, and a faithful reconstruction of the historical partial-determination
bug — which it catches 63 times. Results are byte-identical across `PYTHONHASHSEED`
values.

One gap is worth naming up front:

- **`d*`-separation is not wired into identification.** The oracle exists and is
  sound, but nothing in `identification/` consults it.

`T1` previously carried a broken step (`Fact 4b`, a claimed equality between the
declared determination closure and the true one, which fails on this framework's own
`C = D` example). It has been repaired rather than patched: soundness never needed that
equality, and no longer refers to it. The proof now turns on two separate conditions,
which degrade in opposite ways —

- **validity** — every declared `output_equalities` group really is equal. Soundness
  needs only this. The compiler cannot check it, since it never sees the structural
  functions, so declaring a false equality yields unsound verdicts (142 of them in the
  sweep, which is how we know the hypothesis is load-bearing).
- **declaration completeness** — the declared rules capture *all* functional
  determination. Only completeness needs this, and where it fails the criterion misses
  independences rather than inventing them.

That asymmetry is the property to rely on: **under an incomplete rule set the oracle
refuses, it does not err.**

## Documentation

| File | Contents |
|---|---|
| `whitepaper.md` | Full development: formalism, T1–T7, related work, worked example. |
| `FOUNDATIONS.md` | Definitions: hypergraph SCM, typed incidence, the three do-operators. |
| `SPEC.md` | Normative compiler semantics: objects, queries, theorem dispatch, refusals. |
| `MINIMAL_EXAMPLE.md` | The reaction-network example end to end. |
| `THEOREM_T1.md` | `d*`-separation soundness and completeness. |
| `THEOREM_T2_T3.md` | Mechanism chain rule; deletion and replacement identifiers. |
| `THEOREM_T4_T5.md` | Latent mechanisms; districts; reduction of variable interventions. |
| `THEOREM_H1_PLUS.md` | Hidden variables (`T6`), the `T7` track, the hyper-hedge conjecture. |

## Development

```bash
python -m pytest -q
python -m ruff check .
python -m pyright
```

## Repository layout

```text
src/causal_hypergraphs/
  graph/            typed incidence and validation
  expression/       probability expression algebra
  identification/   T2/T3/T4/T6 compilers, Pearl-ID backend, T7 track
  separation/       d*-separation and determination closure
  semantics/        finite-discrete evaluation of estimands

minimal_model/      NumPy reference implementation
tests/              compiler, semantics, and separation tests
tests/conformance/  model generator and exact-ground-truth checkers
```
