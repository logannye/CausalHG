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

- `delete(m)` — remove the mechanism; its orphaned outputs fall back to a declared
  joint policy `P0^m(out(m))`.
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
    print(result.expression)   # P(A) * P(B) * P(E) * P(F | C,E) * P0_m1(C,D)
    for assumption in result.assumptions:
        print(assumption.code) # C1 C2 C3 C4 P0 Observed boundary Downstream positivity
    for step in result.derivation:
        print(step.label)      # Validate graph / Factorize / Omit target factor
```

The estimand is an AST, not a string. It renders to text and LaTeX, exposes its
scope and the primitive kernels it references, and has a canonical key for
comparison.

### Feedback loops

Most regulatory networks have one, and C1 used to reject the whole graph at construction —
a fifty-link chain with one completely disjoint two-cycle elsewhere could not be built, so
nothing about any part of it could be asked.

Lemma 1.1's proof needs acyclicity of the sub-system it is applied to, not of the ambient
graph. An ancestrally-closed set of variables is self-contained and its noises are
independent by C2, so its marginal factorizes on its own. The question is therefore whether
the *query's own closure* is acyclic — and that closure is taken on the **post-intervention**
graph, because a deletion replaces its target's factor with a policy and so severs
everything above it:

```python
graph = MechanismGraph(
    variables={"a", "b", "Y", "q", "R"},
    mechanisms={
        "m1": {"inputs": {"a"}, "outputs": {"b"}},    # m1 and m2 form a two-cycle
        "m2": {"inputs": {"b"}, "outputs": {"a"}},
        "m3": {"inputs": {"b"}, "outputs": {"Y"}},    # downstream of the cycle
        "far": {"inputs": {"q"}, "outputs": {"R"}},   # disjoint from it
    },
)

graph.cyclic_mechanisms          # frozenset({'m1', 'm2'})
graph.mechanism_components()     # (('far',), ('m1', 'm2'), ('m3',))

# the loop is in another component
identify(graph, DeleteMechanism("far", outcomes={"R"}))   # Identified: P0_far(R)

# deleting m1 severs its own input, so the loop is no longer upstream of anything needed
identify(graph, DeleteMechanism("m1", outcomes={"Y"}))    # Identified:
                                                          #   sum_{b} P(Y | b) * P0_m1(b)

# but deleting m2 still needs m1's kernel, and m1 is on the cycle
identify(graph, DeleteMechanism("m2", outcomes={"Y"}))    # Unknown
identify(graph, DeleteMechanism("far", outcomes={"Y"}))   # Unknown
```

Components come back sorted, with each component's members sorted, so a cost or a refusal
never varies with dictionary ordering.

The middle case is the one worth dwelling on. Feedback upstream of a knockdown is the
normal case in biology, and it is answerable: the intervention cuts the edge the loop would
have travelled. The last two are the limit of that. `graph.cyclic_mechanisms` is a fact
about the *observational* graph, and for a mechanism on an observational cycle
`P(out(m) | in(m))` is not its structural kernel however much the intervention severs
elsewhere — so a cycle a deletion breaks, but whose kernels the answer still needs, is
still refused. Only the ancestry walk moves to the post-intervention graph; which kernels
are trustworthy does not.

Both halves are measured. With a two-cycle downstream the estimand still matches the true
post-deletion law; with the cycle inside the closure the same machinery is **68% wrong**,
because for a mechanism on a cycle the observational conditional is not its structural
kernel — its inputs and outputs are mutually determined.

An answer on a cyclic graph declares what it rests on. `C1` is replaced by `C1 (local)`,
and `Solvability` is added: under C1 the law is defined by sampling in topological order,
which is total, and without C1 there is no such procedure — the law is the pushforward of
the noise through the solution of `V = F(V, U)`, which may have none or many.

That assumption carries more weight than "the law exists", and it is the one place where a
cycle reaches past the query's closure. Where solutions fail to exist, the recorded data
are the **solvable subpopulation**, and solvability is an event that depends on the
variables — so conditioning on it is a *selection*. In a measured case where a strictly
downstream cycle has no solution for 47% of noise draws, the conditional kernels of the
acyclic ancestry come out a third wrong, and the marginal of an exogenous variable upstream
of everything moves from mean 0 to mean 0.5. An acyclic closure does not protect against
it. The compiler never sees `F`, so it records this rather than checking it, exactly as it
does for C2.

Three things refuse rather than answer on a cyclic graph: `d_separated` and
`check_covariates`, because `THEOREM_T1.md`'s soundness proof rests on C1 (its Lemma 2.1
ends "C1 forbids cycles in `G_E`"), and `latent_project_to_variable_admg`, because a Pearl
ADMG is acyclic by definition.

### Three outcomes, and the difference between them is the point

`Identified` carries a formula. `Unknown` says this compiler cannot do it and what would
help. `Unidentified` says **no formula exists** — a much stronger claim, so it comes with a
witness.

Deleting a mechanism whose output is never observed is the case that separates them.
`P0^m(out(m))` is a policy over the *values* of the outputs, and relabelling a hidden
variable preserves every observed distribution while moving a policy defined on its
labels — so two models can agree on everything measurable and disagree on the answer:

```python
identify(graph, DeleteMechanism("m_h", outcomes={"Y"}), allow_t7=True)
# Unidentified: relabelling ['h'] leaves every observed distribution unchanged and
#               changes the policy, so no formula in the observed law can answer this.
# witness.hidden_outputs        ('h',)
# witness.observed_descendants  ('Y',)
```

If that hidden output reaches *nothing* observed, the verdict flips completely: the caller
declared the policy over every output, so its marginal over the observed ones is a sum over
a table already in hand, and the query is identified. One consumer of the hidden variable
is the whole difference.

The witness has a stated limit rather than a hidden one. It needs the policy to
distinguish the labels it permutes, so a permutation-invariant policy escapes it, and a
declared `output_equalities` group containing an observed variable pins the labels and
withdraws it — in which case the verdict falls back to `Unknown`.

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
    fallback_variables={"A", "B", "E", "F"},   # no P0 policy covering C or D
)

result = identify(graph, DeleteMechanism("m1"))

result.status              # 'unknown'
result.reason              # 'Mechanism deletion would orphan outputs without a
                           #  declared fallback policy.'
result.missing_variables   # ('C', 'D')
result.suggestions[0]      # 'Declare the joint fallback policy P0_m1(C,D) over
                           #  the orphaned outputs.'
```

Three outcomes exist: `Identified`, `Unknown` (this compiler cannot do it, and here
is what would help), and `Unidentified` (a backend produced a non-identification
witness). Because `identify` returns the base type, the type checker will not let you
read an estimand before establishing that you have one.

### Evaluating an estimand

An estimand is only meaningful if it can be *computed*. `causal_hypergraphs.semantics`
evaluates one against a finite discrete model you supply, which is what makes the
compiler's claims testable rather than merely well-formed:

```python
from causal_hypergraphs.semantics import DiscreteModel, evaluate

model = DiscreteModel(domains=..., joint=..., fallbacks={"m1": {(0, 0): 0.5, ...}})
value = evaluate(result.expression, model, {"A": 0, "B": 1, "C": 0, "D": 0, "E": 1, "F": 1})
```

Evaluation is total or loud: an undefined quantity raises `UndefinedEstimand` rather
than quietly becoming `nan`.

### Asking about a few readouts

A full-joint query enumerates every variable's domain. Passing `outcomes` reduces the
estimand to the outcome's **ancestral closure** — every factor outside it is a conditional
summing to one, so it is dropped rather than summed:

```python
identify(graph, DeleteMechanism("m1", outcomes={"F"}))
```

The reduction is exact, and `expression.footprint()` reports what enumeration would now
cost (`scope()` says what a caller must supply; they differ at `SumOut`).

When the target mechanism falls outside the closure, the estimand collapses to
`P(outcomes)`. That is stronger than a numerical coincidence: the expression mentions
neither the mechanism nor its policy, so the answer *cannot* depend on what the
intervention installs.

### What a query costs: variable elimination

The reduction changes the exponent; it does not remove one. An ancestry of 56 variables is
as unenumerable as one of 20,000. So estimands are evaluated by **variable elimination**:
summation distributes over a product, so each inner sum is computed once and kept, and the
largest object built is a table over one *bucket* — the variables that meet at a single
elimination step — rather than over the whole ancestry.

```python
from causal_hypergraphs import plan_elimination

plan = plan_elimination(q.expression, {name: (0, 1) for name in genes})
plan.summary()
# 'eliminate 59 variable(s) at induced width 1; largest table 4 entries,
#  against 576460752303423488 assignments for enumeration'
```

Measured on a synthetic 20,000-gene sparse GRN (fan-in 3, regulators drawn from a moving
window), by hop distance from the intervention — this table is produced by
`test_a_sparse_gene_network_is_affordable_near_the_intervention_and_not_far_from_it`:

| hops | ancestry | enumeration | largest elimination table |
|---|---|---|---|
| 0 | 1 var | `2**1` | 1 entry |
| 1 | 10 vars | `2**10` | 16 entries |
| 2 | 19 vars | `2**19` | 16 entries |
| 3 | 93 vars | `2**93` | 32 entries |
| 4 | 101 vars | `2**101` | 32 entries |
| 5 | 106 vars | `2**106` | 64 entries |
| 6 | 171 vars | `2**171` | 512 entries |
| 7 | 291 vars | `2**291` | `2**24` entries |
| 8 | 380 vars | `2**380` | `2**41` entries |

All nine rows are pinned by that test, not just the three the prose leans on. Hop 0 is the
target's own output, so the estimand is the policy alone -- one variable, one cell.

Both halves matter. Near the intervention the win is total — a hundred-variable ancestry
for the price of a 64-entry table. Around seven hops the ancestries of different branches
start to overlap, the width climbs, and the query becomes unaffordable again. That wall is
real, and `eliminate` refuses by name rather than trying:

```text
IntractableQuery: Elimination needs a table of 16777216 entries (24 variable(s),
induced width 23), above the 1048576-entry bound. The variables [...] meet at one
elimination step, so widening the bound helps only if the width is nearly affordable;
otherwise ask about a narrower outcome, or measure a variable that splits the bucket.
```

`estimate(..., method="enumerate")` walks the whole footprint instead, and exists because
agreement between the two is what verifies the fast path. `estimate` reports what the query
cost in its `summary()`.

### Continuous readouts: `E[Y | do]`

A biological readout is a number, not a category, and binning it is not neutral — it can
create or destroy the data support the estimator checks. `identify_expectation` avoids the
need. The outcome appears in exactly one chain-rule factor, so it can be folded into a
conditional mean and never enumerated:

```python
q = identify_expectation(graph, DeleteMechanism("reg"), "target")
str(q.expression)
# 'sum_{TF,genotype,stim} E[target | TF] * P(genotype) * P(stim) * P0_reg(TF)'
q.expression.footprint()      # {'TF', 'genotype', 'stim'} -- 'target' is absent

data = Dataset.from_records(rows, unit="donor", measures=("target",))
estimate(q, data, fallbacks={"reg": knockdown}, bootstrap=300)
# 138.3, 95% CI [136.7, 139.8]   -- 3,000 distinct float values, never binned
```

`measures=` names numeric columns kept as real values. `E[Y | Z]` is a group mean over
matching rows, and an empty group raises with its stratum, so it lands in the same
certificate discharge as an empty conditioning cell.

That the outcome's co-outputs also drop out is a consequence of C1, not an assumption: a
co-output that were an ancestor of the outcome would make its own mechanism its own
ancestor. Two cases are refused rather than guessed — an outcome produced by the target
mechanism (its post-intervention law *is* the policy you supplied), and one whose
identifier is a quotient (no per-mechanism factor to fold into).

### Which covariates are safe to condition on

The compiler's own estimands need no adjustment set, but analysts stratify, filter, and add
regression terms anyway — and conditioning on a marker *downstream of the perturbation*
looks like ordinary covariate control while silently removing part of the effect being
measured.

```python
check_covariates(graph, DeleteMechanism("knockdown"), "IFNG",
                 ["donor", "stim", "exhaustion_marker", "batch"]).summary()
```

```text
Conditioning around do(knockdown) with outcome 'IFNG':

  Structural -- post-treatment, no distributional assumption involved:
    !! exhaustion_marker: post-treatment: reachable from 'knockdown' in the graph, so
       conditioning on it removes part of the effect being measured. Structural, not an
       assumption.

  Warning -- may open a back-door path; rests on faithfulness, so this is not a proof of
  harm:
    (none)

  Admissible: ['donor', 'stim', 'batch']
```

The two findings are kept apart because their evidence differs. **Post-treatment** is a
structural fact about the graph. **Path opening** is detected on the back-door graph — the
graph with the target's outgoing edges severed, so any surviving connection is non-causal —
by `d_separated` going from a separation verdict to a non-verdict; since that oracle is
sound but complete only under faithfulness, it is a warning rather than a proof of harm.

`admissible` means only that neither failure mode was detected. It is not a certificate
that adjusting for the covariate yields an unbiased estimate.

Both findings are checked against exact ground truth over 120 generated models — 2,596
verdicts. Post-treatment is structural, so it is verified in *both* directions against a
reachability walk written independently of the library's. Path-opening is verified in the
sound direction only: every separation the report relies on must be a real conditional
independence in the post-deletion law, which is what the back-door graph's law is, and 693
such claims are checked with an exact division-free test. The warning fires 38 times and
all 38 land on a genuine dependence — reported rather than gated, since d\*-separation is
complete only under faithfulness. A sweep reporting *zero* flags would be indistinguishable
from testing on the full graph, where the causal path keeps the mechanism and the outcome
connected whatever the covariate does, so the flag count is gated too.

### Estimating from data

`causal_hypergraphs.estimation` runs the same estimand against an actual dataset, and
**discharges the estimand's positivity certificates against that dataset**:

```python
from causal_hypergraphs import Dataset, estimate

data = Dataset.from_records(records, unit="donor")   # or Dataset.from_counts(table, names)
est = estimate(
    result, data,
    fallbacks={"m1": {(0, 0): 0.5, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.5}},
    bootstrap=500,
)

est.values[(0, 1, 1, 0, 1, 1)]   # the estimated post-intervention probability
est.interval[(0, 1, 1, 0, 1, 1)] # its 95% unit-bootstrap interval
print(est.summary())
```

Three things distinguish this from evaluating a formula:

**Certificates come due.** `identify` records `Target positivity`, `Downstream positivity`
and — on the T7/Pearl branch — `Backend positivity`, because they are properties of the
distribution, not of the graph. Against data they are checkable, and they are checked.
Which branch answered does not change that: an estimand that conditions on something while
recording no positivity certificate is an undisclosed requirement, and a test asserts that
no branch produces one. A conditioning cell with no observations
produces a named `SupportFailure`, and the affected points are *absent* from `values`
rather than present as `nan`:

```text
Checked against the data:
  Downstream positivity: FAIL
  16 of 64 point(s) undefined across 1 empty stratum/strata
  ! P(F | C,E) undefined at C=1, E=1 (16 point(s) unreachable)
```

Even when every certificate holds, `support.min_stratum_count` reports the sparsest data
cell the estimand actually read — a quotient can be perfectly well defined and still rest
on six rows.

**The unit of independence is declared, not assumed.** `unit=` names what is
exchangeable, and the bootstrap resamples those, not rows. It matters: on data with 20
donors contributing 50 correlated rows each, resampling units gives an interval **6.6×
wider** than resampling rows — close to the √50 = 7.1 the correlation implies. The default
treats each row as independent, which is the narrowest and most optimistic choice, so the
estimate always prints which one it used.

**What the data cannot check is stated first.** Positivity is discharged; `C2` — no
unmeasured confounding across mechanisms — is not, and cannot be. It is the assumption
most likely to be false and least likely to be noticed once a number is on the screen, so
`summary()` leads with the unverifiable list and only then reports what was verified.

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

- Typed mechanism graph with C3/C4 validation at construction. **C1 is a per-query
  condition, not a construction-time veto**: a cyclic graph is a legitimate object, and the
  compiler refuses only queries whose own ancestral closure reaches a cycle.
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
- Finite-discrete semantics for evaluating estimands, by enumeration (`evaluate`, the
  reference) or by variable elimination (`eliminate`, the default in `estimate`).
- `plan_elimination`: what a query will cost, and its induced width, before paying it.
- Marginal queries: `outcomes` reduces an estimand to its ancestral closure, exactly.
- Expectation functionals: `E[Y | do]` for a continuous readout, no binning.
- Covariate admissibility: `check_covariates` separates structural post-treatment
  findings from faithfulness-dependent path-opening warnings.
- A data-facing estimator: `Dataset` (+ declared unit of independence), `estimate`,
  positivity discharge against the empirical support, and unit-bootstrap intervals
  measured at 94.3% coverage over 1,568 intervals. Kernels are counted per factor, so
  no joint over the estimand's footprint is ever assembled.
- `d*`-separation on the bipartite blowup with equality-based determination closure,
  by Bayes-Ball reachability in `O(V + E)`.
- Latent projection to a Pearl ADMG, checked against the library's own Proposition T4.0:
  a mechanism has one shared noise, so its outputs are a bidirected clique and the
  districts are exactly `{out(m)}`.
- A verdict for every hidden *output* of a deleted mechanism. One that reaches an
  observation is `Unidentified` with a relabelling witness; one that reaches nothing is
  summed out of the declared policy, which identifies it.
- The **Shpitser-Pearl ID algorithm** over Pearl ADMGs: sound and complete for
  `P(y | do(x))`, returning a hedge when no formula exists. Its estimands are narrowed by
  m-separation and folded back into joint district kernels, so what comes out is estimable
  and not merely correct — a conditional on the whole topological prefix is a stratum no
  dataset has a row for.
- An opt-in `T7` reduction via `identify(..., allow_t7=True)`, for a target whose *input*
  is hidden. Multi-output targets included: `do(out(m))` is one multi-variable Pearl query.
- `minimal_model/`: a NumPy reference implementation with executable semantics for
  all three do-operators and per-mechanism counterfactual abduction.

## Not yet supported

- **Refuting** a mechanism query from a Pearl hedge. When ID fails on the projection the
  verdict is `Unknown` with the hedge attached, never `Unidentified`. Two gaps sit between
  them: Shpitser-Pearl completeness refutes identifiability over *all* semi-Markovian
  models of the ADMG, while this projection's preimage is a strictly smaller class; and the
  mechanism query is a *mixture* against a supplied policy, so every term failing does not
  make the mixture fail. Closing that is conjecture H1+, which `THEOREM_H1_PLUS.md` marks
  open.
- `replace(m, m')` under a hidden boundary. Deletion installs an unconditional policy, so
  `P(Y | delete m) = Σ P0(x) · P(Y | do(out(m) = x))`. A replacement kernel *reads*
  `in(m)`, so that identity does not apply and the reduction needs the joint
  `P(Y, in(m) | do(out(m)))`. Refused rather than approximated by the deletion identity.
- Hyper-hedge completeness. Open conjecture.
- Complete Pearl-ID beyond the currently implemented backend cases.
- Markov-kernel mechanisms; richer role typing (substrate / enzyme / product);
  mechanism-correlated noise.
- Identification *inside* a cycle. Not a gap in the implementation: on the two-cycle the
  models sharing an observational law form a curve along which the post-deletion variance
  is unbounded and the covariance takes both signs, so there is nothing in `P(V)` to
  identify the answer with. σ-separation and the simple-SCM machinery (Forré & Mooij;
  Bongers et al.) would be the route to the cases that *are* identifiable.
- A cycle strictly *upstream* of the intervention. The deletion severs it, so the query
  looks answerable, and it would be a valuable case — feedback upstream of a knockdown is
  the normal situation in biology. It is refused because the ancestral reduction computes
  its closure on the observational graph and would keep the cyclic factors, whose product
  does not integrate away (`Σ P(x|y)P(y|x)` runs over `[1, 2]`). Taking it needs the
  reduction rebuilt on the post-*intervention* law.
- Queries wider than their treewidth. Elimination moved the frontier a long way — a
  106-variable ancestry costs a 64-entry table — but around seven hops into a sparse GRN
  the branches' ancestries overlap, the induced width passes 20, and the query is
  unaffordable again. `eliminate` refuses by name rather than attempting it. Getting past
  that needs approximation (loopy propagation, sampling) or conditioning, which would
  trade the exactness this library is built on and is not implemented.
- A better elimination order. Min-fill is the standard heuristic, not an optimal order,
  and its own cost is quadratic in the ancestry — so on a very wide closure the *ordering*
  becomes the bottleneck before the elimination does.
- Continuous *conditioning* variables. Outcomes may be continuous via `E[Y | do]`;
  anything conditioned on must still be binned by the caller, deliberately.

## Status and known gaps

The suite is `317 passed, 1 xfailed`, with ruff and CI on Python 3.11 and 3.13.

Correctness is established by a randomized differential harness (`tests/conformance/`)
rather than by comparing rendered strings. It generates models satisfying C1–C4 with
strictly positive, structurally sparse, and singular ("all outputs equal") kernels,
computes exact interventional laws the compiler never sees, and checks the compiler
against them. On the current sweep:

- **Identifiers.** 5,504 queries over 220 generated models — the full joint, plus one
  marginal query per observed variable; 4,765 identified across all five theorem branches
  (`T2` 1,070 / `T3` 841 / `T4` 1,081 / `T4.1` 959 / `T6` 814), 4,606 verified pointwise
  against the exact interventional law, and 159 skipped because a positivity assumption the
  result *itself records* fails in that model. An estimand that cannot be evaluated while
  recording no such assumption is a failure, not a skip. The branch mix is gated: a branch
  `_theorem` can return that no model reaches fails the sweep, and the population is parsed
  from the compiler rather than listed here.
- **Marginal queries.** 4,528 of those queries pass `outcomes=`, and 3,818 are verified
  against the law marginalized to that outcome. This lane is what exercises the ancestral
  reduction. Until it existed the sweep issued only full-joint queries, so the reduction
  was covered by a single dedicated file — and the gap was not theoretical: reinstating the
  co-output defect that reduction shipped with leaves the sweep green without these cases
  and produces 268 nonconforming estimands with them. The gate requires the marginal lane
  to carry most of the verified queries, not merely to exist.
- **`d*`-separation.** 5,400 (X, Y, Z) triples, **zero unsound verdicts** — every
  claimed separation is an actual conditional independence in the model's own law. On
  models with strictly positive kernels, where faithfulness is generic, there were also
  zero *missed* independences over 5,400 triples, so the criterion is complete where its
  hypothesis holds.

The harness is itself tested against deliberately wrong estimands, an oracle that
separates everything, and a faithful reconstruction of the historical partial-determination
bug — which it catches 63 times. Results are byte-identical across `PYTHONHASHSEED`
values.

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
  identification/   T2/T3/T4/T6 compilers, latent projection, Shpitser-Pearl ID, T7
  separation/       d*-separation and determination closure
  semantics/        finite-discrete evaluation: enumeration and variable elimination
  estimation/       datasets, factored empirical model, estimation, certificate discharge

minimal_model/      NumPy reference implementation
tests/              compiler, semantics, and separation tests
tests/conformance/  model generator and exact-ground-truth checkers
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

The patent grant is the reason for Apache-2.0 over MIT: the identification
results here are methods, and a permissive licence that is silent on patents
leaves a user guessing about the one right they most need.
