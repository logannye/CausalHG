# CausalHG

**Causal inference for directed hypergraphs, with joint mechanisms as first-class objects.**
CausalHG separates graph structure, causal assumptions, identification, estimation,
and simulation. Use it to ask what changes when you intervene on a variable, replace
a processing step, or reset several jointly produced outputs under a new policy.

The library is domain independent: software pipelines, manufacturing, policy models,
and other systems can use the same API. It is a Python library with optional numerical
and graph integrations. It has no experiment-design workbench or required service.

**Version 1.0** establishes the supported contracts below. It does not identify arbitrary
hypergraph models: causal roles and assumptions must be declared, and unsupported queries
are reported explicitly. Read the [capability matrix](docs/capabilities.md).

## Install

Python 3.11 or newer, from this repository or a downloaded release artifact:

```bash
python -m pip install .                 # lightweight core
python -m pip install ".[numerical]"   # continuous estimation and Gaussian simulation
python -m pip install ".[interop]"     # NetworkX, XGI, HyperNetX, pandas adapters
```

The distribution is named `causal-hypergraphs`; import `causal_hypergraphs`.
Core graph, identification, finite evaluation, and structural simulation use only the
standard library. No PyPI publication is implied by the GitHub release.

## Identify and estimate a mechanism intervention

A processing step consumes `A, B` and produces `C, D` jointly. A downstream step
consumes `C, E` and produces `F`. Delete the first step and reset its two outputs
using an explicit joint policy:

```python
from causal_hypergraphs import (
    CausalQuery, Delete, Identified, JointPolicy, MechanismGraph, compile_query,
)

graph = MechanismGraph(
    variables={"A", "B", "C", "D", "E", "F"},
    mechanisms={
        "m1": {"inputs": {"A", "B"}, "outputs": {"C", "D"}},
        "m2": {"inputs": {"C", "E"}, "outputs": {"F"}},
    },
)
policy = JointPolicy(("C", "D"), {(0, 0): 0.75, (1, 1): 0.25})
query = CausalQuery(("F",), Delete("m1", policy))
compiled = compile_query(graph, query)
assert isinstance(compiled.result, Identified)
print(compiled.result.expression)
print([(a.code, a.state) for a in compiled.result.assumptions])
```

Policy coordinates follow the declared order; omitted tuples have zero mass.
Deletion uses the whole joint policy. `Replace("m1", "new_kernel")` instead binds a
separately supplied kernel with the same input/output boundary. `HardIntervention`
changes only its named variables, preserving the original equations/noise of their
co-outputs. `Composite` combines compatible operations and rejects conflicts.

Now create synthetic observations. `C` and `D` share the first step's noise, while
the downstream step has its own error:

```python
from itertools import product
from causal_hypergraphs import Dataset, estimate_query

records = []
for a, b, e, u, v in product(
    (0, 1), (0, 1), (0, 1), (0, 0, 0, 1), (0, 0, 0, 1)
):
    c = a ^ b ^ u
    records.append({"A": a, "B": b, "C": c, "D": c, "E": e, "F": (c & e) ^ v})

data = Dataset.from_records(records)
estimated = estimate_query(compiled, data)
assert estimated.values[(1,)] == 0.3125
assert estimated.support.holds
print(estimated.summary())
```

Finite evaluation is exact for the fitted empirical kernels. Sampling uncertainty,
empty strata, policy concentration, and untested causal assumptions are separate
findings. Use `Dataset.from_records(rows, unit="group_id")` for clustered observations
and `estimate_query(..., bootstrap=200, seed=42)` for unit-bootstrap intervals. Without
a unit column, rows are assumed independent. Missing data require explicit preprocessing.
Bootstrap intervals are approximate; [coverage studies](benchmarks/README.md) report
finite-sample limitations and undercoverage rather than asserting nominal calibration.

## Compare effects

An expectation contrast is one estimand; bootstrap replicates evaluate both arms on
the same resampled units:

```python
from causal_hypergraphs import EffectContrast, HardIntervention

contrast = EffectContrast(
    CausalQuery(("F",), HardIntervention({"C": 1}), kind="expectation"),
    CausalQuery(("F",), HardIntervention({"C": 0}), kind="expectation"),
)
effect = estimate_query(compile_query(graph, contrast), data)
assert effect.values[()] == 0.25
```

`given=` declares baseline conditioning variables. The compiler checks the original
graph's intervention descendants before identifying a joint law and normalizing it.
[`validate_adjustment_set`](docs/adjustment.md) checks whole covariate sets against the
sufficient DAG backdoor criterion. The older `check_covariates` remains a hazard
diagnostic and does not certify complete adjustment.

## Continuous data and executable models

[`fit_linear_gaussian`](docs/continuous.md) fits joint affine mechanisms with continuous
conditioning variables, retains residual covariance among co-outputs, and evaluates
means/contrasts by analytic moment propagation. It reports parametric assumptions,
extrapolation diagnostics, and optional paired unit-bootstrap intervals. Numeric
categorical roots are supported for unconditional linear means. Rank-deficient fits,
unsupported mixed conditional laws, and missing/nonfinite data are refused.

[`HypergraphSCM`](docs/simulation.md) binds graph mechanisms to finite joint kernels,
linear-Gaussian mechanisms, or custom structural functions. Sampling accepts an explicit
RNG or seed. Noise records enable reproducible replay and scoped hard-intervention
counterfactuals. A stochastic kernel alone does not specify counterfactual coupling;
model-based simulation is distinct from identification from observational data.

## Bring your own graph or backend

`DirectedHypergraph` stores incidence, isolated nodes, parallel named hyperedges, cycles,
and overlapping producers. `CausalModelSpec` separately declares a supported causal
interpretation; `MechanismGraph` requires a single producer and independent mechanism
noise, allowing dependence among outputs of one mechanism. Correlated roots can be
modeled as outputs of a joint root mechanism.

[Adapters](docs/interoperability.md) support native incidence records, directed NetworkX
bipartite graphs, XGI directed hypergraphs, HyperNetX membership with explicit roles,
and labeled tables/arrays. They preserve identity mappings and report limitations.
An imported association hypergraph does not automatically become a causal model.

```python
from causal_hypergraphs import dumps, loads

restored = loads(dumps(compiled))
assert restored == compiled
```

Versioned JSON preserves graphs, queries, estimands, assumptions, aliases, finite
estimates, and data-only noise records. Executable function bindings remain external.
The [backend interfaces](docs/backends.md) permit custom finite kernels and continuous
estimators without a plugin registry.

## Supported inference boundaries

| Model or task | 1.0 support |
|---|---|
| Fully observed acyclic independent mechanisms | Unified hard/policy/delete/replace/composite queries, finite distributions and means, contrasts, baseline conditioning. |
| Hidden-variable mechanisms | Selected legacy reductions; observed variable queries through the projected backend. Unsupported reductions return `Unknown`. |
| Ordinary semi-Markovian ADMGs | Separate Shpitser–Pearl ID path for variable interventions; applicable nonidentification witnesses. |
| Continuous estimation | Scoped affine joint-mechanism means; Gaussian baseline conditioning; analytic integration and unit-bootstrap uncertainty. |
| Simulation/counterfactuals | Supplied acyclic SCMs; hard counterfactuals with explicit noise or valid deterministic abduction. |
| Cycles, arbitrary correlated mechanisms, multiple producers | Structural storage; general inference remains outside the stable causal profiles. |

`Identified` means an estimand follows under recorded assumptions. `Unknown` means
this implementation cannot justify an answer. `Unidentified` requires a witness for
the stated model class. A resource limit or an unsupported estimator is a different
failure. Derivations are inspectable records, not machine-checked proofs.

Causal discovery, unrestricted counterfactual identification, general equilibrium
models, interference semantics, and hyper-hedge completeness remain research tracks.
Hypergraph notation does not claim greater expressive power than suitable SCMs with
latents. See the [semantics guide](docs/semantics.md) for precise contracts.

## Documentation and validation

- [API reference](docs/api.md), [capabilities](docs/capabilities.md), and [migration](docs/migration.md).
- [Three runnable examples](examples/README.md): software, manufacturing, and policy.
- [Release validation](docs/release-validation.md), [benchmarks](benchmarks/README.md), and [changelog](CHANGELOG.md).
- [Contributing and compatibility policy](CONTRIBUTING.md).
- [Roadmap](CAUSALHG_DEVELOPMENT_ROADMAP.md) and [implementation status](docs/implementation-status.md).
- [Foundations](FOUNDATIONS.md), [compiler research specification](SPEC.md), and [whitepaper](whitepaper.md).

Theory/reference documents record their original scope; the 1.0 capability matrix
and public API define current support. `minimal_model/` is unshipped reference code.
The original experiment application is preserved at `archive/pre-library-0.1`.

Licensed under [Apache 2.0](LICENSE). See [NOTICE](NOTICE).
