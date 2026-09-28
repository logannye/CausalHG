# Supported capabilities in 1.0

A successful graph import establishes structure only. Causal assumptions, identification,
statistical models, and executable structural functions have separate contracts.

| Capability | Public interface | Supported scope and refusal boundary |
|---|---|---|
| Generic incidence | `DirectedHypergraph`, `DirectedHyperedge` | Immutable string IDs, named parallel edges, isolated nodes, cyclic incidence, multiple producers, finite JSON metadata. Does not imply causality. |
| Causal profile | `CausalModelSpec`, `MechanismGraph` | Independent noise across roots/mechanisms; one producer per variable; joint co-output noise permitted. Unsupported profiles and invalid causal incidence are refused. |
| Unified queries | `CausalQuery`, `EffectContrast`, `compile_query` | Distributions, numeric finite means, paired differences, justified baseline conditioning. New compiler requires global acyclicity. |
| Interventions | `HardIntervention`, `JointPolicy`, `Delete`, `Replace`, `Composite` | Partial co-output surgery, unconditional joint finite policies, same-boundary replacement, disjoint compositions. Conditional policies/overlapping operations unsupported. |
| Full observation | `compile_query` | Independent-mechanism factorization, joint-output marginal surgery, ancestral reduction. Missing required positivity is an estimation issue. |
| Hidden-variable queries | `compile_query` | Observed hard/policy targets through latent projection and ID; no hidden replacements or overlapping projected target/outcome queries. Projection failures are conservatively `Unknown` for mechanism/policy models. |
| Legacy mechanism reductions | `identify`, `DeleteMechanism`, `ReplaceMechanism` | Existing T2–T7 routes retained; hidden-input deletion is opt-in `allow_t7=True`; hidden-output scope and invariance matter. Inspect the returned result. |
| Ordinary ADMG identification | `ADMG`, `identify_effect`, `compile_query` | Shpitser–Pearl ID for ordinary variable interventions. Applicable hedges produce `Unidentified`; a failed policy mixture or restricted model reduction need not. |
| DAG adjustment | `validate_adjustment_set` | Entire-set sufficient backdoor check, including descendants and collider opening. Bidirected, hidden, joint-output, and cyclic inputs return unsupported. Failing this criterion is not nonidentification. |
| Graphical diagnostics | `d_separated`, `check_covariates` | Acyclic bipartite separation and limited covariate hazards. An individual “admissible” covariate is not a complete adjustment certificate. |
| Exact finite evaluation | `evaluate_query`, `semantics.evaluate`, `eliminate` | Explicit domains and joint kernels; reference enumeration and budgeted variable elimination. Opaque nested integration/contrasts have conservative footprint guards. |
| Finite estimation | `Dataset`, `estimate_query`, legacy `estimate` | Empirical kernels, support reports, policy concentration, paired unit-bootstrap contrasts. Missing/nonfinite data are rejected. Continuous measures remain available through legacy expectation queries with finite conditioning. |
| Continuous estimation | `fit_linear_gaussian`, `estimate_continuous` | Fully observed acyclic affine mechanisms, retained joint residual covariance, analytic means/contrasts; Gaussian baseline conditioning; numeric discrete roots for unconditional means. Rank deficiency and unsupported mixtures/conditional discrete laws are refused. |
| Continuous uncertainty | `ContinuousEstimate` | Unit-bootstrap intervals, failed replicate counts, sampling standard error. Outcome model variance is distinct from sampling error; analytic propagation has no Monte Carlo integration error, floating-point error is not certified. |
| Simulation | `HypergraphSCM`, `StructuralMechanism`, `FiniteKernel`, `LinearGaussianMechanism` | Joint observational/intervention sampling, explicit RNG, validated incidence, singular Gaussian noise, replayable provenance. All executable models must be acyclic. |
| Counterfactual simulation | `HypergraphSCM.counterfactual` | Hard interventions with observational noise or complete factual observations and valid deterministic abduction. Kernel-only, stochastic-policy and replacement counterfactuals require extra coupling and are refused. |
| Persistence | `io.dumps/loads`, `to_dict/from_dict` | Versioned data-only graphs, queries, estimands, identification/estimation results and supported noise payloads. No executable functions, unknown tags, or nonfinite numbers. |
| Interoperability | `io` adapters | Native incidence; NetworkX bipartite and XGI directed round trips; HyperNetX import with explicit roles; DataFrame/array records. Metadata/identity restrictions are explicit. |
| Legacy local-cycle identification | `identify` | Stored cycles allowed, but required observational cyclic kernels cause refusal. Some local cases declare extra solvability assumptions. This is not general cyclic inference. |

Continuous estimation does not interpret densities as masses or evaluate arbitrary finite
ASTs. A fitted affine model requires its own statistical assumptions; residual diagnostics
and synthetic benchmark performance cannot establish them for a real dataset.

`Identified` contains an estimand under assumptions. `Unknown` reports a supported
compiler's limitation. `Unidentified` includes a model-class-specific witness. Invalid
inputs, missing kernels, unsupported estimators, and resource limits use distinct
exceptions; they are not evidence of causal nonidentification.

Assumptions record `declared`, `structurally_checked`, `empirically_assessed`, or
`unresolved` states. Empirical support checks do not prove population positivity.
Policy-weight diagnostics use a documented heuristic threshold. Derivations are
inspectable records; hyper-hedge completeness remains an open research question.

General correlated-noise/multiple-producer inference, arbitrary feedback, history-dependent
policies, unrestricted counterfactual identification, discovery, interference, transport,
and neural mechanisms remain extensions beyond this release. See [semantics](semantics.md),
[continuous scope](continuous.md), and [migration](migration.md).
