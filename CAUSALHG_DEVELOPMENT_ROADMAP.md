# CausalHG development roadmap

Prepared September 28, 2026. Proposed direction: a general-purpose Python library for causal inference over explicitly interpreted hypergraphs.

Repository: [logannye/CausalHG](https://github.com/logannye/CausalHG). Baseline reviewed: `db56c051c24be66304265973d3fa29bb187cf211` (`main`). Local checkout: [Causal Hypergraphs](</Users/logannye/Desktop/Galen Health/Causal Hypergraphs>). This document is a development proposal; it does not remove or modify repository code. Version numbers below are proposed milestones, not existing releases or delivery commitments.

**The product should let developers represent causal mechanisms, formulate interventions, identify effects, estimate them from data, and simulate explicit causal models through one coherent library.** Domain-specific workflows should be downstream applications. Experimental-design reports, screen-specific analyses, and a scientific workbench are outside this roadmap.

The existing compiler, expression algebra, identification backend, exact evaluator, and conformance tests provide a useful foundation. Develop those incrementally rather than rewrite the repository. Keep the distribution name `causal-hypergraphs` and import namespace `causal_hypergraphs` while stabilizing behavior; renaming would consume migration effort without adding capability.

**General purpose means broad application domains and extensible model families, with precise mathematical scope.** A hyperedge can mean a joint mechanism, a group of interacting units, a shared latent cause, or simply a recorded association. Those interpretations require different causal semantics. Importing a hypergraph must preserve its structure; assigning causal meaning must be a separate, explicit operation.

A production 1.0 can support a well-defined family of directed mechanism hypergraphs, ordinary DAG special cases, and a separately supported ordinary ADMG profile. General ADMGs are not special cases of the current independent-noise, single-producer mechanism factorization. The library need not solve causal inference for every possible hypergraph, every feedback system, or every hidden-variable query. A clear unsupported result is part of the API contract.

**The release sequence should be dependency-driven.** Each milestone produces usable software and has an explicit completion gate.

| Milestone | Main deliverable | Depends on | Completion gate |
|---|---|---|---|
| 0.2 | Domain-neutral package and verified baseline | Current repository | Demo removed; generic behavior and regression coverage preserved |
| 0.3 | Graph, model, query, and result contracts | 0.2 | Valid structures round-trip; semantic restrictions are explicit |
| 0.4 | Unified intervention and identification API | 0.3 | Supported queries agree with independent reference laws |
| 0.5 | Executable SCMs and scoped counterfactuals | 0.3 and intervention semantics from 0.4 | Simulation preserves joint outputs and correct intervention semantics |
| 0.6 | Extensible estimation, including continuous covariates | 0.4; 0.5 supplies validation models | Published estimator capability matrix and statistical benchmark results |
| 0.9 | Interoperability, performance, and release candidate | Interfaces established in 0.3; full release needs 0.4–0.6 | Adapter, installation, documentation, and performance gates pass |
| 1.0 | Stable supported contracts | 0.9 | Independent methodological review, migration guide, reproducible release checks |

**Phase 0 should remove the application layer while preserving the general inference machinery.** There is no large graphical workbench in the current repository: most application-specific material is in `demo/`, associated tests, and README narrative.

| Current material | Recommended action |
|---|---|
| `demo/preflight.py`, `demo/build_cache.py`, `demo/data/`, `demo/README.md` | Remove from the active repository after preserving a release/tag or other normal Git reference. No history rewriting is needed. |
| `tests/test_demo_smoke.py` | Remove the screen-specific integration tests. Extract any unique mathematical or statistical regression case into a small synthetic fixture first. |
| README preflight, Perturb-seq, and experiment-design sections | Replace with a domain-neutral introduction, capability matrix, and installation-to-estimation tutorial. |
| `identification/covariates.py`, `estimation/`, support diagnostics, bootstrap intervals | Retain. These are general causal/statistical functions, regardless of the example that motivated them. |
| `semantics/elimination.py`, expression AST, identification results and derivations | Retain and strengthen as core library infrastructure. |
| `minimal_model/` | Retain temporarily as research/reference code. Audit it before promoting any behavior into the public runtime. |
| Whitepaper and theorem documents | Keep the mathematical evidence, organize it under theory documentation as appropriate, and reconcile claims with the actual supported API. |

Start by recording the current test results, package exports, dependencies, and representative performance measurements. The README reports a test total, but the migration baseline must come from a fresh run rather than copying that number. Trace imports and documentation links before removal. The shipped package currently has no dependency on `demo/`, so cleanup should not require changing identification behavior.

Replace biological product examples with small synthetic examples spanning at least three domains: a software pipeline, a manufacturing process, and a decision/policy system. Each should have known ground truth, a many-input/many-output mechanism, and a causal question that is not merely an observational conditional mean. Include a shared-noise example that fails if outputs are incorrectly split into independent factors.

Do not remove support checks or uncertainty because they once appeared in a preflight report. Refactor human-readable report formatting away from computation where needed. Document `PolicySupport.effective_n` as the diagnostic it is; its current default floor of 30 must remain an explicit configurable heuristic, never a universal identification or estimability threshold. Finite-sample support checks must not claim to prove population positivity.

Exit gate: a clean installation and the retained tests work without the demo or its data; every removed integration assertion with general value has an independent synthetic replacement; README examples run from the installed package.

**Phase 1 should separate structural representation from causal assumptions before expanding algorithms.** The current `MechanismGraph` combines incidence, observability, fallback declarations, and model restrictions. It rejects multiple producers at construction and coerces identifiers to strings. Its frozen dataclass also contains a mutable mapping, which requires attention before introducing graph-based caches.

Establish three distinct contracts:

1. A compact incidence representation or protocol, with node IDs, hyperedge IDs, input/output incidence, and optional metadata. Existing graph libraries can implement or adapt to this contract. Preserve valid structural data even when a causal algorithm cannot use it.
2. A causal model specification that assigns interpretation, observed/hidden variables, domains, and assumptions to that structure. The initial mechanism profile retains independent mechanism noise and a single producer per variable; acyclicity remains checked at the scope required by each operation.
3. Executable mechanism bindings, added in Phase 3, that provide functions, noise distributions, or conditional kernels. Identification must continue to operate without requiring these functions.

Use explicit model profiles and algorithm capabilities, rather than growing a collection of booleans whose combinations have unclear semantics. Keep the existing Pearl ADMG backend available through its own profile; deferring correlated mechanism noise does not defer ordinary ADMG inference. Conversion between profiles requires a valid, documented reduction. An overlapping-output hypergraph should be representable; it should not silently qualify for the current factorization theorem. Multiple causal producers require an explicit aggregation/composition model or a new supported theory. Simply removing C4 validation is unsound.

Define ID policy deliberately: preserve external identifiers through a collision-free codec and mapping, or reject unsupported identifiers. Never silently collapse distinct IDs such as `1` and `"1"`. Separate unordered incidence from ordered numerical kernel axes, with explicit axis names and validation. Snapshot or freeze structural data so later external mutation cannot invalidate a compiled result.

Add a versioned JSON schema for graph specifications, queries, expression ASTs, assumptions, derivations, and result metadata. Store schema and library versions. Serialization should not execute arbitrary callables; executable mechanisms require separately registered implementations/configuration. Round-trip guarantees apply to the supported data schema, with explicit errors for unsupported metadata.

Normalize result semantics across modules. Identification retains `Identified`, `Unknown`, and `Unidentified`, with stable reason codes. An impossibility claim requires an applicable witness in the declared model class. Estimation/evaluation failures use separate reasons such as insufficient empirical support, unsupported estimator, numerical failure, and resource limit. A time or memory limit is not a statement about identifiability.

Keep assumptions in distinguishable states: declared by the caller, checked structurally, empirically assessed, or unresolved. A derivation trace is an inspectable argument tied to implemented rules; do not advertise it as a proof verified by a formal theorem prover unless that verification is actually added.

Exit gate: deterministic serialization and hashing; collision tests; immutable compilation inputs; round-trip preservation of mechanisms and joint outputs; predictable capability errors for unsupported causal profiles. Existing `MechanismGraph` users have a documented compatibility path.

**Phase 2 should turn the mechanism compiler into a coherent causal-query API.** Preserve the existing Shpitser–Pearl ID implementation and validate its integration. The ordinary variable-intervention backend already exists; rebuilding ID from scratch is not the missing work. The main gaps are unified semantics, query coverage, public contracts, and correct reductions between model families.

Separate the intervention from the requested quantity. An intervention describes what changes; a query requests a distribution, expectation, or contrast under that intervention. Reuse the same intervention object for identification, evaluation, and simulation.

| Query or operation | Initial stable commitment |
|---|---|
| Hard variable intervention `do(X=x)` | Support the validated DAG/ADMG route and a documented interpretation for individual outputs of joint mechanisms. |
| Joint stochastic intervention | Specify one joint policy over the intervention variables; preserve dependence within that policy. |
| Mechanism deletion | Require an explicit or explicitly symbolic joint fallback policy; deletion must not silently mean zeroing all outputs. |
| Mechanism replacement | Preserve the input/output boundary initially; require or declare the replacement kernel. |
| Composite interventions | Support compatible combinations first; reject conflicting assignments, overlapping replacements, or ambiguous policy order. |
| Marginal distributions and expectations | Preserve current support; expose a consistent query interface. |
| Effect contrasts | Support named comparisons such as a difference of expectations; distinguish treatment contrasts from observational differences. |
| Conditional effects and adjustment | Initially support justified baseline-conditioned effects only when a functional has been identified. Validate complete adjustment sets rather than interpreting an individual covariate diagnostic as a sufficient adjustment certificate. |

The semantics of variable intervention on a joint mechanism require an explicit design decision. A conventional intervention on one output should preserve the other output equations and their shared noise, while replacing the targeted component. In the current reference simulator, `do_node` removes the whole producing mechanism. That behavior must not become the default public meaning of ordinary `do(X=x)`.

Refactor the large `identification/api.py` into validation, query reduction, algorithm dispatch, and result construction. Use descriptive backend/capability names publicly; theorem numbers such as T6/T7 remain derivation metadata. Preserve supported behavior while splitting modules, with conformance gates on each change. Deprecate accidental public exposure of placeholders and implementation-only projection objects over a documented transition.

For hidden-variable mechanism queries, preserve conservative behavior. A hedge for a projected ADMG is not automatically a nonidentification witness for a restricted mechanism class or a supplied-policy mixture. Hidden-boundary replacement needs its own valid reduction. Ship supported cases and structured limitations; do not make resolving H1+ a dependency of the general-purpose release.

Strengthen adjustment support in stages: standard DAG cases first, then verified ADMG and joint-mechanism cases. The current `check_covariates` reports useful hazards, but its own documentation says that an “admissible” covariate does not guarantee unbiased adjustment. Rename or clarify that result before users treat it as a complete adjustment algorithm.

Exit gate: an executable model/query capability matrix; all supported routes checked against independent exact intervention laws; every unsupported combination returns the documented reason. Include confounding, front-door cases, singular joint outputs, unused outcomes, hidden boundaries, inconsistent policies, and interventions on a single co-output. Algorithmic equality and numerical equality must be tested separately.

**Phase 3 should ship executable causal models as a separate, compatible layer.** Promote useful concepts from `minimal_model/`, after fixing semantic mismatches and retaining independent test oracles.

Introduce a `HypergraphSCM` binding model structure to joint structural functions, noise samplers, and optional abduction methods. Offer finite categorical kernels and a continuous linear-Gaussian joint-output model as initial built-in implementations, with an interface for user-defined mechanisms. Functions that implement one joint mechanism must sample their outputs together.

Support reproducible observational sampling and intervention sampling. Thread an explicit RNG through the runtime; avoid global seeds. Joint deletion policies must have joint samplers—the reference simulator currently supports only factorizing per-output fallbacks. Record which distribution or structural model produced each simulation result.

Offer scoped counterfactual computation for fully specified SCMs with supplied noise, valid invertible-noise mechanisms, or an explicitly implemented posterior-abduction backend. Reuse the appropriate exogenous noise across worlds. A stochastic replacement/deletion also needs declared cross-world coupling semantics; a policy's marginal distribution alone does not determine individual counterfactuals. Do not equate model-based counterfactual simulation with nonparametric counterfactual identification from observational data.

Allow normalized stochastic kernels for sampling where supported, while distinguishing them from structural functions with cross-world noise semantics. Counterfactual tasks require stronger information; this distinction is also reflected in [DoWhy's counterfactual documentation](https://www.pywhy.org/dowhy/main/user_guide/causal_tasks/what_if/counterfactuals.html).

Exit gate: sampling agrees with exact finite distributions within stated Monte Carlo tolerances; analytic linear-Gaussian cases agree with closed-form results; node interventions preserve unaffected co-outputs; joint policies remain joint; counterfactual tests preserve factual consistency and declared noise coupling. The simulator and identifier must not share all oracle logic, which would allow the same bug to validate itself.

**Phase 4 should make estimation genuinely useful across domains.** Retain the current finite-discrete estimator as the exact/reference path. Add estimator interfaces around the primitives and functionals actually required by an identified expression, rather than passing every query to a generic regression model.

Define backend capabilities for conditional probabilities/densities, conditional expectations, conditional sampling, and numerical integration. Backends advertise which variable types and expression forms they support. Preserve multivariate output dependence; a backend that fits independent output regressions is not a general joint-mechanism backend.

Prioritize continuous conditioning variables and mixed data. Current continuous readouts are useful, but requiring callers to bin all conditioning variables is too restrictive for a general-purpose library. Start with linear-Gaussian joint mechanisms and clearly scoped regression-based g-computation. Add optional flexible estimators only with stated consistency conditions and benchmark coverage. Doubly robust/AIPW estimation is an optional addition for estimands to which it actually applies, not a universal evaluator for arbitrary ASTs.

Extend the expression system with typed integration/expectation operations where finite sums are insufficient. Separate exact evaluation from approximate numerical integration and fitted-model prediction. A continuous evaluator must not assign probability-mass semantics to densities or assume every deterministic joint mechanism has a nonsingular density.

Preserve uncertainty and support diagnostics. Report sampling uncertainty, numerical/Monte Carlo error, and untested causal assumptions separately. Resample independent units; add time-series dependence handling only with an appropriate resampling/model contract. Make missing-data handling explicit: no silent complete-case deletion, imputation, or interpretation of missingness as a causal category. Unsupported cases should fail with actionable machine-readable reasons.

Add sensitivity checks for declared perturbations of assumptions or nuisance models, plus resampling and placebo checks where applicable. Passing these checks is not proof that the graph is true or that no hidden confounding exists.

Exit gate: published simulation studies for bias, interval coverage, support failures, and misspecification across categorical, continuous, mixed, and clustered data. Set tolerances and benchmark populations before evaluating a candidate implementation. Include nonlinear failures and singular outputs so a favorable linear example cannot justify a universal claim.

**Phase 5 should make integration and maintenance as reliable as the mathematics.** Start adapters after Phase 1; they can proceed in parallel with algorithm and estimator work.

Provide native edge-list/incidence-table and JSON import/export first. Then add optional adapters for NetworkX bipartite graphs, XGI directed hypergraphs, and HyperNetX incidence data. [XGI provides directed tail/head structure](https://xgi.readthedocs.io/en/latest/api/tutorials/focus_7.html); [HyperNetX accepts incidence-oriented data and metadata](https://hypernetx.readthedocs.io/en/latest/classes/classes.html). An adapter must require explicit causal roles when the source does not supply them. XGI's directed representation permits structures outside the initial causal profile, so successful import does not imply eligibility for inference.

Every adapter returns an ID mapping and a conversion report. Preserve edge identity, multiplicity, roles, isolated nodes, and declared metadata where supported. Never silently clique-expand a joint mechanism or infer causal direction from undirected membership. Reject or explicitly mark lossy conversions. Ordinary DAG conversion must document the corresponding single-output/noise assumptions.

Add optional DataFrame/array adapters without making a specific table library the core data representation. Keep the symbolic graph/identification layer lightweight. Use optional dependency groups for numerical backends and ecosystem integrations, adding dependencies when justified by measurements and maintenance value. DoWhy is a useful interoperability/reference target for shared model classes, particularly its separation of identification from estimation; it is not an oracle for all hypergraph semantics. [DoWhy architecture](https://www.pywhy.org/dowhy/main/index.html)

Retain exact enumeration as a small-problem reference and variable elimination as the default exact executor. Benchmark graph size, hyperedge arity, induced width, domain cardinality, expression size, and backend separately. Add sparse factors, compiled-plan caching, and improved ordering only where profiling supports them. Cache keys must include immutable structure, query, domains, backend/version, and relevant assumptions. Apply explicit resource budgets, with a distinct resource-limit result. Approximation must be requested and labeled, never silently substituted for exact evaluation.

Strengthen release engineering: run the configured type checker in CI; add `py.typed`; build wheels and source distributions; test installed artifacts outside the checkout rather than relying only on editable installs; test declared Python versions and major operating systems; validate optional extras independently. Publish API references, a semantics guide, a capability matrix, runnable tutorials, a changelog, contribution guidance, and a deprecation policy.

Preserve randomized differential tests and add property-based cases for serialization, identity, graph conversions, intervention composition, and expression binding. Add performance regression cases with recorded hardware/environment and justified budgets. Include near-zero probabilities, zero-support strata, nonfinite values, singular factors, alias capture, and explicit refusal/witness cases in release validation.

Exit gate: a downstream user can install a release artifact, import a graph, declare its causal interpretation, identify and estimate a supported effect, and serialize the result using only public APIs. At least three domain-neutral end-to-end examples work without repository-relative imports. A second implementation can evaluate supported expressions through the documented backend interface.

**The package architecture should evolve around these boundaries, with existing imports supported during migration.** The following is a proposed destination, not a demand to move all files at once:

```text
causal_hypergraphs/
  graph/             incidence, graph protocols, structural algorithms
  models/            causal profiles, domains, executable SCM bindings
  queries/           interventions, requested quantities, effect contrasts
  identification/    validation, reductions, algorithm backends, witnesses
  expression/        typed estimand AST, simplification, binding, rendering
  separation/        supported graphical criteria and determination rules
  semantics/         exact and approximate expression execution
  estimation/        estimator protocols, fitted kernels, uncertainty
  diagnostics/       support, sensitivity, model checks, structured findings
  simulation/        sampling, interventions, scoped counterfactuals
  io/                versioned serialization and optional adapters
```

Avoid building a plugin registry or a second general graph library before there are concrete implementations to integrate. Start with Python protocols and explicit backend arguments. Add registration only when multiple supported implementations make it necessary.

**The 1.0 release gate is a stable and honest capability contract.** It requires usable graph interoperability; unified variable and mechanism queries within documented model classes; exact finite inference; at least one documented continuous estimation path; executable SCMs; scoped counterfactuals; diagnostics; typed/versioned public results; installable release artifacts; and independent methodological review of the supported claims.

It does not require proving hyper-hedge completeness, identifying arbitrary cyclic systems, causal discovery on arbitrary hypergraph datasets, unrestricted mediation/path-specific effects, or a neural architecture. These are separate extensions with their own contracts and evidence. Ordinary scientific examples can remain as examples later; no application workbench is needed to satisfy the release gate.

**After 1.0, broaden model families through isolated, validated research tracks.** Prioritize by demonstrated downstream demand rather than expanding all axes at once.

| Extension | Required work before a stable API |
|---|---|
| Time-indexed models and dynamic interventions | Explicit time semantics, lag restrictions, history-dependent policies, and validated longitudinal estimators. Time expansion changes the model assumptions; it is not a universal repair for feedback. |
| Cyclic/equilibrium mechanisms | Existence/uniqueness or selection semantics, supported separation/identification results, and counterexamples showing limits. |
| Correlated mechanism noise and multiple producers | Explicit latent/composition semantics and rederived supported identification rules. Storage support alone is insufficient. |
| Partial identification and sensitivity bounds | A declared model class, valid bounds, and an optimizer whose numerical results have appropriate guarantees. |
| General conditional identification / IDC extensions, path-specific and counterfactual identification | Extend beyond the baseline-conditioned functionals supported before 1.0. Specify separate query semantics and algorithms; fitted conditional predictions and SCM simulation do not establish identification. |
| Causal discovery | Explicit causal interpretation of incidence, statistical assumptions, equivalence classes/uncertainty, and recovery benchmarks. No automatic causal interpretation of arbitrary association hypergraphs. |
| Interference among units in a hypergraph | A distinct exposure/outcome semantics and estimators; membership hyperedges must not be treated as joint structural mechanisms by default. |
| Transportability and multi-environment inference | Explicit environment/selection assumptions, supported transfer criteria, and validation on shared model classes. |
| Neural mechanisms and learned approximations | Joint-output interfaces, calibration and intervention benchmarks, labeled approximation error, and preserved causal assumptions. |
| H1+ and broader completeness results | Independent theoretical review and model-class-specific witnesses; theorem progress must not be inferred from passing finite tests. |

**The first implementation sequence should consist of small, reviewable changes.** Begin with the following pull requests, keeping behavior changes separate from module moves:

1. Record the baseline and add a capability matrix covering current supported and refused cases; correct contradictory claims about Pearl ID and T7.
2. Extract synthetic regression cases, remove `demo/` and its screen-specific smoke test, and rewrite the README around the library.
3. Specify stable identification/evaluation status codes and assumption states; document diagnostic thresholds as heuristics.
4. Introduce the incidence/model-profile boundary and identifier policy while preserving existing `MechanismGraph` behavior through a compatibility layer.
5. Add versioned graph/query/result serialization and round-trip tests.
6. Write and review the unified intervention semantics, especially individual co-output intervention, joint fallback policies, and conflicts between composed interventions.
7. Implement the query frontend over existing identification backends and add independent conformance cases.
8. Deliver a public finite-discrete SCM runtime with joint sampling and shared intervention semantics; then add continuous models and scoped counterfactuals.

After the contracts in items 3–6 stabilize, interoperability and release engineering can run alongside identification and estimation. Continuous estimation can start from established backend protocols without waiting for general counterfactual identification. Assign separate engineering owners to core/API, algorithms/runtime, and statistics/integration where staffing permits; methodological review should be shared across all three.

Use milestone acceptance gates rather than a speculative completion date. A scheduling estimate should follow the baseline and semantics review: the largest uncertainties are new intervention semantics and statistical scope, not deleting the demo. Every release should increase supported useful cases while preserving the distinction between structural representation, causal identification, statistical estimation, and model-based simulation.

**The repository evidence behind this plan is specific and reproducible.** Current behavior should be checked against these files before implementing each affected change:

- [Package configuration](https://github.com/logannye/CausalHG/blob/db56c051c24be66304265973d3fa29bb187cf211/pyproject.toml): version 0.1.0, Python requirement, zero runtime dependencies, and current development tooling.
- [Graph representation](https://github.com/logannye/CausalHG/blob/db56c051c24be66304265973d3fa29bb187cf211/src/causal_hypergraphs/graph/model.py): incidence normalization, observability, fallback declarations, and C4 validation.
- [Public queries](https://github.com/logannye/CausalHG/blob/db56c051c24be66304265973d3fa29bb187cf211/src/causal_hypergraphs/identification/queries.py) and [results](https://github.com/logannye/CausalHG/blob/db56c051c24be66304265973d3fa29bb187cf211/src/causal_hypergraphs/identification/results.py): current mechanism frontend and structured outcomes.
- [Pearl ID implementation](https://github.com/logannye/CausalHG/blob/db56c051c24be66304265973d3fa29bb187cf211/src/causal_hypergraphs/identification/shpitser.py) and [T7 integration](https://github.com/logannye/CausalHG/blob/db56c051c24be66304265973d3fa29bb187cf211/src/causal_hypergraphs/identification/t7.py): existing algorithms and model-reduction boundaries.
- [Covariate checker](https://github.com/logannye/CausalHG/blob/db56c051c24be66304265973d3fa29bb187cf211/src/causal_hypergraphs/identification/covariates.py#L204-L216): diagnostic scope versus complete adjustment certification.
- [Estimator](https://github.com/logannye/CausalHG/blob/db56c051c24be66304265973d3fa29bb187cf211/src/causal_hypergraphs/estimation/estimator.py) and [data model](https://github.com/logannye/CausalHG/blob/db56c051c24be66304265973d3fa29bb187cf211/src/causal_hypergraphs/estimation/dataset.py): current finite conditioning, continuous measures, unit bootstrap, and policy-support diagnostics.
- [Reference simulator](https://github.com/logannye/CausalHG/blob/db56c051c24be66304265973d3fa29bb187cf211/minimal_model/scm.py#L266-L304): whole-mechanism node deletion and the documented restriction to factorizing fallback samplers.
- [Conformance checker](https://github.com/logannye/CausalHG/blob/db56c051c24be66304265973d3fa29bb187cf211/tests/conformance/checks.py) and [CI](https://github.com/logannye/CausalHG/blob/db56c051c24be66304265973d3fa29bb187cf211/.github/workflows/ci.yml): independent numerical checks worth preserving and release-validation opportunities.

External documentation links were checked on September 28, 2026. They support adapter and architecture recommendations; they do not establish correctness of CausalHG's mathematical claims. No new full test-suite run was performed to produce this roadmap.
