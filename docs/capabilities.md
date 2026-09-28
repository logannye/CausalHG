# Current capabilities

This matrix documents the implementation at the domain-neutral onboarding milestone
(0.2). It does not promise the future APIs in the [roadmap](../CAUSALHG_DEVELOPMENT_ROADMAP.md).
The package is research software with explicitly limited model and estimator support.

## Model and query contracts

| Capability | Current interface | Supported scope and limits |
|---|---|---|
| Directed mechanism hypergraphs | `Mechanism`, `MechanismGraph` | Named input/output boundaries; joint output blocks; observed/hidden variables; symbolic fallback coverage. |
| Model validation | `MechanismGraph` construction | Rejects missing variables, overlapping input/output roles within a mechanism, and multiple producers of one variable. Use distinct string identifiers; other identifiers are currently coerced to strings. |
| Causal assumptions | `Identified.assumptions` | Independent noise across mechanisms is declared, not inferred from data. Shared noise within a joint mechanism is allowed. |
| Mechanism deletion | `identify(graph, DeleteMechanism(...))` | Replaces the target factor with a joint, input-independent fallback policy. Explicit fallback coverage can require refusal when an output is uncovered. The default graph declares symbolic coverage for all variables. |
| Mechanism replacement | `identify(graph, ReplaceMechanism(...))` | Same input/output boundary. A supplied `Mechanism` is checked; a replacement name alone leaves incidence equivalence as an assumption. |
| Full-observation identification | `identify` | Factorization paths for deletion/replacement, including unknown mechanism functions marked latent. This does not permit arbitrary correlated mechanism noises. |
| Hidden variables with observed target boundary | `identify` | Supported T6 cases with explicit positivity conditions when a quotient is needed. Observability and query scope determine the dispatch. |
| Hidden target inputs | `identify(..., allow_t7=True)` | Opt-in deletion reduction through the Pearl backend. The returned result determines whether the particular query is supported. Hidden-boundary replacement is unsupported. |
| Hidden target outputs | `identify(..., allow_t7=True)` | Some deletions return a relabelling witness; outputs with no observed descendants may be marginalized from the supplied policy. Label-invariant policies and declared output equalities limit the witness. |
| Ordinary variable interventions | `ADMG`, `identify_effect` | Separate Pearl ADMG interface and Shpitser–Pearl ID algorithm. A hedge is interpreted within that backend's model class. It does not automatically refute a restricted mechanism query or policy mixture. |
| Marginal distributions | Query `outcomes=` | Reduces retained factors using the query's relevant ancestry. |
| Outcome expectations | `identify_expectation` | Supported factored identifiers can replace an outcome factor with a conditional mean. The current API rejects target-produced outcomes and quotient identifiers. |
| Cyclic graph storage | `MechanismGraph` | Allowed. Identification refuses when it needs kernels from observational cycles. Local supported cases record additional solvability assumptions; breaking a cycle alone is insufficient. |
| Composite and unified interventions | — | No stable unified frontend for variable, mechanism, and composed interventions yet. |

The mechanism profile is narrower than a general ADMG. Importing an association
hypergraph, choosing a direction, or observing a joint distribution does not establish
the causal assumptions used by either interface.

## Expressions, evaluation, and statistics

| Capability | Current interface | Supported scope and limits |
|---|---|---|
| Symbolic estimands | `Expression` and AST nodes | Products, quotients, kernels, sums, fallback/replacement factors, and conditional expectations; scope, footprint, canonical keys, text and LaTeX rendering. |
| Finite-model evaluation | `semantics.DiscreteModel`, `evaluate` | Explicit finite domains, joint probabilities, and supplied intervention policies/kernels. Undefined quantities raise a named error. |
| Exact variable elimination | `semantics.eliminate`, `plan_elimination` | Exact finite-domain computation with an intermediate-table budget. Cost depends on induced width and cardinalities; large queries can raise `IntractableQuery`. |
| Tabular empirical estimation | `Dataset`, `estimate` | Records or count tables; each required conditional kernel is counted on its own variables. All conditioning variables are finite categorical values. |
| Continuous outcomes | `Dataset.from_records(..., measures=...)` | Real-valued outcome means in supported expectation queries, with finite conditioning strata. No continuous-conditioning regression backend yet. |
| Uncertainty | `estimate(..., bootstrap=...)` | Bootstrap intervals with an explicitly declared sampling unit or independent rows by default. Replicate failures are reported. No general time-series resampling contract. |
| Empirical support | `Estimate.support` | Required empty strata, undefined points, and minimum stratum counts. A pass describes these data; it is not proof of population positivity or causal assumptions. |
| Policy concentration | `Estimate.policy` | Policy-weight/count diagnostic with a configurable heuristic floor. It is not a universal effective sample size or identifiability criterion. |
| Numerical policies | `fallbacks=`, `replacements=` | Caller-supplied tables, with explicit entries and ordered tuple axes. These policies describe the intervention, rather than being inferred from observational data. |
| Flexible continuous/mixed estimators | — | Not yet implemented. Binning changes the model and must be a deliberate caller choice. |

Exact evaluation and correct identification are different claims. An exact computation
over empirical kernels can still have sampling error, misspecification, poor support, or
incorrect causal assumptions. Results keep the symbolic assumptions visible alongside
empirical diagnostics.

## Separation, diagnostics, and simulation

| Capability | Current interface | Supported scope and limits |
|---|---|---|
| Graphical separation | `d_separated`, determination rules | Bipartite separation with declared deterministic equalities. Requires an acyclic mechanism graph. Soundness relies on valid declarations; completeness has additional assumptions. |
| Covariate diagnostics | `check_covariates` | Reports post-treatment variables and supported back-door path diagnostics. “Admissible” means limited hazards were not found; it does not certify an adjustment set or an unbiased estimate. Cyclic inputs are rejected. |
| Latent projection | `latent_project_to_variable_admg` | Projection within documented model assumptions; cyclic inputs are rejected. Keep mechanism-output dependence when interpreting the result. |
| Executable SCM simulation | `minimal_model/` reference only | Not shipped as the public runtime. Its node intervention deletes a whole producing mechanism, and deletion samplers support only factorizing fallbacks; do not assume ordinary componentwise intervention or general joint-policy semantics. |
| Counterfactual inference | Reference examples only | No stable public counterfactual API and no general observational counterfactual-identification algorithm. |
| Hypergraph ecosystem adapters and versioned serialization | — | Not yet provided as supported public interfaces. |
| Causal discovery, interference, transport, general feedback | — | Outside current supported algorithms. Hyperedges with these meanings need their own declared semantics. |

## Reading outcomes

- `Identified` supplies an expression under its recorded assumptions. Check the
  assumptions before evaluating it.
- `Unknown` means this implementation cannot justify an answer; it may include a
  suggested algorithm, missing variables, or another diagnostic.
- `Unidentified` carries a backend's nonidentification conclusion and witness in
  the stated model class. A failed computation is not such a witness.
- Malformed graphs/queries and unavailable numerical kernels can raise exceptions.
  Empirical support failures and resource limits concern estimation/evaluation, not
  necessarily causal identifiability.

The derivation records document applied rules and theorem references; they are not
machine-checked formal proofs. The open hyper-hedge completeness question remains
open regardless of the finite numerical test coverage.

See [SPEC.md](../SPEC.md) for compiler semantics and
[migration notes](migration.md) for changes to the public surface.
