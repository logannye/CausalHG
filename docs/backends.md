# Explicit backend interfaces

CausalHG has no plugin registry. Pass a supported backend/model explicitly and keep
its numerical capabilities separate from an identifying formula.

## Finite primitives

`causal_hypergraphs.semantics.Model` is a Python protocol exposing:

- `domains`: mapping of variable names to finite ordered value tuples;
- `conditional(variables, given, assignment)`;
- `fallback(mechanism, variables, assignment, marginalized=())`;
- `replacement(replacement, variables, given, assignment)`;
- `conditional_expectation(target, given, assignment)` when an expression needs it.

`DiscreteModel` supplies explicit joint tables; `estimation.EmpiricalModel` estimates
primitives from a `Dataset`. Both satisfy the same numerical expression contract.
`evaluate_query(compiled, model)` also binds the query's policies and identification
aliases. Missing/undefined kernels must raise the documented semantics errors, not
return zero, `nan`, or an arbitrary smoothing estimate. Kernels need not factor across
joint outputs. Ordered axes belong to numerical tables, distinct from incidence sets.

The reference `evaluate` enumerates finite domains. `eliminate` uses exact variable
elimination for supported factors, and `plan_elimination` exposes induced table cost.
The query wrappers honor `max_entries`; opaque nested integrals may be conservatively
refused even when another algorithm could exploit more structure. Approximation is
never substituted automatically. Densities are not valid finite-domain primitives.

## Continuous models

`ContinuousBackend` requires a `ContinuousCapabilities` declaration and
`estimate(query, **kwargs) -> ContinuousEstimate`. `LinearGaussianFit` is the initial
implementation, constructed by `fit_linear_gaussian`. Its capability record declares
the causal profile, requested quantities, integration method, baseline conditioning,
and supported policies. It computes affine-model moments directly after identification;
it does not send continuous densities through the finite AST evaluator.

Custom continuous estimators must state their identification, statistical consistency,
multivariate-kernel and integration assumptions, and report sampling uncertainty,
numerical error and failed fits separately. Unsupported estimands should use explicit
errors. A `.predict()` regression function alone does not satisfy a causal estimator.

## Model and policy sensitivity

Run the same identified quantity through explicitly named fitted models or replacement
kernels, then compare estimates and their recorded assumptions. `Replace` plus supplied
`LinearGaussianKernel` bindings supports declared changes to mechanism parameters.
Use the same sampling units and seeds for paired resampling comparisons. Such a sweep
answers the declared perturbation; it does not bound unmeasured confounding or certify
robustness to arbitrary misspecification. The published benchmark suite includes null
and nonlinear perturbation controls with known synthetic truth.
