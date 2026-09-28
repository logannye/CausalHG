# Changelog

## 0.6.0

- Added a typed optional continuous backend with joint affine kernels, analytic means, supported Gaussian baseline conditioning, and paired cluster-bootstrap uncertainty.
- Hardened finite and continuous input contracts: missing/nonfinite values, invalid sampling units, ragged finite tables, invalid domains and bootstrap controls are rejected.
- Published categorical, Gaussian, mixed, clustered, null and misspecified simulation studies, including unfavorable coverage results and independent interval references.
- Preserved joint residual dependence and singular output covariance; rejected rank-deficient designs and unsupported conditional mixtures.
- Corrected direct graph identifier collisions that could corrupt separation, and validated declared executable output equalities.

## 0.5.0

- Added public executable SCMs, finite joint kernels, custom structural functions, and optional linear-Gaussian mechanisms.
- Added seeded observational/intervention sampling, replayable noise records, and validated replacement bindings.
- Added scoped hard-intervention counterfactuals with preserved co-output noise; stochastic kernels require explicit coupling.
- Independent review closed a replay bypass of counterfactual coupling requirements.

## 0.4.0

- Unified finite hard/policy/delete/replace/composite queries, means, and paired effect contrasts.
- Preserved joint-output marginal surgery and conservative hidden-variable identification statuses.
- Added baseline-conditioned queries and complete-set DAG backdoor validation.
- Added compiled estimand/result persistence, normalized replacement validation, and nested-integration resource guards.

## 0.3.0

- Added immutable directed incidence and explicit causal model profiles.
- Added shared hard, joint-policy, delete, replace, composite, query, and contrast contracts.
- Added versioned data-only JSON persistence and immutable identification metadata.
- Distinguished declared assumptions from structural checks; corrected hidden-output query scope.
- Resolved the existing type-checking errors and duplicate-ADMG-arc handling.

## 0.2.0

- Refocused the package on domain-independent causal inference.
- Removed the experimental-screen application and its vendored data; the original code is preserved in Git history at `archive/pre-library-0.1`.
- Retained generic estimation, diagnostics, uncertainty, and exact inference.
- Added software, manufacturing, and policy examples with independently calculated answers, and executable README examples.
