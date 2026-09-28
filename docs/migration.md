# Migration notes

## 0.2: domain-neutral onboarding

The package is being developed as a general-purpose causal inference library for
explicitly interpreted hypergraphs. The distribution remains `causal-hypergraphs`
and the import namespace remains `causal_hypergraphs`.

This milestone removes the screen-specific application from the active repository:
`demo/preflight.py`, `demo/build_cache.py`, vendored `demo/data/`, and the associated
demo documentation and smoke tests. The prior application remains recoverable from
Git history. Callers that ran those scripts need the earlier revision; there is no
replacement experiment-design command in the library.

The README now introduces identification and estimation with synthetic joint-output
data. [Current capabilities](capabilities.md) distinguish implemented functionality
from [future work](../CAUSALHG_DEVELOPMENT_ROADMAP.md).

The cleanup does not remove the generic functions used by the application:

- `Dataset`, `estimate`, unit-bootstrap intervals, and empirical support diagnostics.
- Policy concentration diagnostics and their configurable threshold.
- `check_covariates`, graphical separation, and latent projection.
- Symbolic estimands, structured identification results, and variable elimination.

Existing mechanism-query imports and call signatures are unchanged by this milestone.
Theory documents remain at their existing root paths. `minimal_model/` remains
reference code and is not promoted into the package's public API.

Some diagnostic labels can otherwise be read too broadly: empirical support does
not prove population positivity; the policy count threshold is a heuristic; and a
covariate marked “admissible” is not a complete adjustment certificate. This milestone
documents those limits without silently changing their numerical behavior.

Later milestones will document their own API changes and compatibility paths here.
The roadmap is not a guarantee that proposed interfaces already exist.
