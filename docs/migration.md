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

## 1.0: graph, query, simulation and estimation contracts

Native identifiers now require nonempty strings. Earlier implicit string coercion is
removed to prevent collisions such as integer `1` and string `"1"`. Use an explicit
adapter `id_codec` and inspect its reverse mapping. Graph dictionaries and result
aliases are immutable snapshots; reconstruct an object to change it. Canonical mapping
iteration is sorted and must not be used as construction-order traversal.

Use `DirectedHypergraph` for general incidence and `CausalModelSpec` to declare its
causal meaning. `MechanismGraph` retains its single-producer profile; removing this
validation would invalidate existing inference algorithms. Arbitrary stored feedback
and multiple producers do not imply supported causal inference.

Both direct `MechanismGraph` construction and causal-profile conversion require
mechanism IDs distinct from all variable IDs. Rename ambiguous legacy mechanism
IDs explicitly; the old bipartite representation could otherwise merge two different
vertices and return incorrect separation results. `DirectedHypergraph` still preserves
separate node and edge namespaces for general storage.

Prefer `CausalQuery`/`compile_query` with `HardIntervention`, `JointPolicy`, `Delete`,
`Replace`, or `Composite`. The original `identify`/`DeleteMechanism`/`ReplaceMechanism`
interfaces remain compatible. New joint policies have explicit ordered axes and sparse
zero semantics. Legacy fallback/replacement table APIs still require explicit queried
entries. A `JointPolicy` is an unconditional finite mass function, not a density.

An individual-output hard intervention preserves its siblings' structural functions
and original shared noise. Do not use the old reference simulator's `do_node` semantics
as the new runtime contract. `HypergraphSCM` is the shipped runtime; `minimal_model/`
remains independent unshipped reference code. Kernel-only simulation does not license
cross-world counterfactuals without a declared coupling.

`Dataset` now rejects missing/nonfinite data, inconsistent row schemas, invalid domains,
and noninteger counts. Missingness must be handled explicitly before construction; no
automatic dropping or imputation occurs. For continuous conditioning, install the
`numerical` extra and use `fit_linear_gaussian` within its declared affine scope.

Unknown/refused algorithms remain distinct from nonidentification witnesses. A projected
ADMG hedge need not refute a supplied policy mixture or a restricted mechanism model.
`Assumption.state` and reason codes supplement existing human-readable explanations.
The DAG whole-set `validate_adjustment_set` is distinct from individual covariate
hazard diagnostics. Old placeholder imports remain available during migration, but
callers should use the documented query and result types rather than placeholders.

Schema version 1 stores data-only supported objects. Unknown schemas and types are
refused. Functions and fitted executable model bindings are deliberately external;
keep their source/version with your analysis. See the 1.x compatibility/deprecation
policy in [CONTRIBUTING.md](../CONTRIBUTING.md).
