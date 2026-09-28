# Public API reference

The following names form the documented 1.x surface. Inspect their Python signatures
and docstrings for argument types and detailed error conditions. Names beginning with
`_`, internal reduction helpers, and `minimal_model/` are not stable extension points.

| Operation | API |
|---|---|
| Structure | `DirectedHyperedge(name, inputs=(), outputs=(), metadata=None)`; `DirectedHypergraph(nodes, edges, metadata=None)` |
| Causal interpretation | `CausalModelSpec(graph, profile="independent-mechanisms", ...)`; `.to_mechanism_graph()`; `MechanismGraph(variables, mechanisms, ...)` |
| Queries | `CausalQuery(outcomes, intervention, kind="distribution", given=())`; `EffectContrast(left, right)` |
| Operations | `HardIntervention(assignments)`; `JointPolicy(variables, probabilities, name="policy")`; `Delete(target, policy)`; `Replace(target, replacement)`; `Composite(operations)` |
| Identification | `compile_query(graph, query) -> CompiledQuery`; inspect `.result`, `.query`, `.variables`, `.policies` |
| Finite execution | `evaluate_query(compiled, model, ...)`; `estimate_query(compiled, dataset, ...)` |
| Finite data | `Dataset.from_records(records, domains=None, unit=None, measures=())`; `Dataset.from_counts(counts, variables, domains=None)` |
| Continuous fit | `fit_linear_gaussian(graph, records, unit=None, discrete=())`; fitted `.estimate(query, ...)`; `estimate_continuous(fitted, query, ...)` |
| Executable model | `HypergraphSCM(graph, mechanisms, exogenous=None)`; `.sample`, `.sample_with_noise`, `.evaluate_with_noise`, `.abduct`, `.counterfactual` |
| Executable bindings | `StructuralMechanism`; `FiniteKernel`; `LinearGaussianMechanism`; `NoiseRecord` |
| Independent adjustment check | `validate_adjustment_set(graph, treatments, outcomes, covariates=()) -> AdjustmentResult` |
| Persistence | `dumps(value, indent=None)` / `loads(text)`; `to_dict(value)` / `from_dict(document)`; `SCHEMA_VERSION` |

`Identified`, `Unknown`, and `Unidentified` retain status strings, assumptions and
trace/witness information. `Assumption.state` distinguishes declarations and checked
conditions. `CompiledQuery` retains the whole question and bound policies so that
execution cannot silently lose the intervention specification.

`Estimate.values` maps requested-axis tuples to numeric results; a scalar mean has
key `()`. `.variables` specifies axis order. `.support`, `.policy`, `.interval`,
`.replicate_failures`, `.plan`, and `.summary()` explain empirical/numerical behavior.
`ContinuousEstimate.value` is a scalar mean/contrast; interval/standard error concern
sampling uncertainty, whereas `model_variance` concerns the outcome distribution.

Compatibility APIs remain available: `identify`, `DeleteMechanism`, `ReplaceMechanism`,
`identify_expectation`, `ADMG`, `identify_effect`, expression nodes and finite semantics,
`d_separated`, determination rules, `check_covariates`, and `plan_elimination`.
Legacy policy tables have different missing-entry semantics; see [migration](migration.md).
Placeholders such as `T7ReductionPlaceholder` remain importable for migration but are
not the recommended public entry point. Use `compile_query` and inspect its result.

Named error families include `UnsupportedModelError`, `SerializationError`,
`estimation.DatasetError`, `NotIdentified`, `UnsupportedEstimand`,
`semantics.UndefinedEstimand`, `MissingKernel`, `IntractableQuery`, and the continuous
backend's `ContinuousDataError`, `RankDeficientFit`, `UnsupportedContinuousQuery`.
An exception or resource refusal is not a nonidentification witness.

See [backend protocols](backends.md), [interoperability](interoperability.md),
[simulation](simulation.md), and [continuous inference](continuous.md) for extension
contracts. The [compatibility policy](../CONTRIBUTING.md) governs changes after 1.0.
