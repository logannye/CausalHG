# Methodology review for the general-purpose release

Reviewed September 28, 2026. This records cross-review by separate AI coding agents:
reviewers inspected inference, simulation, and numerical estimation code they did not
initially author, using separate exact and analytic numerical oracles. It is not external
academic peer review, formal proof verification, or evidence that a user's causal model
describes their data.
Release/build/CI evidence is recorded separately in [implementation status](implementation-status.md).

## Claims reviewed

The reviewed public scope is an acyclic independent-mechanism model with one producer
per variable, joint output blocks, and mutually independent root variables. Correlated
roots belong in a joint root mechanism. Ordinary semi-Markovian ADMG identification is
a separate model family. Unrestricted hypergraph incidence does not itself establish
either interpretation.

For fully observed mechanism models, a hard intervention on some coordinates of a
joint output preserves the original equations and shared noise of retained coordinates.
Its identifying factor is the **marginal** joint kernel of retained outputs given the
original parents, not the conditional kernel given the forced output values. Deletion
installs one unconditional joint policy. Replacement preserves the input/output boundary.
The compiler and executable simulator were checked against these same semantics using
separate numerical and structural oracles.

Baseline-conditioned queries require baseline variables unaffected by the intervention
in the original graph. The finite compiler identifies a joint post-intervention law
before normalization. The continuous backend additionally assumes joint Gaussian blocks
and refuses singular baseline conditioning rather than using a pseudoinverse to invent
unsupported values. Joint finite policy mixtures and declared discrete roots are excluded
from its Gaussian conditional-mean path.

The ordinary ADMG backend may issue a nonidentification witness within its model class.
A failure after projecting a restricted mechanism model, or before averaging against a
supplied policy, is not automatically a witness against the requested effect. Those
unresolved cases remain `Unknown`. Failure to identify individual contrast arms likewise
does not prove that their difference is unidentified. The legacy symbolic hidden-output
relabelling argument explicitly excludes policies invariant under the proposed relabelling.

`validate_adjustment_set` checks the complete proposed set using the sufficient DAG
backdoor criterion. A failed criterion is not a complete invalid-adjustment or
nonidentification theorem. Unsupported mixed/hidden/joint-output model families receive
an explicit unsupported result.

## Findings corrected during cross-review

| Finding | Corrected contract and regression evidence |
|---|---|
| A hidden output affecting an unrelated measurement could refute an unaffected requested outcome. | Hidden-output reach is scoped to requested outcomes and traverses intermediate measured variables. See `tests/test_hidden_query_scope.py`. |
| Means and conditional ratios could hide large nuisance sums from the elimination planner. | Opaque integrations receive a separate finite-work budget. Flat sum-product queries retain efficient elimination. See `tests/test_inference.py`. |
| Changed-intervention replay could bypass the simulator's counterfactual restrictions. | Replay across intervention worlds uses the same coupling checks as counterfactual evaluation. See `tests/test_simulation.py`. |
| A structural function mutating its supplied noise could change future factual replay. | Evaluation copies saved custom noise before calling a user function. Complete replay consumes no new randomness. |
| Executable bindings could contradict declared original output equalities. | Active original bindings are checked before coordinate surgery; replacement functions may deliberately change the original equalities. |
| A finite positive-mass baseline assumption could leak into continuous results. | Continuous results remove finite-mass certificates and declare continuous overlap and Gaussian conditioning separately. See `tests/test_continuous.py`. |
| Continuous conditional variances were clipped even if materially negative. | Only rounding-scale negative values are clipped; material failures raise an explicit numerical error. |
| Numerical axis declarations accepted unordered containers. | Joint policies and numerical kernels require explicit axis order. |
| A variable and mechanism sharing the name `X` collapsed into one bipartite node, giving a false dependence. | Direct `MechanismGraph` construction now rejects overlapping variable/mechanism IDs. `test_direct_mechanism_construction_cannot_merge_a_root_with_a_mechanism` in `tests/test_graph_contracts.py` covers variables `X,Y,Z` with mechanism `X: Z -> Y`: root `X` must be separated from `Y`, exactly as when the mechanism is named `make_y`. Generic incidence retains separate namespaces. |
| Importing an external XGI graph could fail while looking up an absent CausalHG metadata marker. | External graphs follow the explicit-role conversion path without requiring native metadata. See `test_external_directed_xgi_has_no_required_native_marker` in `tests/test_adapters.py`. |
| Extra attributes added to exported native adapter objects could be silently lost on reimport. | Native NetworkX and XGI imports reject attributes they cannot preserve. See `test_native_graph_imports_refuse_attributes_they_cannot_preserve` in `tests/test_adapters.py`. |

## Evidence and remaining limits

The reviewed regression subsets passed across unified inference, simulation, continuous
estimation, adjustment, and adapter contracts. These include shared-noise surgery,
nonfactorizing policies, front-door alias preservation, exact replay, singular Gaussian
noise, analytic Gaussian moments, whole-set collider checks, paired contrast bootstrap,
cluster resampling, rank deficiency, and explicit unsupported cases. Final release test
totals are recorded separately from these focused checks.

The [categorical validation script](../benchmarks/discrete_validation.py) and
[continuous validation script](../benchmarks/continuous_validation.py) declare their
populations, seeds, sample sizes, bootstrap counts, and acceptance thresholds before
execution. They cover categorical confounding, continuous and mixed predictors,
clustered sampling, a null effect, sparse support, and nonlinear misspecification.
Coverage summaries count all generated datasets and report Monte Carlo uncertainty.
The initial 100-dataset, 99-refit nominal 95% intervals covered 91% of Gaussian effects,
89% of mixed effects, 88% of clustered effects, 97% of null effects, and 87% of categorical
effects. Passing the predefined 85% regression floor does not establish 95% calibration.
These unfavorable results are retained, with a disjoint 300-dataset, 399-refit follow-up
and an [independent reference implementation](../benchmarks/interval_reference.py) that
does not call the library's fit, bootstrap, or inference code.

The follow-up nominal 95% intervals cover 93.33% of Gaussian effects, 94.00% of mixed
effects, 91.33% of clustered effects, 95.33% of null effects, and 92.00% of categorical
effects; all five have zero failed refits or unavailable intervals. The categorical
follow-up's independent asymptotic influence-function Wald reference covers 92.33% on
the same cohort. Neither categorical result establishes nominal calibration. For the
clustered cohorts, percentile coverage is 88% initially and 91.33% in the follow-up,
compared with 94% in both cohorts for an exact, population-specific Student t reference
computed from independent unit means. The persistent cluster and categorical
undercoverage is a release limitation. Correct resampling units and additional bootstrap
draws do not by themselves guarantee finite-sample coverage. The
[benchmark report](../benchmarks/README.md) records all runs, assumptions, environments,
and Monte Carlo uncertainty. The nonlinear control is expected to expose bias; passing
it does not mean that the linear estimator has repaired the misspecification.

Continuous estimation is a fitted affine joint-kernel backend with analytic moment
propagation. It is not a generic continuous evaluator for arbitrary identifying
expressions, a nonparametric estimator, or a universal density model. Its bootstrap
resamples declared independent units, uses the same refit for both contrast arms, and
withholds an interval when a refit fails. Outcome variation, parameter uncertainty,
Monte Carlo error, and numerical error without a certified bound are reported distinctly.

Counterfactuals require a supplied factual noise record or valid deterministic abduction
and retain the relevant noise across worlds. A finite probability kernel alone does not
specify cross-world coupling; choosing `FiniteKernel.as_structural()` explicitly adds
an ordered inverse-CDF model assumption. Stable counterfactual support is restricted to
hard interventions. Posterior abduction and stochastic/replacement cross-world coupling
are outside this release's supported contract.

Incidence checks cannot establish that the graph is true or that different mechanisms'
noise is independent. Finite empirical support does not establish population positivity.
The open hyper-hedge completeness question, arbitrary feedback, unrestricted mediation,
and causal discovery remain outside the release claims. Core identification references
include [Shpitser and Pearl's joint-ID result](https://ftp.cs.ucla.edu/pub/stat_ser/r327.pdf),
[conditional identification](https://arxiv.org/abs/1206.6876), and
[Pearl's explanation of the backdoor criterion](https://pmc.ncbi.nlm.nih.gov/articles/PMC2836213/).
