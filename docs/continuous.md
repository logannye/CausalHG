# Continuous and mixed numeric estimation

The optional numerical backend fits an affine model to each joint-output mechanism
and evaluates causal means analytically. It is useful when conditioning variables
are continuous and a linear additive model is scientifically justified. Install the
`numerical` extra from the repository checkout to use it:

```sh
python -m pip install '.[numerical]'
```

This backend supports fully observed, acyclic `MechanismGraph` models with
independent mechanisms. Every exogenous variable is an independent root. Correlated
root variables must instead be outputs of an explicit joint mechanism; fitting
separate roots does not estimate their cross-covariance. The graph and its causal
assumptions must be supplied by the caller. A good statistical fit does not verify
those assumptions or establish identification.

## Example

```python
import numpy as np

from causal_hypergraphs.estimation.continuous import fit_linear_gaussian
from causal_hypergraphs.graph import MechanismGraph
from causal_hypergraphs.queries import CausalQuery, EffectContrast, HardIntervention

rng = np.random.default_rng(12)
x = rng.normal(size=600)
y = 1 + 2 * x + rng.normal(size=600)
rows = [{"X": a, "Y": b} for a, b in zip(x, y, strict=True)]
graph = MechanismGraph(
    ("X", "Y"), {"response": {"inputs": "X", "outputs": "Y"}}
)
query = EffectContrast(
    CausalQuery(("Y",), HardIntervention({"X": 1}), "expectation"),
    CausalQuery(("Y",), HardIntervention({"X": 0}), "expectation"),
)
fitted = fit_linear_gaussian(graph, rows)
result = fitted.estimate(query, bootstrap=399, seed=42)
assert abs(result.value - 2) < 0.15
print(result.summary())
```

`estimate_continuous(fitted, query, ...)` provides the same evaluation as
`fitted.estimate(query, ...)`. `ContinuousBackend` and `ContinuousCapabilities`
describe the scoped backend contract. Neither is a general evaluator of arbitrary
identification expressions. `estimate_continuous` also accepts a custom backend
implementing that protocol. The backend owns validation and refusal for unsupported
queries or options; the wrapper does not expand its declared capabilities.

## Model and operations

For mechanism inputs `X` and joint outputs `Y`, the fitted model is
`Y = intercept + coefficients @ X + residual`. Multivariate least squares fits
the conditional means together and retains the full residual covariance matrix.
Residual covariance uses the residual degrees of freedom. Independent root means
and variances use the corresponding sample moments. Prediction-design rank
deficiency is refused; singular joint output covariance is supported.

Unconditional means and second moments require affine conditional means and the
stated noise moments. Gaussianity is additionally needed for the supported
baseline-conditioning formula. Gaussianity, linearity, constant residual
covariance, causal sufficiency, independent roots, and independence between
mechanisms are assumptions; this backend does not test them.

| Query or operation | Behavior |
| --- | --- |
| Single-outcome expectation | Analytic mean and fitted outcome variance |
| Difference of expectations | Paired mean contrast; no cross-world variance claim |
| `HardIntervention` | Overwrites requested coordinates |
| `JointPolicy` | Numeric finite policy; retains dependence between policy coordinates |
| `Delete` | Uses the declared joint fallback policy |
| `Replace` | Requires a supplied affine joint-output kernel with the same boundary |
| `Composite` | Uses the shared frontend's compatibility rules |
| Baseline conditioning | Joint Gaussian formula with nonsingular baseline covariance |
| Distribution query | Refused |

Overwriting one output of a joint mechanism retains the other outputs' marginal
residual law. It does not condition that residual law on the overwritten value.
An explicit replacement can change output equalities; observations used to fit the
original mechanism must satisfy its declared equalities.

For `Replace("mechanism_name", "new_kernel")`, pass
`replacements={"new_kernel": kernel}`. A `LinearGaussianKernel` or the simulation
runtime's compatible `LinearGaussianMechanism` supplies the kernel. Coefficient
rows follow `outputs`, columns follow `inputs`, and covariance follows `outputs`.
Axis names are ordered sequences; unordered sets and mappings are rejected.
Replacement parameters are treated as fixed during bootstrap resampling.

## Baseline and mixed numeric data

Declare baseline variables in `CausalQuery.given` and bind their values with
`fitted.estimate(query, given={...})`. The bindings must match exactly. The shared
query compiler rejects post-intervention conditioning outside its baseline
contract. Conditional inference uses the fitted joint Gaussian mean and covariance
and refuses singular or numerically ill-conditioned baseline covariance; it does
not select a pseudoinverse extension to unsupported conditioning values.

Numeric discrete variables are supported as exogenous predictors when explicitly
declared, for example `fit_linear_gaussian(graph, rows, discrete=("T",))` for a
binary root treatment. Their empirical moments enter unconditional means. This
assumes a linear conditional mean, including any effect homogeneity implied by
that model. An intervention at an unobserved discrete category is refused.
Endogenous discrete variables require another backend. This backend does not
infer variable types from a small set of numeric values.

Baseline conditioning is refused when discrete roots are declared or a
nondegenerate finite intervention policy is used. Such models can yield mixtures;
the Gaussian conditioning formula is not silently applied to them. Unconditional
policy means and variances remain available through exact moment propagation.

## Data, uncertainty, and diagnostics

Every graph variable must be present and finite in every record. Missing values,
nonnumeric values, hidden variables, and cycles are refused. Predictor designs
must identify the declared coefficients, with enough rows left to estimate
residual covariance. No imputation, automatic regularization, or causal discovery
is performed.

`unit="subject_id"` declares independent sampling units for cluster resampling.
All rows in each sampled unit move together. Units may contain different numbers
of rows; the fitted point estimand remains row weighted. A contrast uses the same
resampled fitted model in both arms. The caller must choose a defensible independent
unit; the backend does not infer dependence from the data.

With `bootstrap > 1`, uncertainty is a percentile bootstrap interval at `level`
(default `0.95`). Its standard error describes parameter-sampling uncertainty.
Any failed bootstrap refit is counted and withholds the interval and standard
error; failures are not discarded to calculate a success-only interval.
`successful_replicates` and `bootstrap_failures` make that accounting explicit.

`model_variance` is outcome variation under the fitted intervention model, not
the standard error of its estimated mean. It is absent for contrasts because no
cross-world noise coupling is specified. Analytic integration has
`monte_carlo_error=0`; floating-point numerical error is unquantified and recorded
as `numerical_error=None`. Bootstrap endpoint simulation error and model
misspecification are not incorporated in the interval.

Marginal range diagnostics flag extrapolation outside observed coordinate ranges.
Being inside those ranges does not establish joint overlap. Continuous conditions
are not claimed to have positive point mass. Result assumptions replace finite
positivity labels with the relevant continuous overlap and model assumptions.

The validation populations, original coverage results, independent follow-up
checks, and deliberately misspecified examples are documented in
[the benchmark report](../benchmarks/README.md). Bootstrap intervals are
approximate; they do not have a finite-sample or universal calibration guarantee.
