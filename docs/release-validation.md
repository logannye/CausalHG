# 1.0 release validation

The release sequence begins at `db56c051c24be66304265973d3fa29bb187cf211` and preserves
that application's source at `archive/pre-library-0.1`. Implementation commits close
0.2, 0.3, 0.4, 0.5, 0.6, and 0.9 in order before the 1.0 version is finalized.

## Evidence

The baseline had 352 passing tests and one future-work xfail; Ruff passed and Pyright
reported 26 errors. The stale xfail has been replaced by a test of the supported refusal.
The expanded suite covers independent exact SCM laws, ordinary ADMG identification,
separation, hidden boundaries, support/alias failure cases, joint-output interventions,
query composition, serialization, data contracts, numerical fitting, replay/counterfactual
coupling, adapters, and execution budgets. Type-checking errors have been resolved.

The final local run on Python 3.14.3/macOS passed all 582 tests in 95.02 seconds,
including optional numerical and interoperability cases. Ruff and Pyright passed,
and the diff whitespace check was clean. GitHub checks report each platform's counts,
including skips where optional graph adapters are not installed.

The 1.0 wheel built from its source distribution installed in clean Python 3.11 and
3.14 environments, including a 3.14 environment with no optional dependencies. It
imported every module without importing NumPy and passed
an end-to-end identify/estimate/serialize check plus all three installed domain examples
from outside the checkout. The release workflow repeats installed-wheel and source-
distribution checks and uploads both artifacts. Numerical and interoperability extras
have their own validation coverage.

The CI matrix checks Python 3.11, 3.12, 3.13 and 3.14 on Linux, plus 3.14 on macOS and
Windows. These are checks to inspect on the release commit, not a prediction that a
pending workflow will succeed. Release completion requires their successful result.

## Methodology and statistics

[Independent agent cross-review](methodology-review.md) documents the supported
claims and concrete fixes it produced. It is not external academic peer review or a
formal theorem certification. Existing randomized conformance oracles were retained;
new simulation and inference tests use independent finite/analytic answers.

[Benchmark populations and saved results](../benchmarks/README.md) publish bias,
coverage, failed-fit/interval counts, sparse-support refusals, nonlinear and null-effect
controls, and resource measurements. Original unfavorable interval results are retained
alongside the larger disjoint confirmation and independent statistical references.
Nominal bootstrap intervals have finite-sample limitations; the release does not claim
universal 95% calibration. No empirical benchmark verifies real-world causal assumptions.

## Remaining scope boundaries

The library's stable capability contract is in [capabilities.md](capabilities.md).
General cyclic identification, correlated mechanism noise, multiple-producer inference,
causal discovery, unrestricted counterfactual identification and hyper-hedge completeness
remain research extensions. Continuous estimation is explicitly affine with separately
declared Gaussian assumptions for baseline conditioning. The package is useful within
these profiles and refuses unsupported combinations.

GitHub release artifacts are distinct from publication to PyPI. No package-index
publication or external academic endorsement is part of this release.
