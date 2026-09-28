# Reproducible validation

These synthetic populations check known causal answers, uncertainty behavior,
refusal on unsupported data, and resource scaling. They establish evidence for the
declared model classes; they do not establish correctness for arbitrary data or
calibration of every nominal confidence interval.

## Commands

Install the package with its `numerical` extra. From a checkout, `PYTHONPATH=src`
can be used to select the local source. The initial runs used the scripts' default
100 datasets, 99 bootstrap refits per dataset, and seed `20260928`:

```sh
PYTHONPATH=src python benchmarks/continuous_validation.py
PYTHONPATH=src python benchmarks/discrete_validation.py
PYTHONPATH=src python benchmarks/performance.py
```

A follow-up investigation uses a disjoint seed block, 300 datasets, and 399
bootstrap refits without changing the estimators or thresholds:

```sh
PYTHONPATH=src python benchmarks/continuous_validation.py --replicates 300 --bootstrap 399 --seed 73192026
PYTHONPATH=src python benchmarks/discrete_validation.py --replicates 300 --bootstrap 399 --seed 73192026
```

The library's statistical runs use Python 3.14.6 and NumPy 2.4.1. Independent
reference runs use Python 3.14.3, NumPy 2.5.3, and SciPy 1.18.1. Performance has its
own recorded environment below. JSON reports preserve unrounded results and all
missing-interval and failed-refit counts. Missing intervals count as failures of
coverage, not as discarded datasets. Bias is the average estimated quantity minus
its known causal truth. Coverage Monte Carlo standard errors are
`sqrt(coverage * (1 - coverage) / number_of_datasets)`.

## Populations and gates fixed before the first run

The continuous and mixed populations have independent root variables `X` and `Z`
and a joint mechanism producing `Y` and `S`. The outcome is
`Y = 1 + slope * X + 0.7 * Z + residual`; the sibling shares part of its residual.
The query is `E[Y | do(X=1)] - E[Y | do(X=0)]`.

| Population | Data | Truth |
| --- | --- | --- |
| Gaussian | 400 independent rows, normal `X` and `Z`, slope 2 | 2 |
| Mixed | 400 independent rows, binary `X` with probability 0.45, normal `Z` | 2 |
| Clustered | 40 independent groups of 10 rows, group-level predictors and shared residual | 2 |
| Null effect | 400 independent Gaussian rows, slope 0 | 0 |
| Categorical | 400 independent binary `Z,T,Y` rows with observed confounding | 0.3 |

The clustered fit resamples independent groups, retaining all rows in each sampled
group. In the categorical population, `P(Z=1)=0.5`,
`P(T=1 | Z)=0.2+0.6Z`, and `P(Y=1 | T,Z)=0.1+0.3T+0.4Z`. Its query contrasts
`do(T=1)` with `do(T=0)` using the finite empirical backend.

Fixed continuous regression gates are absolute bias at most 0.15, coverage at
least 0.85, and no unavailable intervals. The categorical bias gate is 0.05, with
the same coverage floor and missing-interval gate. These intentionally broad smoke
gates detect regressions; **passing the 0.85 floor does not establish nominal 0.95
calibration**. They have not been retuned after viewing the results.

## Initial results and interval limitation

The [original continuous report](results/continuous-initial.json) and
[original categorical report](results/discrete-initial.json) use 100 datasets and
99 refits. All five populations had zero failed bootstrap refits and zero missing
intervals.

| Population | Bias | Nominal 95% coverage | Coverage MC standard error |
| --- | ---: | ---: | ---: |
| Gaussian | +0.0109 | 91% | 2.86 percentage points |
| Mixed | +0.0047 | 89% | 3.13 percentage points |
| Clustered | +0.0273 | 88% | 3.25 percentage points |
| Null effect | -0.0054 | 97% | 1.71 percentage points |
| Categorical | -0.0063 | 87% | 3.36 percentage points |

The lower coverage in several populations requires caution even though the smoke
gates passed. Percentile intervals are approximate, and 99 refits provide few
draws in each tail. More refits reduce bootstrap endpoint simulation error but do
not correct finite-sample bias, finite-cluster behavior, or model misspecification.
An independent reference comparison and a larger disjoint cohort investigate
these effects; the original results remain part of the evidence.

The independent [`interval_reference.py`](interval_reference.py) requires NumPy
and SciPy and does not call the library's fit, bootstrap, or inference code. It
reproduces the populations and computes ordinary least-squares Student t intervals.
For the clustered population it first averages each unit's responses; identical
within-unit predictors and the known homogeneous Gaussian unit errors make this
an exact reference for that specific population. The categorical reference uses a
saturated influence-function Wald interval and is itself asymptotic.

```sh
python -m pip install scipy
PYTHONPATH=src python benchmarks/interval_reference.py --seed 20260928 --replicates 100
PYTHONPATH=src python benchmarks/interval_reference.py --seed 73192026 --replicates 300
```

| Population | Initial independent reference | Disjoint 300-dataset reference |
| --- | ---: | ---: |
| Gaussian | 95% | 94.67% |
| Mixed | 92% | 94.33% |
| Clustered, unit-mean t | 94% | 94.00% |
| Null effect | 96% | 95.33% |
| Categorical, asymptotic Wald | 89% | 92.33% |

See the [initial reference report](results/interval-reference-initial.json) and
[confirmation reference report](results/interval-reference-confirmation.json),
which also give exact binomial intervals for simulation coverage. Reference and
library point-estimate biases agree on the same cohorts. The original percentile
cluster interval covers less often than the population-specific exact t reference;
choosing the correct resampling unit alone does not ensure finite-sample interval
calibration.

The [categorical confirmation](results/discrete-confirmation.json) with 399 refits
has bias +0.00456 and coverage 92.00% (Monte Carlo standard error 1.57 percentage
points), with zero failed refits or unavailable intervals. Its independent
asymptotic reference covers 92.33%. Both remain below nominal 95% in this cohort;
additional bootstrap draws do not eliminate that observed limitation.

The [continuous confirmation](results/continuous-confirmation.json) completes the
same disjoint 300-dataset cohort with 399 refits. It also has zero failed refits or
unavailable intervals:

| Population | Bias | Percentile coverage | Coverage MC standard error | Independent t reference |
| --- | ---: | ---: | ---: | ---: |
| Gaussian | -0.0032 | 93.33% | 1.44 percentage points | 94.67% |
| Mixed | +0.0145 | 94.00% | 1.37 percentage points | 94.33% |
| Clustered | -0.0077 | 91.33% | 1.62 percentage points | 94.00% |
| Null effect | -0.0014 | 95.33% | 1.22 percentage points | 95.33% |

Point estimation remains close to truth in these declared linear populations, and
the mixed and null intervals track their independent references. The clustered
percentile interval still undercovers relative to its exact population-specific
reference. The supported claim is an approximate bootstrap interval with explicit
resampling-unit and failure accounting, **not an established 95% calibration
envelope for small numbers of clusters or every finite categorical population**.
No estimator or threshold was tuned to these confirmation results. Changes made
during the run only tightened invalid sampling-unit rejection and the typed
backend interface; they do not affect any valid benchmark population.

## Negative controls and assumption sensitivity

The categorical sparse-support control sets `T=Z`, eliminating required treatment
strata within confounder strata. Estimation reports failed support and leaves the
contrast undefined. It does not substitute zero for the missing answer.

The continuous nonlinear control uses `X` uniform on `[-2,2]` and
`Y = 1 + 2X + coefficient * X**2 + noise`, with 600 rows. The query sets `X=1.8`.
The unchanged affine model increasingly misses the known answer:

| Quadratic coefficient | Known mean | Initial average bias |
| --- | ---: | ---: |
| 0 | 4.60 | +0.005 |
| 0.25 | 5.41 | -0.477 |
| 0.50 | 6.22 | -0.958 |
| 1 | 7.84 | -1.920 |

The coefficient-1 bias exceeds the predefined negative-control threshold of 1.3.
This is an explicit sensitivity check against a false affine assumption. It does
not repair misspecification or provide bounds for unmeasured confounding.

## Finite evaluator resource scaling

[`performance.py`](performance.py) supplies a separate finite `Model` implementation
for a symmetric Markov chain, whose stationary probability is analytically
`1 / cardinality`. The [saved report](results/performance.json) records Python
3.14.3 on macOS 26.2, arm64. Timings include graph construction, query compilation,
planning, and factor elimination while allocation tracing is enabled.

| Nodes | Cardinality | Induced width | Largest factor entries | Time | Peak traced allocation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 2 | 1 | 4 | 1.7 ms | 32 KB |
| 40 | 2 | 1 | 4 | 8.6 ms | 71 KB |
| 100 | 2 | 1 | 4 | 36.6 ms | 150 KB |
| 40 | 4 | 1 | 16 | 11.8 ms | 66 KB |

All answers match the independent stationary-law oracle; the ten-node case also
matches enumeration. A separate wide-factor case is refused before evaluation
when its factor exceeds the configured entry budget. These are width-one
measurements on one machine, not a guarantee of scalability for dense graphs.
Traced allocations are not total process memory.
