# Implementation status

Baseline: db56c051c24be66304265973d3fa29bb187cf211. The active implementation is on codex/general-purpose-1.0.

Baseline validation (Python 3.14.3, macOS): 352 passed, 1 xfailed in 39.30 seconds; Ruff passed. Pyright reported 26 pre-existing errors. Editable installation succeeded after rebuilding the installation.

| Milestone | Status | Evidence |
|---|---|---|
| 0.2 | Complete | Demo removed; 3 exact synthetic examples and 6 regression tests; README snippets execute; full worktree 359 passed, 1 xfailed (includes 7 persistence tests) |
| 0.3 | Complete | Immutable graph/model/query contracts, versioned JSON round trips, scoped hidden-output regression tests, clean Pyright |
| 0.4 | Complete | 19 unified inference tests, 14 adjustment tests, persistence integration; independent semantic review corrected nested resource-budget handling |
| 0.5 | Complete | 18 runtime tests including finite/Gaussian analytic checks, singular outputs, replay and coupling refusals; independent implementation review |
| 0.6 | Pending | Statistical backends and continuous covariates |
| 0.9 | Pending | Adapters, packaging, typing, release validation |
| 1.0 | Pending | Independent review, stable contracts, GitHub checks |
