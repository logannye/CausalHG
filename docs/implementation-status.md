# Implementation status

Baseline: db56c051c24be66304265973d3fa29bb187cf211. The active implementation is on codex/general-purpose-1.0.

Baseline validation (Python 3.14.3, macOS): 352 passed, 1 xfailed in 39.30 seconds; Ruff passed. Pyright reported 26 pre-existing errors. Editable installation succeeded after rebuilding the installation.

| Milestone | Status | Evidence |
|---|---|---|
| 0.2 | Complete | Demo removed; 3 exact synthetic examples and 6 regression tests; README snippets execute; full worktree 359 passed, 1 xfailed (includes 7 persistence tests) |
| 0.3 | In progress | Structural/model contracts and persistence |
| 0.4 | Pending | Unified intervention/query frontend |
| 0.5 | Pending | Executable SCMs and scoped counterfactuals |
| 0.6 | Pending | Statistical backends and continuous covariates |
| 0.9 | Pending | Adapters, packaging, typing, release validation |
| 1.0 | Pending | Independent review, stable contracts, GitHub checks |
