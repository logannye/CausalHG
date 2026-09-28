# Contributing

Use Python 3.11 or newer. Install an editable checkout with optional dependencies:

```bash
python -m pip install -e ".[dev,numerical,interop]"
python -m pytest -q
python -m ruff check .
python -m pyright --pythonpath python
python -m build
```

Keep graph storage, causal assumptions, identification, statistical estimation, and
specified-model simulation separate. A new identifying route needs its model class,
assumptions, failure semantics, and an independent exact or analytic oracle. Numerical
agreement alone does not prove an identification theorem. Do not weaken refusal tests
to expand a capability claim. Include singular outputs and unsupported boundaries.

Preserve joint-output dependence, explicit axis order, stable identity, and independent
sampling units. Missingness, causal direction, cross-world coupling, and approximation
must be explicit. Optional dependencies must not be imported by the lightweight core.

Update the capability matrix, API documentation, runnable examples, changelog, and
migration notes with public behavior changes. Run the benchmark populations in
`benchmarks/` when changing estimators or numerical execution; report the seed,
environment, populations, bias/coverage, and failure counts alongside results.

The 1.x compatibility surface is the documented public API. Additive APIs can ship in
minor versions; corrections to invalid results may ship in patch versions with a
migration note. Other incompatible changes require a major version. Deprecations
receive a documented replacement and remain available through at least the next minor
release. Private underscore names and reference code in `minimal_model/` are excluded.
Serialization schema versions evolve independently of package versions; readers reject
unknown versions instead of guessing. Executable functions are never loaded from JSON.

Pull requests must pass the required CI checks. Release wheels and source distributions
are built from reviewed commits and tested from outside the repository. Publishing to a
package index is a separate release action, not a side effect of merging a change.
