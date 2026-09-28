# Model and query semantics

CausalHG distinguishes structural data, causal assumptions, identification, estimation,
and simulation. A hyperedge becomes a causal mechanism only after input/output roles
and a supported causal interpretation have been declared.

## Supported model families

`DirectedHypergraph` stores directed incidence, including cycles, overlapping outputs,
and separately named parallel edges. It does not assert a causal factorization.
`CausalModelSpec` converts a supported independent-mechanism profile to `MechanismGraph`.
That profile requires a single producer per variable and independent exogenous noise
between mechanisms (including exogenous variables). Joint outputs of one mechanism
may remain dependent, share noise, or satisfy deterministic equalities.

Ordinary semi-Markovian `ADMG` inference is a separate profile. An ADMG with arbitrary
bidirected structure need not have the mechanism profile's factorization. Successful
identification on a projection transports an appropriate identifying formula; failure
on that projection does not necessarily prove failure in a narrower model class.

Native identifiers are nonempty strings. External adapters must preserve an explicit
identity mapping or reject unsupported identifiers. Integer `1` and string `"1"`
must never be silently merged. Incidence membership and numerical kernel axis order
are distinct: a joint policy's tuple entries follow its declared variable order.

## Intervention contracts

`HardIntervention` replaces only the named variable coordinates by fixed values. If a
mechanism jointly produces B and C, intervening on B preserves C's original structural
function and noise. In a fully observed factorization, this uses the marginal factor
for the retained outputs, not their conditional distribution given the forced value.

`JointPolicy` supplies an unconditional finite probability mass function, independent
of the original exogenous noise. All positive-mass support must be listed; omitted
tuples have zero mass. Coordinates are ordered. It is not a continuous density and
must not be replaced by a product of its marginals.

`Delete` removes a mechanism and installs a joint policy over all its outputs.
Deletion does not implicitly mean setting outputs to zero. `Replace` preserves the
target's incidence and names a separately supplied replacement kernel or executable
mechanism. Both the original and replacement boundary must match.

`Composite` combines compatible operations. Operations that touch overlapping outputs
are rejected rather than assigned an arbitrary precedence. An empty composite is an
observational query. Conditional or history-dependent policies require additional
theory and are outside the initial finite policy contract.

`CausalQuery` separates an intervention from a requested distribution or expectation.
Baseline-conditioned queries are supported only through an identified joint law and
normalization; conditioning variables cannot be intervened variables or descendants
of the intervention in the original graph. An effect contrast compares the same
quantity under two specified interventions. A regression conditional alone is not
an identified causal effect.

## Results and limitations

`Identified` contains an estimand, theorem reference, assumptions, and derivation.
`Unknown` means the implementation has not identified this query. `Unidentified`
requires a witness applicable to the query and its model family. Human-readable
reasons are accompanied by stable status/reason codes. A missing estimator, inadequate
empirical support, or a resource budget limit is not proof of nonidentification.

Assumption states distinguish declarations, structural checks, empirical assessments,
and unresolved conditions. The default state is declared. Derivation records are
inspectable arguments, not proofs checked by a formal theorem prover. No software
inspection of incidence can establish independence of the real world's noise terms.

`check_covariates` is a diagnostic. Its `admissible` flag means that its implemented
hazard checks found no obstruction; it is not a complete adjustment-set certificate.
Finite empirical support does not prove population positivity. `PolicySupport` uses
an explicitly reported heuristic threshold and a policy-weighted row-count diagnostic;
that quantity is not the universal effective sample size of every estimator.

## Execution and counterfactuals

Exact finite evaluation and variable elimination evaluate a mathematical expression.
Fitted estimation additionally depends on statistical assumptions. Numerical
approximation must be selected explicitly and report its error/cost separately from
sampling uncertainty. Continuous models require a supported numerical backend; a
finite mass-function evaluator cannot treat density values as probability masses.

SCM simulation assumes supplied structural mechanisms and noise distributions.
Counterfactuals preserve the relevant noise across worlds and need a supplied noise
record or a valid abduction procedure. A stochastic kernel's marginal law does not
determine a cross-world coupling. Counterfactual simulation from a specified model
does not establish counterfactual identification from observational data.

## Persistence

`causal_hypergraphs.io.dumps/loads` use a versioned, deterministic, data-only JSON
envelope. Supported graphs, queries, expressions, and identification results preserve
tuple axes, joint policies, aliases, assumptions, and derivations. Unknown schemas,
unregistered types, executable functions, and nonfinite numeric values are rejected.
Executable models bind functions separately; deserialization never executes a
function supplied in a payload.
