# Executable causal models

`HypergraphSCM(graph, mechanisms, exogenous=...)` binds each acyclic graph mechanism
to one joint model with the same input/output incidence. Each unproduced variable
needs an independent root sampler accepting `random.Random`. Correlated roots should
instead be outputs of a joint root mechanism. User callables must use the supplied
RNG/noise rather than hidden global state.

```python
from causal_hypergraphs import FiniteKernel, HardIntervention, HypergraphSCM, MechanismGraph

graph = MechanismGraph({"X", "Y"}, {"pair": {"inputs": (), "outputs": ("X", "Y")}})
kernel = FiniteKernel((), ("X", "Y"), {(): {(0, 0): .75, (1, 1): .25}})
model = HypergraphSCM(graph, {"pair": kernel.as_structural()})
factual, noise = model.sample_with_noise(seed=42)
other = model.counterfactual(HardIntervention({"X": 1}), noise=noise)
assert other["X"] == 1 and other["Y"] == factual["Y"]
assert model.evaluate_with_noise(noise) == factual
```

`FiniteKernel` accepts ordered input/output axes and a mapping from each input tuple
to a normalized joint probability table. For nonempty input axes, declare complete
`input_domains`. Missing output tuples have zero mass. `.as_structural()` explicitly
chooses inverse-CDF coupling with the stored outcome order. This is one model choice;
it is not identified by the kernel's probabilities. Kernel-only models sample but
refuse counterfactuals.

`StructuralMechanism(inputs, outputs, function, noise_sampler, abduct=None)` provides
custom functions. The function consumes an input mapping and its shared noise, then
returns exactly the output mapping. Optional deterministic abduction must be a valid
noise inverse; replay verifies consistency, while uniqueness remains a model assumption.
Noninvertible posterior abduction is outside the stable interface.

`LinearGaussianMechanism` accepts output-by-input coefficients, an intercept vector,
and a positive-semidefinite joint residual covariance. NumPy is optional and required
only for this mechanism. Singular covariances are supported. Additive residuals are
abducted from complete observations, with consistency checked against their support.

`sample(n, seed=..., intervention=..., replacements=...)` returns independent rows.
`sample_with_noise` also records the draws and intervention; `evaluate_with_noise`
replays without consuming randomness. Replacement replay requires the original bound
replacement model. Changing worlds requires the same coupling checks as counterfactuals.
Noise records with data-only payloads support versioned JSON persistence. Custom noise
objects must be treated as immutable by callers and may not be serializable.

The stable counterfactual operation accepts hard interventions and an observational
noise record or complete factual observation with deterministic abduction. Policy,
deletion, and replacement counterfactuals require additional cross-world coupling and
are refused. Simulation from a supplied model does not establish identification from
observational data. The older `minimal_model/` implementation is reference code only.
