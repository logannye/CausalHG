# Validate a complete adjustment set

`validate_adjustment_set(graph, treatments, outcomes, covariates)` applies the sufficient
DAG backdoor criterion to the **whole proposed set**. It accepts an ordinary DAG represented
by `ADMG` without bidirected edges, or a fully observed acyclic `MechanismGraph` with one
output per mechanism. It rejects treatment descendants in the original graph and checks
d-separation after removing outgoing treatment arrows. A `pass` certificate supports
adjustment under the declared causal graph and population positivity; data support and
graph correctness remain separate assumptions. A `fail` result means this sufficient
criterion failed, and does not claim the effect is unidentified or that every adjustment
method fails. Other model families return `unsupported`. This is separate from the older
individual-covariate hazard diagnostic `check_covariates`. See the backdoor criterion in
[Pearl, An Introduction to Causal Inference](https://pmc.ncbi.nlm.nih.gov/articles/PMC2836213/).

```python
from causal_hypergraphs import ADMG
from causal_hypergraphs.identification.adjustment import validate_adjustment_set

graph = ADMG(("C", "X", "Y"), (("C", "X"), ("C", "Y"), ("X", "Y")))
certificate = validate_adjustment_set(graph, treatments="X", outcomes="Y", covariates="C")
assert certificate.status == "pass"
```
