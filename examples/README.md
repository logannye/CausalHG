# Synthetic examples

Install the library (`python -m pip install -e .` from the repository root), then run:

```sh
python examples/software_pipeline.py
python examples/manufacturing.py
python examples/policy.py
```

The scripts import the installed `causal_hypergraphs` package. They need no downloads,
external datasets, or third-party runtime dependencies. Their small local helper only
constructs exact synthetic frequencies and runs the public identification/estimation APIs.

| Example | Intervention | Known probability of outcome 1 |
| --- | --- | --- |
| Software pipeline | Delete joint cache/queue routing and install a coupled policy | `11/40 = 0.275` |
| Manufacturing | Replace joint length/strength production, conditional on batch quality | `67/128 = 0.5234375` |
| Policy allocation | Delete joint stipend/training allocation and install a coupled policy | `19/32 = 0.59375` |

Each example retains two downstream mechanisms, generates its observational law directly
from explicit synthetic kernels, and supplies a hand-derived answer independently of the
compiler. Exact integer frequencies remove sampling variation. The script checks both
variable elimination and enumeration against that answer, and prints a JSON report with
the estimand, support result and computational cost. The output is about these declared
models; it is not evidence about a real software system, factory, or public program.

Joint outputs are consequential: replacing a joint intervention policy with the product
of its marginals can change the answer, even though each output has the same marginal law.
