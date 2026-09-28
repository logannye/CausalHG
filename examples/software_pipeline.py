"""A routing stage jointly produces cache and queue states before two service stages.

Run after installing the library: ``python examples/software_pipeline.py``.
All numbers are synthetic. Deletion installs an explicitly coupled joint policy;
it does not mean setting every output to zero.
"""

from fractions import Fraction as F
from itertools import product

from _finite import bernoulli, print_report, run_example

from causal_hypergraphs import DeleteMechanism, MechanismGraph


def main() -> dict:
    graph = MechanismGraph(
        variables={"load", "cache", "queue", "timeout", "error"},
        mechanisms={
            "routing": {"inputs": ("load",), "outputs": ("cache", "queue")},
            "service": {"inputs": ("cache", "queue"), "outputs": ("timeout",)},
            "response": {"inputs": ("timeout",), "outputs": ("error",)},
        },
    )
    timeout = {(0, 0): F(1, 10), (0, 1): F(3, 5), (1, 0): F(1, 20), (1, 1): F(2, 5)}
    routing = {
        0: {(0, 0): F(2, 5), (0, 1): F(1, 10), (1, 0): F(1, 10), (1, 1): F(2, 5)},
        1: {(0, 0): F(1, 10), (0, 1): F(2, 5), (1, 0): F(2, 5), (1, 1): F(1, 10)},
    }
    joint = {}
    # Tuple order is alphabetical: cache, error, load, queue, timeout.
    for cache, error, load, queue, timed_out in product((0, 1), repeat=5):
        joint[(cache, error, load, queue, timed_out)] = (
            F(1, 2) * routing[load][(cache, queue)]
            * bernoulli(timed_out, timeout[(cache, queue)])
            * bernoulli(error, F(1, 10) + F(7, 10) * timed_out)
        )
    policy = {(0, 0): 0.5, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.5}
    # Independent arithmetic: timeout = (1/10 + 2/5)/2 = 1/4;
    # error = 1/10 + (7/10)(1/4) = 11/40.
    return run_example(
        name="software pipeline", graph=graph,
        query=DeleteMechanism("routing", outcomes=("error",)), joint=joint,
        fallbacks={"routing": policy}, expected=F(11, 40),
    )


if __name__ == "__main__":
    print_report(main())
