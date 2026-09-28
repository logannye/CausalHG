"""A synthetic allocation rule jointly supplies a stipend and training.

Run after installing the library: ``python examples/policy.py``.
A joint allocation policy changes program uptake, which changes an employment
indicator. These probabilities illustrate the API; they describe no real policy.
"""

from fractions import Fraction as F
from itertools import product

from _finite import bernoulli, print_report, run_example

from causal_hypergraphs import DeleteMechanism, MechanismGraph


def main() -> dict:
    graph = MechanismGraph(
        variables={"eligibility", "stipend", "training", "uptake", "employment"},
        mechanisms={
            "allocation": {"inputs": ("eligibility",), "outputs": ("stipend", "training")},
            "participation": {"inputs": ("stipend", "training"), "outputs": ("uptake",)},
            "placement": {"inputs": ("uptake",), "outputs": ("employment",)},
        },
    )
    uptake = {(0, 0): F(1, 8), (0, 1): F(3, 8), (1, 0): F(1, 2), (1, 1): F(7, 8)}
    allocation = {
        0: {(0, 0): F(1, 2), (0, 1): F(1, 8), (1, 0): F(1, 8), (1, 1): F(1, 4)},
        1: {(0, 0): F(1, 4), (0, 1): F(1, 8), (1, 0): F(1, 8), (1, 1): F(1, 2)},
    }
    joint = {}
    # Tuple order: eligibility, employment, stipend, training, uptake.
    for eligibility, employment, stipend, training, participated in product((0, 1), repeat=5):
        joint[(eligibility, employment, stipend, training, participated)] = (
            F(1, 2) * allocation[eligibility][(stipend, training)]
            * bernoulli(participated, uptake[(stipend, training)])
            * bernoulli(employment, F(1, 4) + F(1, 2) * participated)
        )
    policy = {(0, 0): 0.25, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.75}
    # P(uptake=1) = (1/4)(1/8) + (3/4)(7/8) = 11/16;
    # P(employment=1) = 1/4 + (1/2)(11/16) = 19/32.
    return run_example(
        name="policy allocation", graph=graph,
        query=DeleteMechanism("allocation", outcomes=("employment",)), joint=joint,
        fallbacks={"allocation": policy}, expected=F(19, 32),
    )


if __name__ == "__main__":
    print_report(main())
