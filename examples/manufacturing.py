"""Replace a shaping process that jointly determines length and strength.

Run after installing the library: ``python examples/manufacturing.py``.
This synthetic replacement reads batch quality and preserves the typed boundary.
Inspection and shipping are two further mechanisms whose kernels remain unchanged.
"""

from fractions import Fraction as F
from itertools import product

from _finite import bernoulli, print_report, run_example

from causal_hypergraphs import Mechanism, MechanismGraph, ReplaceMechanism


def main() -> dict:
    graph = MechanismGraph(
        variables={"batch", "length", "strength", "inspection", "shipped"},
        mechanisms={
            "shaping": {"inputs": ("batch",), "outputs": ("length", "strength")},
            "inspection": {"inputs": ("length", "strength"), "outputs": ("inspection",)},
            "shipping": {"inputs": ("inspection",), "outputs": ("shipped",)},
        },
    )
    joint = {}
    # Tuple order: batch, inspection, length, shipped, strength.
    for batch, inspection, length, shipped, strength in product((0, 1), repeat=5):
        joint[(batch, inspection, length, shipped, strength)] = (
            bernoulli(batch, F(1, 4)) * F(1, 4)
            * bernoulli(inspection, F(1, 8) + F(3, 4) * length * strength)
            * bernoulli(shipped, F(1, 4) + F(1, 2) * inspection)
        )
    replacement = {}
    for batch in (0, 1):
        both_pass = F(1, 2) + F(1, 4) * batch
        for length, strength in product((0, 1), repeat=2):
            replacement[((length, strength), (batch,))] = float(
                bernoulli(length, both_pass) if length == strength else 0
            )
    query = ReplaceMechanism(
        "shaping",
        Mechanism("precision_shaping", inputs=("batch",), outputs=("length", "strength")),
        outcomes=("shipped",),
    )
    # P(length=strength=1) = (3/4)(1/2) + (1/4)(3/4) = 9/16.
    # P(inspection=1) = 1/8 + (3/4)(9/16) = 35/64;
    # P(shipped=1) = 1/4 + (1/2)(35/64) = 67/128.
    return run_example(
        name="manufacturing", graph=graph, query=query, joint=joint,
        replacements={"precision_shaping": replacement}, expected=F(67, 128),
    )


if __name__ == "__main__":
    print_report(main())
