import itertools
import random

from causal_hypergraphs import (
    Dataset,
    DeleteMechanism,
    Identified,
    MechanismGraph,
    estimate,
    identify,
)


def test_readme_primary_snippet_smoke() -> None:
    graph = MechanismGraph(
        variables={"A", "B", "C", "D", "E", "F"},
        mechanisms={
            "m1": {"inputs": {"A", "B"}, "outputs": {"C", "D"}},
            "m2": {"inputs": {"C", "E"}, "outputs": {"F"}},
        },
        observed_variables={"A", "B", "C", "D", "E", "F"},
    )

    result = identify(graph, DeleteMechanism("m1"))

    # Narrowing is the point of the result hierarchy: an estimand is unreachable until the
    # caller has established that the query was identified.
    assert isinstance(result, Identified)
    assert result.status == "identified"
    assert str(result.expression) == "P(A) * P(B) * P(E) * P(F | C,E) * P0_m1(C,D)"
    assert result.theorem == "T2"


def test_readme_estimation_snippet_smoke() -> None:
    """The README's estimation example must run, and produce what it says it produces.

    A snippet that no longer executes is worse than no snippet: it is the first thing a
    reader tries, and its failure is attributed to the library rather than to the docs.
    """
    graph = MechanismGraph(
        variables={"A", "B", "C", "D", "E", "F"},
        mechanisms={
            "m1": {"inputs": {"A", "B"}, "outputs": {"C", "D"}},
            "m2": {"inputs": {"C", "E"}, "outputs": {"F"}},
        },
    )
    result = identify(graph, DeleteMechanism("m1"))
    assert isinstance(result, Identified)

    rng = random.Random(0)
    names = ("A", "B", "C", "D", "E", "F")
    records = [
        dict(zip(names, values, strict=True), donor=f"d{index % 20}")
        for index, values in enumerate(
            rng.choices(list(itertools.product((0, 1), repeat=len(names))), k=2_000)
        )
    ]

    data = Dataset.from_records(records, unit="donor")
    est = estimate(
        result,
        data,
        fallbacks={"m1": {(0, 0): 0.5, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.5}},
        bootstrap=50,
    )

    point = (0, 1, 1, 0, 1, 1)
    assert isinstance(est.values[point], float)
    assert est.interval[point] is not None
    assert "donor" in est.unit

    summary = est.summary()
    assert "Not checked" in summary
    assert "Checked against the data" in summary
    # Both certificate codes the README names must be the ones the estimator recognizes.
    assert est.support.checked == ("Downstream positivity",)
