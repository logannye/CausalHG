import itertools
import random
from pathlib import Path

from causal_hypergraphs import (
    Dataset,
    DeleteMechanism,
    Identified,
    MechanismGraph,
    Unknown,
    check_covariates,
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


def test_readme_names_are_importable_from_the_top_level_package() -> None:
    """A README that shows a bare call the top-level package cannot supply is a dead end.

    `plan_elimination` is printed in the cost section with no import line, and it did not
    live on the package root -- so the natural `from causal_hypergraphs import
    plan_elimination` raised `ImportError` and the README never printed the path that
    works. Every name the README uses in a code block must resolve from one import.
    """
    import causal_hypergraphs

    for name in ("plan_elimination", "check_covariates", "identify_expectation"):
        assert hasattr(causal_hypergraphs, name), name
        assert name in causal_hypergraphs.__all__, f"{name} is reachable but undeclared"


def test_readme_feedback_loop_block() -> None:
    """The cyclic-graph block, including the component ordering it prints.

    The README claimed `(('m1', 'm2'), ('far',))`. Components come back sorted, so the
    real answer leads with `far` -- and being sorted is the documented guarantee that a
    cost or a refusal never varies with dictionary ordering, which makes a wrong printed
    order a claim about determinism, not a typo.
    """
    graph = MechanismGraph(
        variables={"a", "b", "Y", "q", "R"},
        mechanisms={
            "m1": {"inputs": {"a"}, "outputs": {"b"}},
            "m2": {"inputs": {"b"}, "outputs": {"a"}},
            "m3": {"inputs": {"b"}, "outputs": {"Y"}},
            "far": {"inputs": {"q"}, "outputs": {"R"}},
        },
    )

    assert graph.cyclic_mechanisms == frozenset({"m1", "m2"})
    assert graph.mechanism_components() == (("far",), ("m1", "m2"), ("m3",))

    # The loop is in another component.
    assert isinstance(identify(graph, DeleteMechanism("far", outcomes={"R"})), Identified)

    # Deleting m1 severs its own input, so the loop stops being upstream of anything the
    # answer needs. This is the carve-out that makes feedback-upstream-of-a-knockdown
    # answerable at all.
    severed = identify(graph, DeleteMechanism("m1", outcomes={"Y"}))
    assert isinstance(severed, Identified), severed
    assert str(severed.expression) == "sum_{b} P(Y | b) * P0_m1(b)"
    assert not ({"a"} & severed.expression.footprint()), severed.expression

    # And its limit: deleting m2 breaks the cycle too, but the answer still needs m1's
    # kernel, and for a mechanism on an observational cycle that conditional is not its
    # structural kernel. `far` cannot reach Y at all, so it needs the whole loop.
    assert isinstance(identify(graph, DeleteMechanism("m2", outcomes={"Y"})), Unknown)
    assert isinstance(identify(graph, DeleteMechanism("far", outcomes={"Y"})), Unknown)


def test_readme_covariate_block_prints_what_the_readme_shows() -> None:
    """Three literal differences hid here: a dropped header, a truncated sentence, and
    `not a proof:` where the code says `not a proof of harm:`.

    The distinction the last one carries is the whole point of the section -- d-separation
    is sound but complete only under faithfulness, so a detected path is a warning and not
    a demonstration that adjusting does harm.
    """
    graph = MechanismGraph(
        variables={"donor", "stim", "batch", "TF", "exhaustion_marker", "IFNG"},
        mechanisms={
            "knockdown": {"inputs": {"donor", "stim"}, "outputs": {"TF"}},
            "m_marker": {"inputs": {"TF"}, "outputs": {"exhaustion_marker"}},
            "m_ifng": {"inputs": {"TF", "batch"}, "outputs": {"IFNG"}},
        },
    )

    summary = check_covariates(
        graph,
        DeleteMechanism("knockdown"),
        "IFNG",
        ["donor", "stim", "exhaustion_marker", "batch"],
    ).summary()

    assert "Conditioning around do(knockdown) with outcome 'IFNG':" in summary
    assert "Structural, not an assumption." in summary
    assert "this is not a proof of harm:" in summary
    assert "Admissible: ['donor', 'stim', 'batch']" in summary


def test_readme_empty_stratum_block_reports_the_counts_it_prints() -> None:
    """`3 of 16 point(s) undefined` was impossible for this estimand: the scope is 64
    points and an empty (C=1, E=1) cell takes out every one of the 16 that names it.

    Worse, the block above it indexed a point inside that very stratum, so a reader
    following the README in order would have hit a `KeyError` on the line after the one
    telling them the cell was empty.
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
    records = []
    for index in range(3_000):
        c = rng.randint(0, 1)
        records.append(
            {
                "A": rng.randint(0, 1),
                "B": rng.randint(0, 1),
                "C": c,
                "D": rng.randint(0, 1),
                # (C=1, E=1) never co-occurs, so that conditioning cell is empty while
                # both columns still show both levels.
                "E": 0 if c == 1 else rng.randint(0, 1),
                "F": rng.randint(0, 1),
                "donor": f"d{index % 20}",
            }
        )

    est = estimate(
        result,
        Dataset.from_records(records, unit="donor"),
        fallbacks={"m1": {(0, 0): 0.5, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.5}},
    )
    summary = est.summary()

    # Compared as a CONTIGUOUS BLOCK, not by three substring probes. The README shows a
    # region of output, so the claim is about the region: three `in summary` assertions
    # stayed green while `Policy support` was inserted between the second line and the
    # third, and the README quietly under-reported what the tool prints. A gate a wrong
    # document passes is the shape this file exists to prevent.
    block = "Checked against the data:" + summary.split("Checked against the data:")[1]
    assert [line.strip() for line in block.strip().splitlines()] == [
        "Checked against the data:",
        "Downstream positivity: FAIL",
        "16 of 64 point(s) undefined across 1 empty stratum/strata",
        "Policy support: PASS",
        "P0_m1 rests on 1500 effective row(s) of 3000 (2x the reported count)",
        "! P(F | C,E) undefined at C=1, E=1 (16 point(s) unreachable)",
    ]
    # ...and the README must show exactly that block.
    readme = Path("README.md").read_text()
    for line in block.strip().splitlines():
        assert line.strip() in readme, line.strip()
    # Absent, never nan: the affected points are not in `values` at all.
    assert (0, 1, 1, 0, 1, 1) not in est.values


def test_readme_open_back_door_block_prints_what_the_readme_shows() -> None:
    """The third covariate finding, gated like the two the README already showed.

    The README block is a claim about output, and the last three claims it made about this
    module's output were wrong rather than stale. This one is checked line for line --
    including `Closes it: ['donor']`, which is the whole point of the section: a check that
    only asks "does conditioning OPEN a path?" cannot name the covariate that closes one.
    """
    graph = MechanismGraph(
        variables={"donor", "stim", "batch", "TF", "exhaustion_marker", "IFNG"},
        mechanisms={
            # `donor` now feeds the outcome as well as the target: an open back-door path.
            "knockdown": {"inputs": {"donor", "stim"}, "outputs": {"TF"}},
            "m_marker": {"inputs": {"TF"}, "outputs": {"exhaustion_marker"}},
            "m_ifng": {"inputs": {"TF", "batch", "donor"}, "outputs": {"IFNG"}},
        },
    )

    report = check_covariates(
        graph,
        DeleteMechanism("knockdown"),
        "IFNG",
        ["donor", "stim", "exhaustion_marker", "batch"],
    )
    summary = report.summary()

    assert report.back_door_open
    assert report.blocks_path == ("donor",)
    assert "there is an open back-door path, so the effect is confounded before any" in summary
    assert "Closes it: ['donor']" in summary
    assert "reported as undecided, not as clean" in summary
    # And the invariant the whole repair exists for, asserted on the README's own example.
    for verdict in report.verdicts:
        if not verdict.path_test_applicable:
            assert not verdict.admissible, verdict.covariate
