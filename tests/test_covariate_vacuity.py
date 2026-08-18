"""A check that runs, passes, and cannot report is the failure this module names.

`check_covariates` decided path-opening as

    separated_alone = d_separated(back_door, target, outcome)      # computed once
    opens = separated_alone and not d_separated(back_door, target, outcome, (name,))

so whenever the target and outcome were *already* d-connected in the back-door graph --
which is what an unmeasured common cause produces, and therefore the normal case in any
real system -- `opens` was False for every covariate and the report read

    Admissible: ['A', 'B', 'C', 'W']

with both finding sections empty. Three things were wrong at once, and the second is the
one that makes it more than cosmetic:

1. Nothing distinguished "tested and clean" from "the test could not fire".
2. A collider the module flags correctly on a clean graph goes **unflagged** the moment a
   confounder exists. The finding is not merely unstated, it is lost.
3. `W` is *unobserved*. Covariates were validated against `graph.variable_set`, never
   `graph.observed_set`, so the report advised conditioning on a latent variable.

The module's own docstring warned about exactly this shape for the name-clash case --
"skipping the path-opening test would leave a check that runs, passes, and cannot report
-- every covariate would come back admissible for the wrong reason" -- and the guard was
installed there and not here.
"""

from __future__ import annotations

import pytest

from causal_hypergraphs import DeleteMechanism, MechanismGraph, check_covariates

QUERY = DeleteMechanism("m_T", outcomes=("Y",))
BASE = {
    "m_A": {"inputs": (), "outputs": ("A",)},
    "m_B": {"inputs": (), "outputs": ("B",)},
    "m_C": {"inputs": ("A", "B"), "outputs": ("C",)},
}


def clean_graph() -> MechanismGraph:
    """A -> m_T, and a collider C with parents A and B; no open back-door path."""
    return MechanismGraph(
        variables={"A", "B", "C", "T", "Y"},
        mechanisms={
            **BASE,
            "m_T": {"inputs": ("A",), "outputs": ("T",)},
            "m_Y": {"inputs": ("T", "B"), "outputs": ("Y",)},
        },
    )


def confounded_graph(*, w_observed: bool) -> MechanismGraph:
    """The same graph plus W, a common cause of the target and the outcome."""
    observed = {"A", "B", "C", "T", "Y"} | ({"W"} if w_observed else set())
    return MechanismGraph(
        variables={"A", "B", "C", "T", "Y", "W"},
        mechanisms={
            **BASE,
            "m_W": {"inputs": (), "outputs": ("W",)},
            "m_T": {"inputs": ("A", "W"), "outputs": ("T",)},
            "m_Y": {"inputs": ("T", "B", "W"), "outputs": ("Y",)},
        },
        observed_variables=observed,
    )


def verdict(report, name: str):
    (found,) = [v for v in report.verdicts if v.covariate == name]
    return found


def test_the_collider_is_still_flagged_when_the_back_door_is_clean() -> None:
    """The behaviour that was already right, pinned so the repair cannot cost it."""
    report = check_covariates(clean_graph(), QUERY, "Y", ("A", "B", "C"))

    assert not report.back_door_open
    assert report.opens_path == ("C",)
    assert report.admissible == ("A", "B")


def test_a_confounder_no_longer_hides_the_collider() -> None:
    """Defect 2, and the reason this is not cosmetic: the finding was LOST, not unstated.

    `C` is the same collider as above. Adding an unmeasured common cause elsewhere in the
    graph must not silence a verdict about `C`.
    """
    report = check_covariates(confounded_graph(w_observed=False), QUERY, "Y", ("A", "B", "C"))

    assert report.back_door_open
    assert "C" not in report.admissible


def test_an_open_back_door_is_reported_and_not_read_as_clean() -> None:
    """Defect 1. `admissible` must not be asserted on a test that could not run."""
    report = check_covariates(confounded_graph(w_observed=False), QUERY, "Y", ("A", "B", "C"))

    assert report.back_door_open
    for name in ("A", "B", "C"):
        found = verdict(report, name)
        assert not found.path_test_applicable
        assert not found.admissible
    assert report.admissible == ()
    text = report.summary()
    assert "open back-door path" in text
    assert "Admissible: (none)" in text or "Admissible: []" in text


def test_a_covariate_that_closes_the_back_door_is_named() -> None:
    """The previously-vacuous branch now answers the question a user actually has.

    `W` is the confounder. Conditioning on it *closes* the open back-door path, which is
    the entire purpose of covariate adjustment -- and the old code, which only ever asked
    whether a covariate OPENS a path, could not express it.
    """
    report = check_covariates(confounded_graph(w_observed=True), QUERY, "Y", ("A", "B", "C", "W"))

    assert report.back_door_open
    assert report.blocks_path == ("W",)
    assert verdict(report, "W").admissible
    assert not verdict(report, "A").admissible
    assert "closes" in verdict(report, "W").reason


def test_an_unobserved_covariate_is_never_admissible() -> None:
    """Defect 3. `W` is latent here; you cannot condition on what you did not measure.

    It closes the back-door path in the graph, so a purely graphical check calls it the
    ideal covariate. That is precisely why observation has to be checked separately: the
    graph cannot tell you what the assay measured.
    """
    report = check_covariates(confounded_graph(w_observed=False), QUERY, "Y", ("W",))

    found = verdict(report, "W")
    assert not found.admissible
    assert not found.path_test_applicable
    assert "not observed" in found.reason
    assert report.blocks_path == ()


def test_post_treatment_still_outranks_everything() -> None:
    """A descendant of the intervention is refused whatever the back-door graph says."""
    graph = MechanismGraph(
        variables={"A", "B", "C", "T", "Y", "W", "M"},
        mechanisms={
            **BASE,
            "m_W": {"inputs": (), "outputs": ("W",)},
            "m_T": {"inputs": ("A", "W"), "outputs": ("T",)},
            "m_M": {"inputs": ("T",), "outputs": ("M",)},
            "m_Y": {"inputs": ("T", "B", "W"), "outputs": ("Y",)},
        },
        observed_variables={"A", "B", "C", "T", "Y", "M"},
    )
    report = check_covariates(graph, DeleteMechanism("m_T", outcomes=("Y",)), "Y", ("M",))

    found = verdict(report, "M")
    assert found.post_treatment
    assert not found.admissible
    assert "Structural" in found.reason


def test_the_report_leads_with_the_graph_level_finding() -> None:
    """An open back-door is a fact about the query, not about any one covariate."""
    confounded = check_covariates(confounded_graph(w_observed=True), QUERY, "Y", ("A", "W"))
    clean = check_covariates(clean_graph(), QUERY, "Y", ("A", "B"))

    assert "open back-door path" in confounded.summary()
    assert "open back-door path" not in clean.summary()
    # And the closing covariate is surfaced, not buried in the per-covariate reasons.
    assert "W" in confounded.summary()


@pytest.mark.parametrize("w_observed", [True, False])
def test_admissible_never_includes_a_covariate_whose_test_could_not_run(
    w_observed: bool,
) -> None:
    """The invariant the whole repair exists to establish, stated once, directly."""
    report = check_covariates(
        confounded_graph(w_observed=w_observed), QUERY, "Y", ("A", "B", "C", "W")
    )
    for found in report.verdicts:
        if not found.path_test_applicable:
            assert not found.admissible, found.covariate
