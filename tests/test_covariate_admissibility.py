"""Is this covariate safe to condition on?

The compiler's own identifiers need no adjustment set -- the truncated factorization is a
complete answer. But nobody runs an estimator in isolation. Analysts stratify, filter to a
subpopulation, add a covariate to a regression, and the question of which variables may be
conditioned on is where real analyses break.

The specific trap this exists for: conditioning on a marker that is itself *downstream of
the perturbation*. It looks like sensible covariate control -- the marker is a real
biological quantity that varies across cells -- and it silently destroys the estimate,
because conditioning on a mediator removes part of the very effect being measured. The
graph knows this; nothing was asking it.

Two failure modes, and they do not have the same evidential status, so the report does not
present them as though they do:

*Post-treatment* is structural. A covariate downstream of the intervened mechanism is
post-treatment as a fact about the graph, with no distributional assumption behind it.

*Path opening* is a warning. It is detected by `d_separated` going from a separation
verdict to a non-verdict, and `d_separated` is sound but complete only under faithfulness.
So a flagged covariate is one that *may* open a path, not one proven to.
"""
from __future__ import annotations

import pytest

from causal_hypergraphs import (
    DeleteMechanism,
    MechanismGraph,
    ReplaceMechanism,
    check_covariates,
)


def _mediator_graph() -> MechanismGraph:
    """perturbed -> marker -> readout. `marker` is the trap."""
    return MechanismGraph(
        variables={"context", "perturbed", "marker", "readout"},
        mechanisms={
            "target": {"inputs": ("context",), "outputs": ("perturbed",)},
            "middle": {"inputs": ("perturbed",), "outputs": ("marker",)},
            "output": {"inputs": ("marker",), "outputs": ("readout",)},
        },
    )


def _verdict(report: object, name: str):
    return next(v for v in report.verdicts if v.covariate == name)  # type: ignore[attr-defined]


# --- post-treatment ---------------------------------------------------------------


def test_a_downstream_marker_is_flagged_post_treatment() -> None:
    """The trap, stated: a perturbation-target marker is not a covariate."""
    report = check_covariates(
        _mediator_graph(), DeleteMechanism("target"), "readout", ["marker"]
    )
    verdict = _verdict(report, "marker")

    assert verdict.post_treatment
    assert not verdict.admissible
    assert "post-treatment" in verdict.reason.lower()
    assert not report.admissible
    assert "marker" in report.summary()


def test_an_output_of_the_intervened_mechanism_is_post_treatment() -> None:
    """The treated variables themselves are the extreme case of post-treatment."""
    report = check_covariates(
        _mediator_graph(), DeleteMechanism("target"), "readout", ["perturbed"]
    )

    assert _verdict(report, "perturbed").post_treatment


def test_a_pre_treatment_variable_is_admissible() -> None:
    """The check must let something through, or it is just a refusal that always fires."""
    report = check_covariates(
        _mediator_graph(), DeleteMechanism("target"), "readout", ["context"]
    )
    verdict = _verdict(report, "context")

    assert verdict.admissible
    assert not verdict.post_treatment
    assert not verdict.opens_path
    assert report.admissible == ("context",)


# --- collider opening -------------------------------------------------------------


def test_a_collider_descendant_is_flagged_as_opening_a_path() -> None:
    """Conditioning can create a dependence as well as destroy one.

    `probe` is produced from both the intervention's own input and the readout's, so its
    producing mechanism is a collider on the path between them. `probe` is *not* downstream
    of the intervention, so this isolates the second criterion from the first: it is
    flagged for opening a path and not for being post-treatment.
    """
    graph = MechanismGraph(
        variables={"a", "b", "perturbed", "readout", "probe"},
        mechanisms={
            "target": {"inputs": ("a",), "outputs": ("perturbed",)},
            "output": {"inputs": ("b",), "outputs": ("readout",)},
            "assay": {"inputs": ("a", "b"), "outputs": ("probe",)},
        },
    )
    report = check_covariates(graph, DeleteMechanism("target"), "readout", ["probe"])
    verdict = _verdict(report, "probe")

    assert verdict.opens_path
    assert not verdict.post_treatment
    assert not verdict.admissible
    assert "open" in verdict.reason.lower()
    # The warning is explicitly weaker than the structural one.
    assert "faithful" in verdict.reason.lower()


# --- degenerate inputs ------------------------------------------------------------


def test_the_outcome_itself_is_never_admissible() -> None:
    report = check_covariates(
        _mediator_graph(), DeleteMechanism("target"), "readout", ["readout"]
    )
    verdict = _verdict(report, "readout")

    assert not verdict.admissible
    assert "outcome" in verdict.reason.lower()


def test_an_unknown_covariate_is_rejected() -> None:
    with pytest.raises(ValueError, match="ghost"):
        check_covariates(_mediator_graph(), DeleteMechanism("target"), "readout", ["ghost"])


def test_an_unknown_outcome_is_rejected() -> None:
    with pytest.raises(ValueError, match="ghost"):
        check_covariates(_mediator_graph(), DeleteMechanism("target"), "ghost", ["context"])


def test_a_name_shared_by_a_mechanism_and_a_variable_raises() -> None:
    """A check that cannot report must not report success.

    If a mechanism and a variable share a name they are one node in the bipartite blowup,
    and the separation queries become ambiguous. Skipping the path-opening test there would
    leave every covariate looking admissible for a reason that has nothing to do with the
    covariate -- the failure mode where a guard runs, passes, and cannot fire.
    """
    graph = MechanismGraph(
        variables={"target", "x", "y"},
        mechanisms={"target": {"inputs": ("x",), "outputs": ("y",)}},
    )

    with pytest.raises(ValueError, match="both a mechanism and a variable"):
        check_covariates(graph, DeleteMechanism("target"), "y", ["x"])


def test_replacement_queries_are_checked_the_same_way() -> None:
    """`replace` leaves the wiring intact, so downstream is downstream either way."""
    report = check_covariates(
        _mediator_graph(), ReplaceMechanism("target", "target_prime"), "readout", ["marker"]
    )

    assert _verdict(report, "marker").post_treatment


# --- the report -------------------------------------------------------------------


def test_the_report_separates_the_two_kinds_of_finding() -> None:
    """Structural fact and faithfulness-dependent warning must not read alike."""
    graph = MechanismGraph(
        variables={"a", "b", "perturbed", "marker", "readout", "probe"},
        mechanisms={
            "target": {"inputs": ("a",), "outputs": ("perturbed",)},
            "middle": {"inputs": ("perturbed",), "outputs": ("marker",)},
            "output": {"inputs": ("b", "marker"), "outputs": ("readout",)},
            "assay": {"inputs": ("a", "b"), "outputs": ("probe",)},
        },
    )
    report = check_covariates(
        graph, DeleteMechanism("target"), "readout", ["a", "b", "marker", "probe"]
    )

    assert set(report.admissible) == {"a", "b"}
    assert "marker" in report.post_treatment
    assert "probe" in report.opens_path
    summary = report.summary()
    assert "post-treatment" in summary.lower()
    assert summary.index("post-treatment") < summary.lower().index("faithful")
