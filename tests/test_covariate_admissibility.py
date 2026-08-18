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
    Mechanism,
    MechanismGraph,
    ReplaceMechanism,
    check_covariates,
    d_separated,
)
from tests.conformance.checks import conditional_independence_holds
from tests.conformance.generation import generate_model


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


# --- randomized conformance ---------------------------------------------------------
#
# The fixtures above are hand-built graphs whose answers were written down by the same
# person who wrote the code. What follows checks the report against an *exact
# distributional oracle* over generated models, which is what makes it evidence.
#
# The oracle exists because of what `_severed` does. Removing the target mechanism's
# outputs leaves `out(m*)` with no producer, so in the back-door graph those variables are
# exogenous -- which is precisely the post-deletion model with an independent policy, a law
# the generator already computes exactly. And in that graph the mechanism node has parents
# and no children, so every path out of it leaves through `in(m*)`: `d_separated(back_door,
# m*, Y | Z)` holds exactly when `in(m*)` is separated from `Y` given `Z`. By soundness of
# d*-separation that must be a real conditional independence in the post-deletion law, and
# `conditional_independence_holds` decides it without sampling.

COVARIATE_MODELS = 120


def _reachable_from(graph: MechanismGraph, target: str) -> set[str]:
    """Variables reachable from a mechanism's outputs, computed here, not imported.

    A differential oracle has to be a second implementation. Calling the library's own
    `_descendants` would check only that a function equals itself.
    """
    seen: set[str] = set()
    frontier = list(graph.get_mechanism(target).outputs)
    while frontier:
        variable = frontier.pop()
        if variable in seen:
            continue
        seen.add(variable)
        for name in graph.mechanisms:
            mechanism = graph.get_mechanism(name)
            if variable in mechanism.inputs:
                frontier.extend(mechanism.outputs)
    return seen


def _back_door(graph: MechanismGraph, target: str) -> MechanismGraph:
    """The target's outgoing edges severed -- rebuilt here rather than imported."""
    mechanisms = dict(graph.mechanisms)
    original = graph.get_mechanism(target)
    mechanisms[target] = Mechanism(
        target, inputs=original.inputs, outputs=(), latent=original.latent
    )
    return MechanismGraph(
        variables=graph.variables,
        mechanisms=mechanisms,
        observed_variables=graph.observed_variables,
        fallback_variables=graph.fallback_variables,
    )


@pytest.fixture(scope="module")
def covariate_sweep() -> dict[str, int]:
    """Check every verdict against exact ground truth; fail on the first unsound one."""
    tally = {
        "reports": 0,
        "verdicts": 0,
        "post_treatment": 0,
        "opens_path": 0,
        "admissible": 0,
        "separation_claims_checked": 0,
        "opens_path_with_real_dependence": 0,
        "vacuous_sole_input": 0,
    }
    failures: list[str] = []

    for seed in range(COVARIATE_MODELS):
        model = generate_model(seed, allow_hidden=False)
        graph = model.graph()
        for spec in model.mechanisms:
            mechanism = graph.get_mechanism(spec.name)
            inputs = tuple(mechanism.inputs)
            if not inputs:
                continue
            # The back-door law: `out(m*)` exogenous is exactly the post-deletion model.
            law = model.interventional_delete(spec.name)
            back_door = _back_door(graph, spec.name)
            downstream = _reachable_from(graph, spec.name)

            for outcome in model.variables:
                if outcome in mechanism.outputs or outcome in inputs:
                    continue
                candidates = [v for v in model.variables if v != outcome]
                report = check_covariates(
                    graph, DeleteMechanism(spec.name), outcome, candidates
                )
                tally["reports"] += 1
                separated_alone = d_separated(back_door, spec.name, outcome)

                if separated_alone:
                    tally["separation_claims_checked"] += 1
                    if not conditional_independence_holds(
                        law, model.variables, inputs, (outcome,), ()
                    ):
                        failures.append(
                            f"seed {seed}/{spec.name}/{outcome}: claimed "
                            f"{spec.name} indep {outcome} on the back-door graph, but "
                            f"{list(inputs)} and {outcome} are dependent in the "
                            "post-deletion law"
                        )

                for verdict in report.verdicts:
                    tally["verdicts"] += 1
                    name = verdict.covariate

                    # Structural, so it is exact in both directions.
                    if verdict.post_treatment != (name in downstream):
                        failures.append(
                            f"seed {seed}/{spec.name}/{outcome}: post_treatment="
                            f"{verdict.post_treatment} for {name!r}, reachable="
                            f"{name in downstream}"
                        )
                    if verdict.post_treatment:
                        tally["post_treatment"] += 1
                        continue
                    if name == outcome:
                        continue
                    # Conditioning on one of `in(m*)` blocks only the path leaving through
                    # it, so the analogue drops that parent from the left-hand side rather
                    # than excluding the case. Where it was the only parent nothing is left
                    # to be dependent and the claim is vacuous, which is the one case worth
                    # counting out.
                    left = tuple(v for v in inputs if v != name)
                    if not left:
                        tally["vacuous_sole_input"] += 1
                        continue

                    if verdict.opens_path:
                        tally["opens_path"] += 1
                        if not conditional_independence_holds(
                            law, model.variables, left, (outcome,), (name,)
                        ):
                            tally["opens_path_with_real_dependence"] += 1
                        continue

                    tally["admissible"] += 1
                    if separated_alone:
                        # The report's actual claim: conditioning on this opened nothing
                        # that was blocked. Sound direction, so it is gated.
                        tally["separation_claims_checked"] += 1
                        if not conditional_independence_holds(
                            law, model.variables, left, (outcome,), (name,)
                        ):
                            failures.append(
                                f"seed {seed}/{spec.name}/{outcome}: {name!r} reported "
                                "admissible, but conditioning on it makes "
                                f"{list(left)} and {outcome} dependent in the "
                                "post-deletion law"
                            )

    tally["failures"] = len(failures)
    if failures:
        pytest.fail(
            f"{len(failures)} unsound covariate verdict(s):\n  " + "\n  ".join(failures[:10])
        )
    return tally


def test_every_separation_the_report_relies_on_is_a_real_independence(
    covariate_sweep,
) -> None:
    """Soundness, gated. The sweep fails on the first counterexample; this pins the size."""
    assert covariate_sweep["failures"] == 0
    assert covariate_sweep["separation_claims_checked"] >= 600, covariate_sweep
    assert covariate_sweep["verdicts"] >= 2_000, covariate_sweep


def test_the_post_treatment_finding_is_exact(covariate_sweep) -> None:
    """Structural, so it is checked in both directions against a second implementation.

    Unlike path-opening this rests on no distributional assumption, so a disagreement
    either way is a defect rather than a completeness gap.
    """
    assert covariate_sweep["failures"] == 0
    assert covariate_sweep["post_treatment"] >= 800, covariate_sweep


def test_the_path_opening_lane_actually_fires(covariate_sweep) -> None:
    """A warning that never fires is decoration, and this one nearly cannot fire by accident.

    Path-opening is only detectable on the *back-door* graph. On the full graph the causal
    path keeps the mechanism and the outcome d-connected whatever the covariate does, so the
    separated-then-not transition never happens and every covariate comes back admissible.
    A sweep that reported zero flags would look identical to that bug.
    """
    assert covariate_sweep["opens_path"] >= 25, covariate_sweep


def test_flagged_covariates_usually_correspond_to_a_real_dependence(
    covariate_sweep,
) -> None:
    """Completeness, reported rather than gated -- the same treatment the separation sweep
    gives its own misses.

    d*-separation is complete only under faithfulness, so a flagged covariate *may* open a
    path. Most flags landing on a genuine dependence is evidence the warning is worth
    printing; a flag that lands on none is not a defect, and gating it would turn a
    generic-position argument into a hard requirement.
    """
    flagged = covariate_sweep["opens_path"]
    real = covariate_sweep["opens_path_with_real_dependence"]
    assert real >= 0.75 * flagged, covariate_sweep
