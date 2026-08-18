"""Deleting a mechanism whose output is hidden is not a missing algorithm. It is a proof.

`identify` distinguishes three outcomes, and the distinction is the point of the library:
`Identified` (here is the formula), `Unknown` (this compiler cannot do it, and here is what
would help), and `Unidentified` (no formula exists). Returning `Unknown` where the truth is
`Unidentified` is not conservative -- it promises an algorithm that cannot exist, and sends
the reader to look for one.

That is what the T7 track does today for the commonest hidden-boundary shape. Over the
conformance generator's models, of the mechanisms whose boundary contains a hidden
variable, roughly four in five have a hidden *output* rather than a hidden input. Every one
of them is refused as `Unknown` with `next_algorithm="T7 Pearl-ID reduction"`.

No reduction will identify them, and the reason is short. `delete(m)` installs a policy
`P0^m(out(m))` over the *values* of `out(m)`. When one of those values is a variable the
data never records, nothing in the observed distribution pins down which value is which:
relabelling a hidden variable is a symmetry of every observable and is not a symmetry of a
policy defined on its labels. So two models can agree on everything measurable and disagree
on the answer, which is exactly non-identifiability.

The rule is decidable from the graph alone, with no model in hand:

- a hidden output with at least one observed descendant  -> `Unidentified`, with the
  relabelling witness;
- hidden outputs, none of which reaches an observation   -> the deletion cannot move any
  observable, so the observed law is unchanged. That is an answer, not a refusal;
- outputs all observed, some input hidden                -> the genuine T7 case.

One honesty note, kept in the refusal itself rather than in a comment: the witness needs
`P0` to distinguish the labels it permutes. A policy that is invariant under every
permutation of the hidden output -- a uniform one, say -- escapes the argument. `identify`
never sees `P0`'s numbers, so it must answer for a general policy, and this refusal says so.
"""
from __future__ import annotations

import pytest

from causal_hypergraphs import (
    DeleteMechanism,
    Identified,
    MechanismGraph,
    RelabellingWitness,
    Unidentified,
    identify,
)


def _hidden_output_graph() -> MechanismGraph:
    """m_h produces the hidden `h`; `Y` is observed and downstream of it."""
    return MechanismGraph(
        variables={"h", "Y"},
        mechanisms={
            "m_h": {"inputs": (), "outputs": ("h",)},
            "m_y": {"inputs": ("h",), "outputs": ("Y",)},
        },
        observed_variables={"Y"},
    )


# --- the verdict ------------------------------------------------------------------


def test_deleting_a_mechanism_with_a_hidden_output_is_unidentified_not_unknown() -> None:
    """The verdict must say "no formula exists", not "we have not built it yet"."""
    result = identify(_hidden_output_graph(), DeleteMechanism("m_h", outcomes={"Y"}), allow_t7=True)

    assert isinstance(result, Unidentified), result
    assert "h" in result.reason
    assert "relabel" in result.reason.lower()


def test_the_refusal_carries_a_witness_naming_the_hidden_output() -> None:
    """A refusal with no witness is an opinion. The witness is what makes it checkable."""
    result = identify(_hidden_output_graph(), DeleteMechanism("m_h", outcomes={"Y"}), allow_t7=True)

    assert isinstance(result, Unidentified)
    witness = result.witness
    assert isinstance(witness, RelabellingWitness), witness
    assert witness.hidden_outputs == ("h",)
    assert "Y" in witness.observed_descendants


def test_the_refusal_states_the_case_it_does_not_cover() -> None:
    """A permutation-invariant policy escapes the argument, and the refusal says so
    rather than overclaiming, because `identify` never sees the policy's numbers."""
    result = identify(_hidden_output_graph(), DeleteMechanism("m_h", outcomes={"Y"}), allow_t7=True)

    assert isinstance(result, Unidentified)
    assert isinstance(result.witness, RelabellingWitness)
    assert "invariant" in result.witness.explanation.lower()


# --- the witness is real ----------------------------------------------------------


def test_the_witness_construction_actually_exhibits_two_indistinguishable_models() -> None:
    """The claim, executed. Two models over the same graph, identical on every observable,
    different after the deletion.

    This is the whole justification for returning `Unidentified`, so it is computed here
    rather than asserted. Arithmetic only -- the point is the *existence* of the pair, and
    building it through the library's own evaluator would prove a fact about the evaluator.
    """
    prior = {0: 0.5, 1: 0.5}
    original = {0: {0: 0.9, 1: 0.1}, 1: {0: 0.2, 1: 0.8}}          # P(Y | h)
    relabelled = {0: original[1], 1: original[0]}                   # h's values swapped
    policy = {0: 0.8, 1: 0.2}

    def observed(kernel: dict) -> dict[int, float]:
        return {y: sum(prior[h] * kernel[h][y] for h in (0, 1)) for y in (0, 1)}

    def after_deletion(kernel: dict) -> dict[int, float]:
        return {y: sum(policy[h] * kernel[h][y] for h in (0, 1)) for y in (0, 1)}

    for y in (0, 1):
        assert observed(original)[y] == pytest.approx(observed(relabelled)[y], abs=1e-12)

    gap = max(abs(after_deletion(original)[y] - after_deletion(relabelled)[y]) for y in (0, 1))
    assert gap > 0.4, gap


def test_a_uniform_policy_is_the_case_the_witness_cannot_reach() -> None:
    """The stated exception, also executed. Under a permutation-invariant policy the two
    models agree after the deletion too, so the witness proves nothing there -- which is
    why the refusal names the exception instead of claiming to have ruled it out."""
    prior = {0: 0.5, 1: 0.5}
    original = {0: {0: 0.9, 1: 0.1}, 1: {0: 0.2, 1: 0.8}}
    relabelled = {0: original[1], 1: original[0]}
    uniform = {0: 0.5, 1: 0.5}

    def after_deletion(kernel: dict) -> dict[int, float]:
        return {y: sum(uniform[h] * kernel[h][y] for h in (0, 1)) for y in (0, 1)}

    for y in (0, 1):
        assert after_deletion(original)[y] == pytest.approx(
            after_deletion(relabelled)[y], abs=1e-12
        )
    assert prior == uniform  # the fixture's symmetry, stated so it is not a coincidence


# --- the case that is an answer rather than a refusal ------------------------------


def test_a_hidden_output_that_reaches_nothing_observed_leaves_the_observed_law_alone() -> None:
    """No observed descendant means the intervention cannot move any observable.

    So the answer is the observational law -- identified, and stronger than a numerical
    coincidence: the estimand mentions neither the mechanism nor its policy, so it *cannot*
    depend on what the deletion installs.
    """
    graph = MechanismGraph(
        variables={"h", "Y"},
        mechanisms={
            "m_h": {"inputs": (), "outputs": ("h",)},
            "m_y": {"inputs": (), "outputs": ("Y",)},
        },
        observed_variables={"Y"},
    )
    result = identify(graph, DeleteMechanism("m_h", outcomes={"Y"}), allow_t7=True)

    assert isinstance(result, Identified), result
    assert str(result.expression) == "P(Y)"


# --- the hidden-input case is untouched -------------------------------------------


def test_a_hidden_input_is_still_the_pearl_reduction_case() -> None:
    """The classification must separate the two, not refuse both.

    The front-door example has a hidden *input* and is identified through the Pearl
    reduction; nothing about the hidden-output verdict may touch it.
    """
    from causal_hypergraphs.examples import frontdoor_hidden_boundary_graph

    result = identify(
        frontdoor_hidden_boundary_graph(), DeleteMechanism("m_x", outcomes={"Y"}), allow_t7=True
    )

    assert isinstance(result, Identified), result
    assert result.theorem == "T7"


def test_a_dead_end_hidden_output_does_not_make_the_whole_deletion_invisible() -> None:
    """The reach question is about *all* of `out(m)`, not only its hidden part.

    A mechanism producing an observed `C` and a hidden dead-end `h` still resets `C` when
    it is deleted, so anything downstream of `C` moves. Asking only whether the *hidden*
    outputs reach an observation gets the predicate right and the population wrong, and
    answers `P(outcomes)` -- which is not a conservative answer but a wrong one, since the
    observational law is exactly what the intervention changes.

    Verified numerically below rather than only structurally, because the failure mode is a
    plausible-looking estimand rather than a crash.
    """
    graph = MechanismGraph(
        variables={"C", "h", "Y"},
        mechanisms={
            "m1": {"inputs": (), "outputs": ("C", "h")},
            "m2": {"inputs": ("C",), "outputs": ("Y",)},
        },
        observed_variables={"C", "Y"},
    )
    result = identify(graph, DeleteMechanism("m1", outcomes={"Y"}), allow_t7=True)

    if isinstance(result, Identified):
        assert str(result.expression) != "P(Y)", (
            "deleting m1 resets the observed output C, so P(Y) is the one answer that "
            "cannot be right"
        )

    # The arithmetic the wrong estimand would get wrong.
    policy = {0: 0.8, 1: 0.2}
    conditional = {0: {0: 0.9, 1: 0.1}, 1: {0: 0.1, 1: 0.9}}
    observational_c = {0: 0.5, 1: 0.5}
    observational_y = {
        y: sum(observational_c[c] * conditional[c][y] for c in (0, 1)) for y in (0, 1)
    }
    after = {y: sum(policy[c] * conditional[c][y] for c in (0, 1)) for y in (0, 1)}
    assert abs(observational_y[0] - after[0]) > 0.2


def test_a_hidden_output_reaching_nothing_still_needs_its_siblings_checked() -> None:
    """The genuinely invisible case, kept sharp: *nothing* in `out(m)` reaches an
    observation. Only then is the observational law the answer."""
    graph = MechanismGraph(
        variables={"h1", "h2", "Y"},
        mechanisms={
            "m1": {"inputs": (), "outputs": ("h1", "h2")},
            "m_y": {"inputs": (), "outputs": ("Y",)},
        },
        observed_variables={"Y"},
    )
    result = identify(graph, DeleteMechanism("m1", outcomes={"Y"}), allow_t7=True)

    assert isinstance(result, Identified), result
    assert str(result.expression) == "P(Y)"
