"""Replacement incidence: verify it when it is available, assume it only when it is not.

SPEC.md records `rho(m') = rho(m)` as an assumption certificate because v1 takes the
replacement by name and has no object to check. When the caller does supply the
replacement's incidence, an unchecked certificate is no longer honest: the compiler can
discharge it, and must reject a mismatch rather than emit a proof-carrying estimand for a
query whose semantics are undefined.
"""
from __future__ import annotations

import pytest

from causal_hypergraphs import Identified, ReplaceMechanism, identify
from causal_hypergraphs.examples import reaction_graph
from causal_hypergraphs.graph import Mechanism

INCIDENCE_ASSUMPTION = "Replacement incidence"


def test_name_only_replacement_still_records_the_assumption() -> None:
    """With no object to check, the certificate is the honest outcome. Unchanged behaviour."""
    result = identify(reaction_graph(), ReplaceMechanism("m1", "m1_prime"))

    assert isinstance(result, Identified)
    assert any(item.code == INCIDENCE_ASSUMPTION for item in result.assumptions)
    assert all(step.label != "Verify replacement incidence" for step in result.derivation)


def test_matching_incidence_discharges_the_assumption_into_a_proof_step() -> None:
    """Supplying a conforming replacement turns an assumption into something proved."""
    replacement = Mechanism("m1_prime", inputs=("A", "B"), outputs=("C", "D"))
    result = identify(reaction_graph(), ReplaceMechanism("m1", replacement))

    assert isinstance(result, Identified)
    assert str(result.expression).endswith("P_m1_prime(C,D | A,B)")
    assert all(item.code != INCIDENCE_ASSUMPTION for item in result.assumptions)
    assert any(step.label == "Verify replacement incidence" for step in result.derivation)


def test_mismatched_incidence_is_rejected() -> None:
    """rho(m') != rho(m) makes the query ill-formed; refuse rather than certify it."""
    wrong_outputs = Mechanism("m1_prime", inputs=("A", "B"), outputs=("C",))
    with pytest.raises(ValueError, match="incidence"):
        identify(reaction_graph(), ReplaceMechanism("m1", wrong_outputs))

    wrong_inputs = Mechanism("m1_prime", inputs=("A",), outputs=("C", "D"))
    with pytest.raises(ValueError, match="incidence"):
        identify(reaction_graph(), ReplaceMechanism("m1", wrong_inputs))


def test_replacement_label_must_be_usable_in_an_estimand() -> None:
    """The name is rendered into the estimand and its LaTeX, so it must be a real label."""
    for bad in ("", "   ", "m1 prime", "; DROP TABLE --"):
        with pytest.raises(ValueError, match="replacement name"):
            ReplaceMechanism("m1", bad)
