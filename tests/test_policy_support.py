"""The row count in the header is not the row count behind the answer.

A deletion estimand is ``sum_t (downstream)(t) * P0^m(t)``. Each level ``t`` weights a
factor estimated from the rows that *show* that level, so a policy leaning on a level the
data barely populates produces an answer resting on those few rows -- while the report
says `11183 row(s) in 8 unit(s)`. Positivity does not catch it: the cell is non-empty, so
every existing certificate reads PASS. This is the same shape as the defect PR #11 closed
(policy mass the data cannot represent) one step further in: mass the data can *technically*
represent and cannot *support*.

Measured on real Perturb-seq data (Norman 2019, GSE133344) before this module existed: a
CRISPRa arm's policy put 0.931 of its mass on one level of the target, and the estimate was
reported against a headline of 11,183 rows.
"""

from __future__ import annotations

import pytest

from causal_hypergraphs import DeleteMechanism, MechanismGraph, identify
from causal_hypergraphs.estimation import Dataset, estimate


def _chain_graph() -> MechanismGraph:
    """A -> T -> Y. Deleting m_T installs a policy on T; Y is the readout."""
    return MechanismGraph(
        variables={"A", "T", "Y"},
        mechanisms={
            "m_A": {"inputs": (), "outputs": ("A",)},
            "m_T": {"inputs": ("A",), "outputs": ("T",)},
            "m_Y": {"inputs": ("T",), "outputs": ("Y",)},
        },
    )


def _skewed_data(rare: int, common: int) -> Dataset:
    """`rare` rows at T=1 and `common` rows at T=0, every cell non-empty."""
    rows = []
    for index in range(common):
        rows.append({"unit": index % 10, "A": 0, "T": 0, "Y": index % 2})
    for index in range(rare):
        rows.append({"unit": index % 10, "A": 1, "T": 1, "Y": index % 2})
    return Dataset.from_records(
        rows, unit="unit", domains={"A": (0, 1), "T": (0, 1), "Y": (0, 1)}
    )


def _estimate(data: Dataset, policy: dict[tuple[int], float]):
    result = identify(_chain_graph(), DeleteMechanism("m_T", outcomes=("Y",)))
    return estimate(result, data, fallbacks={"m_T": policy})


def test_a_policy_leaning_on_a_rare_level_reports_the_rows_it_actually_rests_on() -> None:
    """The number that matters is the effective one, and it must be far below the header."""
    data = _skewed_data(rare=40, common=4000)
    estimated = _estimate(data, {(0,): 0.05, (1,): 0.95})

    assert data.n_rows == 4040
    (policy,) = estimated.policy
    assert policy.mechanism == "m_T"
    # 1 / (0.05**2 / 4000 + 0.95**2 / 40) == 44.3..., i.e. the 40 rare rows, not 4040.
    assert policy.effective_n == pytest.approx(44.3, abs=0.5)
    assert policy.overstatement == pytest.approx(4040 / 44.3, rel=0.02)
    assert policy.overstatement is not None and policy.overstatement > 50
    # 44 rows clears the estimability floor, so the *gate* passes -- and the report still
    # has to say the answer stands on a ninetieth of the table. The disclosure is the
    # point; the threshold is a convention and must not be what carries it.
    assert policy.holds
    assert "91x the reported count" in estimated.summary()


def test_a_policy_matching_the_data_keeps_almost_every_row() -> None:
    """The gate must not fire on the ordinary case, or it is noise rather than a check."""
    data = _skewed_data(rare=2000, common=2000)
    estimated = _estimate(data, {(0,): 0.5, (1,): 0.5})

    (policy,) = estimated.policy
    assert policy.effective_n == pytest.approx(4000.0)
    assert policy.holds


def test_the_effective_count_is_the_inverse_variance_sum_and_not_a_row_tally() -> None:
    """Pinned against hand arithmetic: a constant, a total, or a minimum all differ."""
    data = _skewed_data(rare=100, common=900)
    estimated = _estimate(data, {(0,): 0.25, (1,): 0.75})

    (policy,) = estimated.policy
    expected = 1.0 / (0.25**2 / 900 + 0.75**2 / 100)
    assert policy.effective_n == pytest.approx(expected)
    assert expected == pytest.approx(176.1, abs=0.5)
    # The three quantities it must not be confused with.
    assert policy.effective_n != pytest.approx(float(data.n_rows))
    assert policy.effective_n != pytest.approx(100.0)
    assert policy.rows == {(0,): 900, (1,): 100}


def test_a_policy_below_the_estimability_floor_fails_the_gate() -> None:
    """Below the floor the cell is not merely thin, it is not estimable."""
    data = _skewed_data(rare=12, common=4000)
    estimated = _estimate(data, {(0,): 0.05, (1,): 0.95})

    (policy,) = estimated.policy
    assert policy.effective_n == pytest.approx(13.3, abs=0.5)
    assert policy.effective_n < policy.floor
    assert not policy.holds
    assert "Policy support: FAIL" in estimated.summary()


def test_a_policy_failure_does_not_make_positivity_read_fail() -> None:
    """Two different findings. Conflating them was the defect PR #14 closed for the backend.

    Every conditioning cell here is populated, so positivity genuinely holds; only the
    policy's leverage is bad. A report that flipped positivity to FAIL would be describing
    a stratum that is not empty.
    """
    data = _skewed_data(rare=12, common=4000)
    estimated = _estimate(data, {(0,): 0.05, (1,): 0.95})

    (policy,) = estimated.policy
    assert not policy.holds
    assert estimated.support.holds
    # And the two count different things, which is the reason they are separate fields:
    # positivity reports the thinnest cell the evaluator *touched* -- here the numerator
    # {T=1, Y=0}, 6 of the 12 rare rows -- while the policy weighs the 12 rows at T=1 that
    # its mass actually redistributes onto.
    assert estimated.support.min_stratum_count == 6
    assert estimated.support.thinnest_stratum == {"T": 1, "Y": 0}
    assert policy.rows[(1,)] == 12


def test_the_report_says_which_level_the_policy_leans_on() -> None:
    """A verdict without the offending level cannot be acted on."""
    data = _skewed_data(rare=12, common=4000)
    estimated = _estimate(data, {(0,): 0.05, (1,): 0.95})

    text = estimated.summary()
    assert "Policy support: FAIL" in text
    assert "T=1" in text
    assert "0.95" in text
    assert "4012" in text  # the header count it is contradicting


def test_an_estimand_with_no_policy_reports_no_policy_certificate() -> None:
    """`replace` and plain marginals install no P0, so there is nothing to weigh."""
    graph = _chain_graph()
    result = identify(graph, DeleteMechanism("m_T", outcomes=("T",)))
    data = _skewed_data(rare=100, common=900)
    estimated = estimate(result, data, fallbacks={"m_T": {(0,): 0.5, (1,): 0.5}})

    # The estimand is the policy itself; it reads no conditional kernel from the data, so
    # there is no data-backed leverage to weigh. `None` here is not "few rows" -- it is
    # "the question does not apply", and a number in its place would be a warning about
    # nothing. Pinned explicitly: asserting only that a line is printed leaves this branch
    # unguarded, which a mutation dropping the leverage test survived.
    (policy,) = estimated.policy
    assert policy.mechanism == "m_T"
    assert policy.effective_n is None
    assert policy.overstatement is None
    assert policy.holds
    assert "carries no data-backed leverage to weigh" in estimated.summary()
