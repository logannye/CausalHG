"""The whitepaper's claims are gated, for the reason its README claims already are.

Two failures this pins, both found by reading the paper against the code rather than by
any test:

**"A strict generalization of Pearl's structural causal models"** stood as the opening
sentence of the abstract *and* of the conclusion, while §1.2 two paragraphs later said the
contrary — "the contribution is not new expressive power" — and `README.md` said "It is not
more expressive." The sentence a referee reads first contradicted the repo's own position,
and no test could tell.

**The nearest prior art went uncited.** Shpitser & Tchetgen Tchetgen (2016) place node
interventions inside *edge* interventions and give both the edge g-formula and the
recanting witness; that is mechanism-level identification theory a decade old. Ma et al.
(2022) do causal inference on hypergraphs, which makes the flat claim that hypergraph work
is "descriptive" false as written. A copy of the latter sat untracked in the repository
root while the abstract claimed the territory was empty.

These are cheap to check and expensive to miss, so they are checked.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WHITEPAPER = Path("whitepaper.md").read_text()
REFERENCES = WHITEPAPER.split("## References")[1].split("\n---")[0]
BODY = WHITEPAPER.split("## References")[0]


def test_the_paper_does_not_claim_to_be_a_strict_generalization() -> None:
    """It may say the phrase, but only to deny it -- never as an assertion."""
    for match in re.finditer(r"[^.]*strict generaliz[^.]*\.", WHITEPAPER):
        sentence = match.group(0)
        assert "not a strict generaliz" in sentence or "**not** a strict generaliz" in sentence, (
            f"asserts strict generalization: {sentence.strip()[:160]}"
        )


def test_the_abstract_and_the_conclusion_agree_with_section_1_2() -> None:
    """The three places the positioning is stated must not contradict each other."""
    abstract = BODY.split("## Abstract")[1].split("## 1. Introduction")[0]
    conclusion = BODY.split("## 10. Conclusion")[1]

    assert "not new expressive power" in BODY  # §1.2, the honest version, still present
    for section, name in ((abstract, "abstract"), (conclusion, "conclusion")):
        assert "reformulation" in section, name
        # Checked per sentence, not by stripping a literal: the abstract writes the denial
        # as `**not** a strict generalization`, so a substring subtraction misses the
        # markdown emphasis and reports a claim that is not being made.
        for match in re.finditer(r"[^.]*strict generaliz[^.]*\.", section):
            assert "not" in match.group(0).split("strict generaliz")[0][-24:], (
                f"{name} asserts strict generalization: {match.group(0).strip()[:120]}"
            )


@pytest.mark.parametrize(
    ("surname", "year", "why"),
    [
        ("Shpitser, I., & Tchetgen Tchetgen", "2016", "edge interventions; the nearest prior art"),
        ("Ma, J.", "2022", "causal effects on hypergraphs; refutes the 'descriptive' claim"),
        ("Yao, J., & Evans", "2022", "edge interventions in linear SEMs"),
        ("Correa, J., & Bareinboim", "2020", "sigma-calculus; our intervention space is a subset"),
    ],
)
def test_the_nearest_prior_art_is_cited(surname: str, year: str, why: str) -> None:
    assert f"{surname}" in REFERENCES, f"missing reference ({why})"
    entry = [line for line in REFERENCES.split("\n") if surname in line]
    assert entry and year in entry[0], f"wrong year for {surname} ({why})"


def test_the_paper_distinguishes_rather_than_merely_citing_the_hypergraph_causal_work() -> None:
    """A citation that does not say how the work differs is not a distinction.

    Ma et al.'s hyperedges carry interference *between units*; ours are the mechanisms
    *within* one. Naming that is the difference between citing prior art and answering it.
    """
    assert "Ma et al. (2022)" in BODY
    assert "interference" in BODY
    assert "units" in BODY


def test_mechanism_level_intervention_is_not_claimed_as_new() -> None:
    """The honest version of the contribution, asserted where a referee will look."""
    assert "We do not claim priority on intervening at the level of a mechanism." in BODY


def test_every_in_text_citation_has_a_reference_entry() -> None:
    """Two dangling citations were found this way: Eberhardt & Scheines, and Tian & Pearl 2001."""
    listed = {
        line.split(",")[0].strip()
        for line in REFERENCES.split("\n")
        if re.match(r"^[A-Z][a-z]", line)
    }
    citation = re.compile(
        r"\b([A-Z][A-Za-z\-]+)(?:\s*&\s*[A-Z][A-Za-z\-]+| et al\.)?\s*\(\d{4}\)"
    )
    cited = {match.group(1) for match in citation.finditer(BODY)}
    # Multi-author forms put a middle author's surname first in the entry, so a cited
    # surname counts as present if it appears anywhere in the reference list.
    missing = sorted(name for name in cited if name not in listed and name not in REFERENCES)
    assert not missing, f"cited but not in the reference list: {missing}"


def test_the_reference_list_is_alphabetical() -> None:
    """Cheap, and the two insertions made while adding prior art both broke it."""
    names = [
        line.split("(")[0].strip()
        for line in REFERENCES.split("\n")
        if re.match(r"^[A-Z][a-z]+,", line)
    ]
    assert names == sorted(names, key=str.lower)
    assert len(names) >= 26
