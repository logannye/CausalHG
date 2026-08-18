"""The demo's numbers are claims, so they are pinned like every other claim here.

`test_readme_smoke.py` exists because three README statements were wrong rather than stale.
A demo is worse: it is the artefact someone runs to decide whether to believe the library,
and its figures are quoted in `demo/README.md`. Every number quoted there is asserted below
against a real run, so the two cannot drift apart silently.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo" / "preflight.py"


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the demo end to end, as a user would, and return its verdicts.

    A subprocess rather than an import: the demo manipulates `sys.path` so it can be run
    from a checkout, and running it the way it is documented is the only way to catch a
    demo that no longer starts.
    """
    out = tmp_path_factory.mktemp("demo") / "report.json"
    completed = subprocess.run(
        [sys.executable, str(DEMO), "--json", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(out.read_text())


def test_the_demo_reads_the_screen_the_readme_describes(report: dict) -> None:
    assert report["cells"] == 65_359
    assert report["control_cells"] == 11_183
    assert report["arms"] == 105
    # The unit of independence, and the reason the interval is not built from 65,359 rows.
    assert report["gemgroups"] == 8


def test_every_verdict_carries_evidence_independent_of_the_compiler(report: dict) -> None:
    """A verdict with no evidence beside it is the thing this demo exists not to be."""
    names = [entry["name"] for entry in report["verdicts"]]
    assert names == [
        "cycles",
        "post_treatment",
        "dead_readout",
        "degenerate_binning",
        "policy_leverage",
    ]
    for entry in report["verdicts"]:
        assert entry["evidence"].strip(), entry["name"]
        assert entry["verdict"].strip(), entry["name"]


def test_the_curated_pathway_answers_no_non_trivial_query(report: dict) -> None:
    """The headline finding. If this ever changes, the demo's whole framing changes."""
    (cycles,) = [e for e in report["verdicts"] if e["name"] == "cycles"]
    assert "89 refused" in cycles["verdict"]
    assert "6 identified" in cycles["verdict"]
    assert "0 are non-trivial" in cycles["verdict"]


def test_unrolling_in_time_is_what_restores_the_answers(report: dict) -> None:
    """The constructive half, pinned row by row -- a table that drifts is folklore."""
    rows = {row["timepoints"]: row for row in report["unrolled"]}
    assert rows[2]["nontrivial"] == 0
    assert rows[3]["nontrivial"] == 35
    assert rows[4]["nontrivial"] == 69
    # Every one inside the elimination budget, which is the half of the claim that makes
    # the other half useful: unrolling would buy nothing if it bought intractability.
    assert rows[3]["max_width"] == 8
    assert rows[4]["max_width"] == 12
    assert all(row["identified"] == 100 for row in report["unrolled"])


def test_the_policy_certificate_fires_on_the_real_arm(report: dict) -> None:
    """The guard that did not exist until this data was run through the library."""
    policy = report["policy"]
    assert policy["n_rows"] == 11_183
    assert policy["effective_n"] == pytest.approx(3733.5, abs=1.0)
    assert policy["on_target_fold"] == pytest.approx(24.8, abs=0.2)
    assert "Policy support: PASS" in policy["summary"]
    assert "3x the reported count" in policy["summary"]
