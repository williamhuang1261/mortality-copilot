"""Guards on the README.

A README is the most-read and least-tested file in a repository, and it is the
one that quotes numbers from four other files. These tests check that what it
claims still matches what the pipeline produced, and that no placeholder text
survived.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
CARD = ROOT / "artifacts" / "model_card.json"
RETRIEVAL = ROOT / "docs" / "retrieval_eval.md"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.mark.parametrize("heading", [
    "## What it does", "## Architecture", "## Quickstart", "## Data",
    "## Cohort", "## Methodology", "## Results", "## Retrieval",
    "## Agentic tool-use mode", "## Engineering notes", "## Limitations",
    "## Licence",
])
def test_readme_has_section(readme, heading):
    assert heading in readme


def test_disclaimer_appears_before_anything_else(readme):
    """A reader must hit the disclaimer before the results."""
    disclaimer = readme.index("not an underwriting system")
    assert disclaimer < readme.index("## Results")
    assert disclaimer < len(readme) * 0.15, "disclaimer is buried too far down"


def test_no_placeholder_text_survived(readme):
    for token in ("TODO", "TBD", "XXX", "FIXME", "lorem ipsum", "<insert", "0.XX"):
        assert token.lower() not in readme.lower(), f"placeholder {token!r} in README"


def test_every_local_link_and_image_resolves(readme):
    targets = re.findall(r"\]\((?!https?://)([^)#]+)\)", readme)
    assert targets, "expected some local links"
    for target in targets:
        assert (ROOT / target).exists(), f"broken local link: {target}"


@pytest.mark.skipif(not CARD.exists(), reason="model card not generated")
def test_results_table_matches_the_model_card(readme):
    """The README must not drift from the artifact it is quoting."""
    card = json.loads(CARD.read_text())
    section = readme.split("## Results", 1)[1].split("##", 1)[0]
    for row in card["validation"]["metrics"]:
        assert row["AUC"] in section, (
            f"{row['Model']} AUC {row['AUC']} is in the model card but not the README"
        )
    c_index = str(card["validation"]["concordance_cox"]["c_index"])
    assert c_index in section, f"Cox C-index {c_index} missing from the README"


@pytest.mark.skipif(not RETRIEVAL.exists(), reason="retrieval eval not generated")
def test_retrieval_numbers_match_the_evaluation_report(readme):
    report = RETRIEVAL.read_text()
    numbers = re.findall(r"\|\s*recall@5\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", report)
    assert numbers, "could not parse recall@5 from docs/retrieval_eval.md"
    base, tuned = numbers[0]
    section = readme.split("## Retrieval", 1)[1]
    assert base in section and tuned in section


def test_readme_states_the_survey_weight_limitation(readme):
    limitations = readme.split("## Limitations", 1)[1]
    assert "Survey weights are not applied" in limitations
    assert "not** the U.S." in limitations or "not the U.S." in limitations


def test_readme_admits_the_unflattering_results(readme):
    """Reporting only the wins would misrepresent the work."""
    assert "random forest loses" in readme.lower()
    assert "does not earn its place" in readme
    assert "not adopted" in readme


def test_readme_does_not_claim_to_have_fine_tuned_an_llm(readme):
    section = readme.split("## Retrieval", 1)[1]
    assert "not instruction-tuning" in section
    assert "bi-encoder" in section


def test_readme_states_the_what_if_scope_limitation(readme):
    section = readme.split("## Agentic tool-use mode", 1)[1].split("##", 1)[0]
    assert "race_eth" in section and "education" in section
    assert "not silently ignored" in section or "silently ignored" in section
