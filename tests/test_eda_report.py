"""Checks on the committed EDA report.

CI has no R, so these validate the generated artefact rather than re-running
the analysis. The arithmetic checks are not busywork: the first version of
R/03_eda.R reported the mean difference with the sign flipped relative to the
means and the confidence interval beside it, which is invisible unless
something cross-checks the columns against each other.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EDA = ROOT / "docs" / "eda.md"


@pytest.fixture(scope="module")
def report() -> str:
    if not EDA.exists():
        pytest.skip("docs/eda.md not generated yet (run `make eda`)")
    return EDA.read_text(encoding="utf-8")


def table_rows(report: str, heading: str) -> list[list[str]]:
    """Return the data rows of the first markdown table under a heading."""
    section = report.split(heading, 1)[1]
    rows = []
    for line in section.splitlines():
        line = line.strip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("- "):
                continue
            rows.append(cells)
        elif rows:
            break
    return rows[1:]  # drop the header


def num(text: str) -> float:
    return float(text.replace("+", ""))


def ci_bounds(text: str) -> tuple[float, float]:
    lo, hi = re.findall(r"[-+]?\d*\.?\d+", text)[:2]
    return float(lo), float(hi)


@pytest.mark.parametrize("heading", [
    "## Cohort",
    "## Categorical predictors versus mortality",
    "## Continuous predictors versus mortality",
    "## Bootstrap confidence intervals",
    "## Survival curves and log-rank tests",
    "## Caveats",
])
def test_report_contains_every_section(report, heading):
    assert heading in report


def test_every_referenced_figure_exists(report):
    figures = set(re.findall(r"\]\((figs/[^)]+)\)", report))
    assert figures, "the report should reference at least one figure"
    for relative in figures:
        assert (ROOT / "docs" / relative).exists(), f"missing figure: {relative}"


def test_mean_difference_matches_the_two_means(report):
    """Difference must equal died - survived, to rounding."""
    for row in table_rows(report, "## Continuous predictors versus mortality"):
        predictor, survived, died, difference = row[0], num(row[1]), num(row[2]), num(row[3])
        assert difference == pytest.approx(died - survived, abs=0.02), (
            f"{predictor}: reported {difference:+} but means give {died - survived:+.2f}"
        )


def test_mean_difference_sign_agrees_with_its_confidence_interval(report):
    for row in table_rows(report, "## Continuous predictors versus mortality"):
        predictor, difference = row[0], num(row[3])
        lo, hi = ci_bounds(row[4])
        assert lo <= difference <= hi, (
            f"{predictor}: difference {difference:+} lies outside its CI [{lo:+}, {hi:+}]"
        )


def test_bootstrap_agrees_with_the_t_test(report):
    """Two independent estimates of the same quantity must not disagree."""
    welch = {r[0]: num(r[3]) for r in
             table_rows(report, "## Continuous predictors versus mortality")}
    for row in table_rows(report, "## Bootstrap confidence intervals"):
        predictor, observed = row[0], num(row[1])
        assert predictor in welch
        assert observed == pytest.approx(welch[predictor], abs=0.02), (
            f"{predictor}: bootstrap says {observed:+}, Welch says {welch[predictor]:+}"
        )


def test_bootstrap_ci_contains_its_own_point_estimate(report):
    for row in table_rows(report, "## Bootstrap confidence intervals"):
        observed = num(row[1])
        lo, hi = ci_bounds(row[3])
        assert lo <= observed <= hi, f"{row[0]}: {observed:+} outside [{lo:+}, {hi:+}]"


def test_survey_weights_caveat_is_present(report):
    """Omitting this would misrepresent what the numbers mean."""
    assert "Survey weights are not applied" in report
    assert "not the U.S. population" in report


def test_smoking_confounding_is_explained(report):
    """The crude smoking result inverts; an unexplained inversion reads as a bug."""
    assert "confounded by age" in report
