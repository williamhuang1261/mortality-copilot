"""Structural tests for R/eq03_models.R.

This test suite runs with no network and no R (see tests/README convention:
CI never executes the R scripts), so it cannot check the fitted numbers
directly -- that verification lives in this extension's plan.md, pasted from
a real run. What it CAN check without R is the two things most likely to
regress silently: that cross-validation is grouped by engine (not by row,
which would leak one engine's trajectory across folds), and that the
mortality pipeline's own R/04_models.R is untouched by this extension, so its
committed metrics cannot have moved.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EQ_SOURCE = (ROOT / "R" / "eq03_models.R").read_text(encoding="utf-8")
MORTALITY_SOURCE = (ROOT / "R" / "04_models.R").read_text(encoding="utf-8")


def test_folds_are_assigned_at_the_engine_level_not_the_row_level():
    assert "make_grouped_folds" in EQ_SOURCE
    assert "fold_by_unit" in EQ_SOURCE
    # The row-level fold vector must be a lookup into the unit-level one.
    assert re.search(r"fold\s*<-\s*fold_by_unit\[as\.character\(train_df\$unit\)\]", EQ_SOURCE)


def test_a_leakage_guard_asserts_no_engine_spans_two_folds():
    assert "stopifnot(" in EQ_SOURCE
    assert "an engine's snapshots must never span two folds" in EQ_SOURCE


def test_cox_uses_an_explicit_status_column_not_an_inline_rep_call():
    """rep(1, nrow(train)) inside a formula string only works by relying on
    R's lazy environment lookup finding the loop's *current* `train` binding
    -- fragile and easy to break. An explicit status column removes the
    ambiguity."""
    assert "train_df$status <- 1L" in EQ_SOURCE
    assert "test_df$status  <- 1L" in EQ_SOURCE
    # The dangerous pattern this replaced: an inline rep() inside a formula
    # string, which only works via R's lazy environment lookup finding the
    # loop's current `train`/`train_df` binding. Only the explanatory
    # comment (quoting the old code) may still mention it.
    code_lines = [line for line in EQ_SOURCE.splitlines() if not line.strip().startswith("#")]
    assert not any("rep(1, nrow(" in line for line in code_lines)


def test_external_holdout_is_scored_after_the_full_data_fit_only():
    """test_df must never appear inside the CV loop -- it is held out
    entirely from fold assignment and from the full-data fit."""
    loop_match = re.search(
        r"for \(k in seq_len\(N_FOLDS\)\) \{(.*?)\n\}", EQ_SOURCE, re.S
    )
    assert loop_match, "could not find the CV loop body"
    assert "test_df" not in loop_match.group(1)
    assert "test_glm" in EQ_SOURCE and "test_cox" in EQ_SOURCE and "test_rf" in EQ_SOURCE


def test_evaluate_is_duplicated_not_imported_from_the_mortality_script():
    """The additive-only guarantee: R/eq03_models.R must not source()
    R/04_models.R, and must define its own evaluate()."""
    assert "source(" not in EQ_SOURCE
    assert re.search(r"evaluate <- function\(y, p, label\)", EQ_SOURCE)


def test_mortality_pipeline_is_untouched_by_this_extension():
    """A handful of literal fingerprints from R/04_models.R's committed
    behaviour. If any of these ever go missing, the mortality script was
    edited and its committed AUC/concordance numbers may have moved."""
    for fingerprint in [
        'N_FOLDS  <- 5',
        'HORIZON  <- 36',
        '`waist` is dropped',
        'saveRDS(list(',
    ]:
        assert fingerprint in MORTALITY_SOURCE, f"missing fingerprint: {fingerprint!r}"
