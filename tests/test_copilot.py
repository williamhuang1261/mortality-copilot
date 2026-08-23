"""Tests for the copilot's grounding and rendering.

None of these need faiss, torch or an LLM: retrieval is injected, so CI can run
them with only the core requirements installed. That is deliberate -- the parts
worth testing are the parts that decide what the note *says*.
"""

from __future__ import annotations

import io
import re

import pytest
from rich.console import Console

from pipeline.copilot import (
    CAVEAT_QUERY,
    PROMPT_PATH,
    Retrieved,
    _caveat_sentence,
    _first_sentence,
    build_prompt,
    extractive_note,
    query_from_case,
)

CASE = {
    "case_id": "case_007",
    "predicted_risk_36mo": 0.1234,
    "risk_decile": 9,
    "prediction_is_out_of_fold": True,
    "observed": {"died_within_36_months": False, "followup_months": 55},
    "features": {
        "age": 71, "sex": "male", "bmi": 31.2, "sbp": 158.0, "dbp": 82.0,
        "smoker": "current", "diabetes": "yes", "prior_chd": "no",
        "prior_cancer": "no", "hdl": 41.0, "hba1c": 7.4,
        "income_ratio": 1.15, "income_ratio_imputed": False,
    },
    "top_drivers": [
        {"variable": "age", "label": "Age", "value": "71",
         "statement": "Age = 71", "contribution_log_odds": 1.42,
         "direction": "increases risk", "relative_to": "cohort mean"},
        {"variable": "smoker", "label": "Smoking status", "value": "current",
         "statement": "Smoking status = current", "contribution_log_odds": 0.41,
         "direction": "increases risk", "relative_to": "cohort mean"},
        {"variable": "hdl", "label": "HDL cholesterol", "value": "41.0",
         "statement": "HDL cholesterol = 41.0", "contribution_log_odds": -0.22,
         "direction": "decreases risk", "relative_to": "cohort mean"},
    ],
}

CAVEAT = Retrieved(
    text=("Public-use Linked Mortality Files Updated May 2022. INTRODUCTION The "
          "National Center for Health Statistics has linked survey data with death "
          "certificate records. Due to the probabilistic nature of the linkage, "
          "those that linked to the NDI are assumed deceased and those that did "
          "not are assumed alive."),
    source="linked-mortality-file-description.pdf", page=1, score=0.72)

CONTEXT = Retrieved(text="Age is the strongest single predictor in this cohort. " * 4,
                    source="model_card.json", page=None, score=0.51)


def test_citation_formats_with_and_without_a_page():
    assert CAVEAT.citation == "[source: linked-mortality-file-description.pdf, page 1]"
    assert CONTEXT.citation == "[source: model_card.json]"


def test_note_reports_the_risk_decile_and_drivers():
    note = extractive_note(CASE, [CONTEXT], [CAVEAT])
    assert "12.3%" in note
    assert "decile 9 of 10" in note
    assert "Age = 71" in note
    assert "Smoking status = current" in note
    assert "HDL cholesterol = 41.0" in note


def test_note_separates_risk_raising_from_risk_lowering_drivers():
    note = extractive_note(CASE, [CONTEXT], [CAVEAT])
    up, down = note.split("Pushing it down:")
    assert "Age = 71" in up and "Smoking status = current" in up
    assert "HDL cholesterol = 41.0" in down


def test_note_never_states_the_prediction_as_fact():
    note = extractive_note(CASE, [CONTEXT], [CAVEAT])
    assert "The model estimates" in note
    assert "will die" not in note.lower()


def test_note_carries_a_citation_for_every_corpus_claim():
    note = extractive_note(CASE, [CONTEXT], [CAVEAT])
    assert CAVEAT.citation in note


def test_caveat_sentence_skips_boilerplate_for_the_actual_caveat():
    """Retrieval picks the passage; this picks the sentence inside it."""
    sentence = _caveat_sentence(CAVEAT.text)
    assert "assumed alive" in sentence
    assert "INTRODUCTION" not in sentence


def test_first_sentence_ignores_short_leading_fragments():
    """A naive split returned the single word "cohort" from the model card."""
    assert _first_sentence("cohort. " + "A properly long sentence about follow-up "
                           "time and censoring in this dataset.").startswith("A properly")


def test_imputed_income_is_disclosed_in_the_note():
    case = {**CASE, "features": {**CASE["features"], "income_ratio_imputed": True}}
    note = extractive_note(case, [CONTEXT], [CAVEAT])
    assert "imputed" in note and "not a measured value" in note


def test_citations_survive_rich_rendering():
    """Rich parses [...] as markup; unescaped, it deletes the citations."""
    from rich.markup import escape
    from rich.panel import Panel

    note = extractive_note(CASE, [CONTEXT], [CAVEAT])
    buffer = io.StringIO()
    Console(file=buffer, width=100, no_color=True).print(Panel(escape(note)))
    rendered = re.sub(r"\s+", " ", buffer.getvalue())
    assert "[source: linked-mortality-file-description.pdf, page 1]" in rendered


# --------------------------------------------------------------- prompting

def test_prompt_template_substitutes_every_placeholder():
    prompt = build_prompt(CASE, [CONTEXT, CAVEAT])
    assert not re.search(r"\{\{[A-Z_]+\}\}", prompt), "unsubstituted placeholder"


def test_prompt_includes_the_grounding_rules_and_the_excerpts():
    prompt = build_prompt(CASE, [CONTEXT, CAVEAT])
    assert "Use ONLY the case record and the reference excerpts" in prompt
    assert "[source: filename, page N]" in prompt
    assert CAVEAT.citation in prompt
    assert "at most 150 words" in prompt.lower()


def test_prompt_flags_imputation_to_the_model():
    case = {**CASE, "features": {**CASE["features"], "income_ratio_imputed": True}}
    assert "IMPUTED, not measured" in build_prompt(case, [CONTEXT])


def test_prompt_template_forbids_medical_and_underwriting_advice():
    # collapse whitespace so the assertions survive re-wrapping of the template
    template = re.sub(r"\s+", " ", PROMPT_PATH.read_text().lower())
    assert "do not give medical advice" in template
    assert "do not recommend an insurance decision" in template
    assert "never state a prediction as a fact about the person's future" in template
    assert "if a value is marked imputed, describe it as imputed" in template


def test_case_query_is_built_from_the_drivers():
    query = query_from_case(CASE)
    assert "Age" in query and "Smoking status" in query


def test_caveat_query_targets_limitations_not_the_case():
    assert "limitations" in CAVEAT_QUERY and "assumed alive" in CAVEAT_QUERY
