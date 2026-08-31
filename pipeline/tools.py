"""Tools the agent CLI (`pipeline/agent.py`) can call.

Every tool here is a pure function over artifacts the R pipeline already
exported (`artifacts/cases.json`, `artifacts/model_card.json`). None of them
fit a new model or touch a pinned number from `R/04_models.R` -- they read
and, for `what_if`, do exact linear arithmetic against the already-fitted
GLM's own coefficients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ------------------------------------------------------------- case lookup


class ToolError(Exception):
    """Raised when a tool cannot satisfy a request -- caught by the agent
    and turned into a plain-English message, never a stack trace."""


def lookup_case(cases: list[dict], case_id: str) -> dict:
    """Return the full case record for `case_id` (e.g. "case_017")."""
    for case in cases:
        if case["case_id"] == case_id:
            return case
    raise ToolError(f"No case with id {case_id!r}. Case ids look like "
                     f"'case_001' through 'case_{len(cases):03d}'.")


def lookup_case_by_number(cases: list[dict], number: int) -> dict:
    """1-based positional lookup, matching `copilot.py --case N`."""
    if not 1 <= number <= len(cases):
        raise ToolError(f"--case must be between 1 and {len(cases)}.")
    return cases[number - 1]


# --------------------------------------------------------- model-card query

SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "purpose": ("purpose", "what is this", "underwriting", "disclaimer",
                "is this real"),
    "outcome": ("outcome", "endpoint", "36 month", "36-month", "definition",
                "why 36"),
    "cohort": ("cohort", "sample size", "population", "how many", "deaths",
               "exclusion", "excluded"),
    "predictors": ("predictor", "feature", "variable", "input", "used",
                   "dropped"),
    "validation": ("validation", "auc", "brier", "calibration",
                   "cross-validation", "cross validation", "performance",
                   "accuracy", "how good"),
    "limitations": ("limitation", "caveat", "weight", "bias",
                     "not representative", "should not", "cannot"),
    "coefficients": ("coefficient", "log-odds", "log odds", "odds ratio",
                      "estimate"),
    "provenance": ("provenance", "source", "where does the data come from",
                    "data come from"),
}


def query_model_card(model_card: dict, question: str) -> dict[str, Any]:
    """Return the model-card section(s) matching `question`'s keywords.

    Answers are the model card's own JSON, verbatim -- never an LLM
    paraphrase of the numbers, so the answer can never drift from the pinned
    validation metrics.
    """
    lowered = question.lower()
    matched = [section for section, keywords in SECTION_KEYWORDS.items()
               if any(keyword in lowered for keyword in keywords)]
    if not matched:
        raise ToolError(
            "No model-card section matches that question. Try asking about "
            "the cohort, validation metrics, predictors, limitations or "
            "coefficients.")
    return {section: model_card[section] for section in matched
            if section in model_card}


# ----------------------------------------------------------------- what-if

# Continuous GLM terms shared between a case's `features` dict and
# `model_card["coefficients"]["logistic_glm"]`. The marginal effect of moving
# one of these by `delta`, holding everything else fixed, is exactly
# `coefficient * delta` in log-odds -- true for any GLM linear predictor,
# independent of which other terms the per-case export happens to carry.
CONTINUOUS_TERMS = ("age", "bmi", "sbp", "dbp", "hdl", "hba1c", "income_ratio")

# Categorical terms with a case-level counterpart. `None` marks the reference
# level (no dummy, contributes 0). Term names match R's `model.matrix` dummy
# naming (`<variable><level>`, no separator), taken verbatim from
# `model_card["coefficients"]["logistic_glm"]`.
CATEGORICAL_TERMS: dict[str, dict[str, str | None]] = {
    "sex": {"female": None, "male": "sexmale"},
    "smoker": {"never": None, "former": "smokerformer",
               "current": "smokercurrent"},
    "diabetes": {"no": None, "borderline": "diabetesborderline",
                 "yes": "diabetesyes"},
    "prior_chd": {"no": None, "yes": "prior_chdyes"},
    "prior_cancer": {"no": None, "yes": "prior_canceryes"},
}

# GLM terms the fitted model uses but the per-case export omits. `what_if`
# refuses these rather than silently ignoring them.
OUT_OF_SCOPE_FEATURES = ("race_eth", "education", "income_missing")


def _coefficient_map(model_card: dict) -> dict[str, float]:
    return {row["term"]: row["estimate_log_odds"]
            for row in model_card["coefficients"]["logistic_glm"]}


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class WhatIfResult:
    case_id: str
    feature: str
    old_value: Any
    new_value: Any
    base_risk: float
    new_risk: float
    log_odds_delta: float

    @property
    def risk_delta_pct_points(self) -> float:
        return (self.new_risk - self.base_risk) * 100


def what_if(cases: list[dict], model_card: dict, case_id: str, feature: str,
            new_value: Any) -> WhatIfResult:
    """Recompute a case's predicted risk with one feature changed.

    Exact for the seven continuous and five categorical terms this function
    knows about (see `CONTINUOUS_TERMS`/`CATEGORICAL_TERMS`): both are a
    real re-application of the already-fitted GLM's own linear coefficients,
    not a new model and not an approximation.
    """
    if feature in OUT_OF_SCOPE_FEATURES:
        raise ToolError(
            f"{feature!r} is used by the fitted model but is not exported "
            f"per-case in artifacts/cases.json, so what_if() cannot vary it "
            f"without re-deriving the other terms. Scope: "
            f"{', '.join(CONTINUOUS_TERMS + tuple(CATEGORICAL_TERMS))}.")

    case = lookup_case(cases, case_id)
    coefficients = _coefficient_map(model_card)
    base_risk = case["predicted_risk_36mo"]
    base_logit = _logit(base_risk)

    if feature in CONTINUOUS_TERMS:
        old_value = case["features"][feature]
        coefficient = coefficients[feature]
        delta_logit = coefficient * (new_value - old_value)
    elif feature in CATEGORICAL_TERMS:
        old_value = case["features"][feature]
        levels = CATEGORICAL_TERMS[feature]
        if new_value not in levels:
            raise ToolError(
                f"{feature!r} must be one of {sorted(levels)}, got "
                f"{new_value!r}.")
        old_term = levels.get(old_value)
        new_term = levels[new_value]
        delta_logit = (coefficients.get(new_term, 0.0)
                        - coefficients.get(old_term, 0.0))
    else:
        raise ToolError(
            f"Unknown feature {feature!r}. Scope: "
            f"{', '.join(CONTINUOUS_TERMS + tuple(CATEGORICAL_TERMS))}.")

    new_logit = base_logit + delta_logit
    new_risk = _sigmoid(new_logit)
    return WhatIfResult(case_id=case["case_id"], feature=feature,
                         old_value=old_value, new_value=new_value,
                         base_risk=base_risk, new_risk=new_risk,
                         log_odds_delta=delta_logit)
