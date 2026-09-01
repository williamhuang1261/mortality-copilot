"""Agent evaluation harness: scores the deterministic dispatcher against a
fixed set of golden multi-turn conversations.

    python -m pipeline.agent_eval

Different from `tests/test_agent.py`'s unit-level assertions: this treats
each scenario as one product-level accuracy unit and reports an aggregate
score the way an agent eval framework would, not a bag of asserts. Two
things are checked per what-if turn, both against a pinned golden number
computed once from `artifacts/model_card.json`'s own fitted coefficients
(never re-derived math):

- **tool selection**: did `parse_utterance` route the turn to the tool a
  human reading the utterance would expect?
- **structured/natural-language consistency**: does `dispatch`'s reply
  string surface the exact same rounded risk percentage that calling
  `tools.what_if` directly with the same arguments produces? Existing
  tests only check that old/new numbers appear as substrings; this closes
  the gap where `format_what_if` could silently round or drift.

Exits non-zero if accuracy is not 100%, so it can gate CI like the rest of
this repo's golden-run tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.agent import Session, dispatch, parse_utterance
from pipeline.tools import ToolError, what_if

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "artifacts" / "cases.json"
MODEL_CARD_PATH = ROOT / "artifacts" / "model_card.json"
REPORT_PATH = ROOT / "artifacts" / "agent_eval_report.json"


@dataclass
class Turn:
    utterance: str
    # None means "this turn must raise a ToolError".
    expect_tool: str | None = None
    expect_error_substring: str | None = None
    # Pinned golden value for what_if turns, computed once from real
    # artifacts (see docs/agent_eval.md for the derivation).
    expect_risk_delta_pct_points: float | None = None
    tolerance: float = 1e-4


@dataclass
class Scenario:
    name: str
    turns: list[Turn]


SCENARIOS: list[Scenario] = [
    Scenario("explicit_case_then_absolute_what_if", turns=[
        Turn("case 1", expect_tool="lookup_case_by_number"),
        Turn("what if age were 80", expect_tool="what_if",
             expect_risk_delta_pct_points=6.947684855695788),
        Turn("what's the AUC?", expect_tool="query_model_card"),
    ]),
    Scenario("percentage_what_if_and_limitations_query", turns=[
        Turn("case 10", expect_tool="lookup_case_by_number"),
        Turn("raise hba1c by 20%", expect_tool="what_if",
             expect_risk_delta_pct_points=0.0006442740294722825),
        Turn("what are the limitations?", expect_tool="query_model_card"),
    ]),
    Scenario("categorical_what_if_and_cohort_query", turns=[
        Turn("case 5", expect_tool="lookup_case_by_number"),
        Turn("what if smoker is current", expect_tool="what_if",
             expect_risk_delta_pct_points=0.293374156625003),
        Turn("how many people are in the cohort?", expect_tool="query_model_card"),
    ]),
    Scenario("case_switch_mid_conversation_uses_latest_context", turns=[
        Turn("case 1", expect_tool="lookup_case_by_number"),
        Turn("case 2", expect_tool="lookup_case_by_number"),
        Turn("what if age were 90", expect_tool="what_if"),
    ]),
    Scenario("what_if_without_case_in_context_fails_closed", turns=[
        Turn("what if age were 80",
             expect_error_substring="No case in context"),
    ]),
    Scenario("unparseable_change_direction_fails_closed", turns=[
        Turn("case 1", expect_tool="lookup_case_by_number"),
        Turn("what if age changes", expect_error_substring="Say how"),
    ]),
]


@dataclass
class TurnResult:
    utterance: str
    passed: bool
    detail: str


@dataclass
class ScenarioResult:
    name: str
    turns: list[TurnResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(turn.passed for turn in self.turns)


def _load_artifacts() -> tuple[list[dict], dict]:
    cases = json.loads(CASES_PATH.read_text())["cases"]
    model_card = json.loads(MODEL_CARD_PATH.read_text())
    return cases, model_card


def run_scenario(scenario: Scenario, cases: list[dict],
                  model_card: dict) -> ScenarioResult:
    session = Session(cases=cases, model_card=model_card)
    result = ScenarioResult(scenario.name)

    for turn in scenario.turns:
        try:
            intent = parse_utterance(session, turn.utterance)
        except ToolError as exc:
            if (turn.expect_tool is None and turn.expect_error_substring
                    and turn.expect_error_substring in str(exc)):
                result.turns.append(TurnResult(
                    turn.utterance, True, f"raised expected error: {exc}"))
            else:
                result.turns.append(TurnResult(
                    turn.utterance, False, f"unexpected error: {exc}"))
            continue

        if turn.expect_tool is None:
            result.turns.append(TurnResult(
                turn.utterance, False,
                f"expected an error, got tool {intent.tool!r}"))
            continue

        if intent.tool != turn.expect_tool:
            result.turns.append(TurnResult(
                turn.utterance, False,
                f"expected tool {turn.expect_tool!r}, got {intent.tool!r}"))
            continue

        detail = f"tool={intent.tool!r}"
        passed = True

        if turn.expect_risk_delta_pct_points is not None:
            direct_result = what_if(cases, model_card, **intent.kwargs)
            delta = direct_result.risk_delta_pct_points
            if abs(delta - turn.expect_risk_delta_pct_points) >= turn.tolerance:
                result.turns.append(TurnResult(
                    turn.utterance, False,
                    f"expected risk delta {turn.expect_risk_delta_pct_points:.6f} "
                    f"pts, got {delta:.6f}"))
                continue
            detail += f", risk_delta={delta:.6f}pts"

            reply = dispatch(session, turn.utterance)
            expected_new_risk_str = f"{direct_result.new_risk:.2%}"
            if expected_new_risk_str not in reply:
                result.turns.append(TurnResult(
                    turn.utterance, False,
                    f"reply did not surface {expected_new_risk_str!r}: {reply!r}"))
                continue
        else:
            dispatch(session, turn.utterance)

        result.turns.append(TurnResult(turn.utterance, passed, detail))

    return result


def evaluate() -> dict[str, Any]:
    cases, model_card = _load_artifacts()
    results = [run_scenario(scenario, cases, model_card)
               for scenario in SCENARIOS]
    turns_total = sum(len(r.turns) for r in results)
    turns_passed = sum(turn.passed for r in results for turn in r.turns)
    return {
        "scenarios_total": len(results),
        "scenarios_passed": sum(r.passed for r in results),
        "turns_total": turns_total,
        "turns_passed": turns_passed,
        "accuracy": turns_passed / turns_total,
        "scenarios": [
            {
                "name": r.name,
                "passed": r.passed,
                "turns": [
                    {"utterance": t.utterance, "passed": t.passed,
                     "detail": t.detail}
                    for t in r.turns
                ],
            }
            for r in results
        ],
    }


def main() -> None:
    report = evaluate()
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"agent eval: {report['scenarios_passed']}/{report['scenarios_total']} "
          f"scenarios, {report['turns_passed']}/{report['turns_total']} turns, "
          f"accuracy {report['accuracy']:.1%}")
    for scenario in report["scenarios"]:
        status = "PASS" if scenario["passed"] else "FAIL"
        print(f"  [{status}] {scenario['name']}")
        for turn in scenario["turns"]:
            mark = "ok" if turn["passed"] else "FAIL"
            print(f"      {mark}: {turn['utterance']!r} -- {turn['detail']}")
    if report["accuracy"] < 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
