"""Pins the agent eval harness's golden accuracy against the real, committed
artifacts (not a fixture) -- if a future change to the coefficients, cases,
or the parser regresses tool selection or the risk-delta math, this fails.
"""

from __future__ import annotations

from pipeline.agent_eval import SCENARIOS, evaluate


def test_golden_scenario_and_turn_counts():
    assert len(SCENARIOS) == 6
    assert sum(len(scenario.turns) for scenario in SCENARIOS) == 15


def test_agent_eval_is_fully_accurate_on_committed_artifacts():
    report = evaluate()
    assert report["scenarios_total"] == 6
    assert report["turns_total"] == 15
    assert report["scenarios_passed"] == 6
    assert report["turns_passed"] == 15
    assert report["accuracy"] == 1.0
