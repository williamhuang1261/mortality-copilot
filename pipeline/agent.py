"""A tool-calling agent over the mortality-copilot artifacts.

    python -m pipeline.agent
    python -m pipeline.agent --llm

Unlike `pipeline/copilot.py` (one case in, one note out, no state between
runs), this is a multi-turn REPL: it remembers the last case discussed, so a
follow-up like "now raise hba1c by 20%" does not need to repeat the case id.

Two dispatch paths, the same fail-closed pattern `copilot.py` already uses
for generation:

- **Deterministic dispatcher (default):** a small regex/keyword parser maps
  an utterance straight to one of the three tools in `pipeline/tools.py`.
  This is what is tested in CI -- no Ollama required.
- **LLM orchestration (`--llm`):** the tool schemas and the running
  conversation are sent to a local Ollama model via its `tools=[...]` chat
  parameter; the model's tool call is executed and its result is fed back
  for a final natural-language reply. Any failure -- no Ollama installed, a
  malformed tool call, an unknown tool name -- falls back to the
  deterministic dispatcher for that turn rather than crashing or guessing.

`what_if()` is never compounded across turns: every what-if is computed
against the original case's base risk, not the previous turn's hypothetical
one, so floating-point drift and confusing double-counting cannot creep in
across a long session. Each what-if reply says plainly which base it used.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape

from pipeline.tools import (
    CATEGORICAL_TERMS,
    CONTINUOUS_TERMS,
    ToolError,
    lookup_case,
    lookup_case_by_number,
    query_model_card,
    what_if,
)

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "artifacts" / "cases.json"
MODEL_CARD_PATH = ROOT / "artifacts" / "model_card.json"

OLLAMA_MODELS = ("llama3.2:3b", "qwen2.5:3b")
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


# ------------------------------------------------------------------ session

@dataclass
class Session:
    """Everything the agent remembers between turns of one conversation."""

    cases: list[dict]
    model_card: dict
    current_case_id: str | None = None
    turns: list[tuple[str, str]] = field(default_factory=list)  # (role, text)

    def resolve_case_id(self, explicit: str | None) -> str:
        case_id = explicit or self.current_case_id
        if case_id is None:
            raise ToolError(
                "No case in context yet. Say 'case 1' (or any case number) "
                "first.")
        return case_id

    def remember(self, role: str, text: str) -> None:
        self.turns.append((role, text))


# --------------------------------------------------------- utterance parsing

FEATURE_ALIASES: dict[str, tuple[str, ...]] = {
    "income_ratio": ("income-to-poverty ratio", "income to poverty ratio",
                      "income ratio", "income"),
    "hba1c": ("hba1c", "a1c"),
    "hdl": ("hdl cholesterol", "hdl", "cholesterol"),
    "sbp": ("systolic blood pressure", "systolic", "sbp"),
    "dbp": ("diastolic blood pressure", "diastolic", "dbp"),
    "bmi": ("body mass index", "bmi"),
    "age": ("age",),
    "smoker": ("smoking status", "smoking", "smoker"),
    "diabetes": ("diabetes",),
    "sex": ("sex",),
    "prior_chd": ("prior chd", "heart disease", "chd"),
    "prior_cancer": ("prior cancer", "cancer"),
}
# Longest alias first, so "income ratio" wins over a bare "income" substring.
_ALIAS_ORDER = sorted(
    ((alias, feature) for feature, aliases in FEATURE_ALIASES.items()
     for alias in aliases),
    key=lambda pair: len(pair[0]), reverse=True)

CASE_RE = re.compile(r"case[\s_#]*0*(\d+)", re.IGNORECASE)
# Direction and the percentage number are matched independently (not by
# proximity): a feature name like "hba1c" contains a digit, which would
# otherwise break a single "word then number" regex before it ever reaches
# the real percentage.
UP_WORDS_RE = re.compile(r"\b(?:raise|raising|increase|higher|up)\b",
                         re.IGNORECASE)
DOWN_WORDS_RE = re.compile(r"\b(?:lower|lowering|decrease|reduce|drop|down)\b",
                           re.IGNORECASE)
PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
ABS_RE = re.compile(r"(?:\bto\b|\bwere\b|=|\bis\b)\s+(-?\d+(?:\.\d+)?)\b",
                    re.IGNORECASE)


def _find_feature(utterance: str) -> str | None:
    lowered = utterance.lower()
    for alias, feature in _ALIAS_ORDER:
        if alias in lowered:
            return feature
    return None


def _find_categorical_value(feature: str, utterance: str) -> str | None:
    lowered = utterance.lower()
    for level in CATEGORICAL_TERMS[feature]:
        if re.search(rf"\b{re.escape(level)}\b", lowered):
            return level
    return None


@dataclass
class ParsedIntent:
    tool: str
    kwargs: dict[str, Any]


def parse_utterance(session: Session, utterance: str) -> ParsedIntent:
    """Deterministic intent parser: utterance -> (tool name, kwargs).

    Raises ToolError with a plain-English message on anything it cannot
    confidently parse, rather than guessing.
    """
    case_match = CASE_RE.search(utterance)
    explicit_case_number = int(case_match.group(1)) if case_match else None

    is_what_if = bool(re.search(r"\bwhat\s*if\b", utterance, re.IGNORECASE))
    feature = _find_feature(utterance) if is_what_if or explicit_case_number is None else None

    if explicit_case_number is not None and not is_what_if and feature is None:
        return ParsedIntent("lookup_case_by_number", {"number": explicit_case_number})

    if is_what_if or feature is not None:
        if feature is None:
            raise ToolError(
                "I couldn't tell which feature to change. Known features: "
                f"{', '.join(sorted(FEATURE_ALIASES))}.")
        case_id = (session.cases[explicit_case_number - 1]["case_id"]
                   if explicit_case_number is not None
                   else session.resolve_case_id(None))

        if feature in CATEGORICAL_TERMS:
            value = _find_categorical_value(feature, utterance)
            if value is None:
                raise ToolError(
                    f"For {feature!r}, say which of "
                    f"{sorted(CATEGORICAL_TERMS[feature])} you mean.")
            return ParsedIntent("what_if",
                                 {"case_id": case_id, "feature": feature,
                                  "new_value": value})

        base_case = lookup_case(session.cases, case_id)
        old_value = base_case["features"][feature]
        pct = PCT_RE.search(utterance)
        going_up = bool(UP_WORDS_RE.search(utterance))
        going_down = bool(DOWN_WORDS_RE.search(utterance))
        absolute = ABS_RE.search(utterance)
        if pct and (going_up or going_down) and not (going_up and going_down):
            sign = 1 if going_up else -1
            new_value = old_value * (1 + sign * float(pct.group(1)) / 100)
        elif absolute:
            new_value = float(absolute.group(1))
        else:
            raise ToolError(
                f"Say how {feature!r} should change, e.g. 'raise it by 10%' "
                f"or 'set it to 30'.")
        return ParsedIntent("what_if",
                             {"case_id": case_id, "feature": feature,
                              "new_value": new_value})

    return ParsedIntent("query_model_card", {"question": utterance})


# ------------------------------------------------------------- tool execution

def format_what_if(result) -> str:
    return (
        f"Case {result.case_id}: changing {result.feature} from "
        f"{result.old_value!r} to {result.new_value!r} moves the predicted "
        f"36-month risk from {result.base_risk:.2%} to {result.new_risk:.2%} "
        f"({result.risk_delta_pct_points:+.2f} points), recomputed from the "
        f"fitted GLM's own coefficient for {result.feature} "
        f"({result.log_odds_delta:+.4f} log-odds). This is relative to the "
        f"case's original modelled risk, not a chained previous what-if."
    )


def format_case(case: dict) -> str:
    drivers = ", ".join(d["statement"] for d in case["top_drivers"][:3])
    return (f"{case['case_id']}: predicted 36-month risk "
            f"{case['predicted_risk_36mo']:.2%}, decile "
            f"{case['risk_decile']}/10. Top drivers: {drivers}.")


def format_model_card_answer(sections: dict) -> str:
    return json.dumps(sections, indent=2)


def dispatch(session: Session, utterance: str) -> str:
    """Run the deterministic path for one turn and update session state."""
    intent = parse_utterance(session, utterance)

    if intent.tool == "lookup_case_by_number":
        case = lookup_case_by_number(session.cases, **intent.kwargs)
        session.current_case_id = case["case_id"]
        return format_case(case)

    if intent.tool == "what_if":
        result = what_if(session.cases, session.model_card, **intent.kwargs)
        session.current_case_id = result.case_id
        return format_what_if(result)

    if intent.tool == "query_model_card":
        sections = query_model_card(session.model_card, **intent.kwargs)
        return format_model_card_answer(sections)

    raise ToolError(f"Unhandled intent {intent.tool!r}.")  # pragma: no cover


# --------------------------------------------------------------- LLM path

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_case_by_number",
            "description": "Look up a modelled case by its 1-based number.",
            "parameters": {
                "type": "object",
                "properties": {"number": {"type": "integer"}},
                "required": ["number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_model_card",
            "description": ("Answer a question about the model's cohort, "
                             "validation metrics, predictors, limitations "
                             "or coefficients, from the model card."),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "what_if",
            "description": (
                "Recompute a case's predicted risk with one feature "
                "changed, using the fitted GLM's own coefficients. "
                f"Continuous features: {', '.join(CONTINUOUS_TERMS)}. "
                f"Categorical features: {', '.join(CATEGORICAL_TERMS)}."),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "feature": {"type": "string"},
                    "new_value": {"type": "string"},
                },
                "required": ["case_id", "feature", "new_value"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a tool-using assistant over a mortality-risk model's own "
    "exported artifacts. Always answer by calling one of the provided "
    "tools -- never invent a number yourself. If the user's request is "
    "ambiguous, call query_model_card with their question verbatim."
)


def available_ollama_model() -> str | None:
    if shutil.which("ollama") is None:
        return None
    try:
        listed = subprocess.run(["ollama", "list"], capture_output=True,
                                text=True, timeout=10).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    for candidate in OLLAMA_MODELS:
        if candidate.split(":")[0] in listed:
            return candidate
    return None


def _ollama_chat(model: str, messages: list[dict], tools: list[dict] | None = None) -> dict:
    payload = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    request = urllib.request.Request(
        OLLAMA_CHAT_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def run_tool_call(session: Session, name: str, arguments: dict) -> str:
    if name == "lookup_case_by_number":
        case = lookup_case_by_number(session.cases, int(arguments["number"]))
        session.current_case_id = case["case_id"]
        return format_case(case)
    if name == "query_model_card":
        sections = query_model_card(session.model_card, arguments["question"])
        return format_model_card_answer(sections)
    if name == "what_if":
        new_value = arguments["new_value"]
        if arguments["feature"] not in CATEGORICAL_TERMS:
            new_value = float(new_value)
        result = what_if(session.cases, session.model_card,
                          arguments["case_id"], arguments["feature"], new_value)
        session.current_case_id = result.case_id
        return format_what_if(result)
    raise ToolError(f"The model tried to call an unknown tool: {name!r}.")


def dispatch_llm(session: Session, model: str, utterance: str) -> str:
    """LLM-orchestrated turn: ask the model which tool to call, run it, ask
    the model to phrase the final reply. Raises on any failure; the caller
    falls back to the deterministic dispatcher."""
    history = [{"role": role, "content": text} for role, text in session.turns]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history,
                {"role": "user", "content": utterance}]

    first = _ollama_chat(model, messages, tools=TOOL_SCHEMAS)
    message = first["message"]
    tool_calls = message.get("tool_calls")
    if not tool_calls:
        return message.get("content", "").strip()

    call = tool_calls[0]["function"]
    tool_result = run_tool_call(session, call["name"], call["arguments"])

    messages.append(message)
    messages.append({"role": "tool", "content": tool_result})
    second = _ollama_chat(model, messages)
    return second["message"].get("content", "").strip() or tool_result


# ------------------------------------------------------------------- CLI

def load_data() -> tuple[list[dict], dict]:
    if not CASES_PATH.exists() or not MODEL_CARD_PATH.exists():
        raise typer.Exit(_fail("Artifacts not found. Run `make models` first."))
    cases = json.loads(CASES_PATH.read_text())["cases"]
    model_card = json.loads(MODEL_CARD_PATH.read_text())
    return cases, model_card


def _fail(message: str) -> int:
    console.print(f"[bold red]{message}[/bold red]")
    return 1


def run_turn(session: Session, utterance: str, use_llm: bool,
             model: str | None) -> str:
    if use_llm and model:
        try:
            reply = dispatch_llm(session, model, utterance)
            engine = f"ollama · {model}"
        except (ToolError, urllib.error.URLError, OSError, KeyError,
                json.JSONDecodeError) as exc:
            console.print(f"[yellow]LLM tool call failed ({exc}); "
                          f"using the deterministic dispatcher.[/yellow]")
            reply = dispatch(session, utterance)
            engine = "deterministic dispatcher (fallback)"
    else:
        reply = dispatch(session, utterance)
        engine = "deterministic dispatcher"
    session.remember("user", utterance)
    session.remember("assistant", reply)
    return f"[{engine}] {reply}"


@app.command()
def main(llm: bool = typer.Option(False, "--llm",
                                   help="Opt into Ollama tool calling.")) -> None:
    cases, model_card = load_data()
    session = Session(cases=cases, model_card=model_card)
    model = available_ollama_model() if llm else None
    if llm and not model:
        console.print("[yellow]--llm requested but no Ollama model found; "
                      "using the deterministic dispatcher for this "
                      "session.[/yellow]")

    console.print("[dim]mortality-copilot agent. Try 'case 1', then "
                  "'what if age were 80', then 'what's the AUC?'. "
                  "Ctrl-D to quit.[/dim]\n"
                  "[dim]Educational demonstration on public survey data. "
                  "Not an underwriting system.[/dim]")
    while True:
        try:
            utterance = console.input("[bold cyan]> [/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not utterance.strip():
            continue
        try:
            console.print(escape(run_turn(session, utterance, llm, model)))
        except ToolError as exc:
            console.print(f"[red]{escape(str(exc))}[/red]")


if __name__ == "__main__":
    app()
