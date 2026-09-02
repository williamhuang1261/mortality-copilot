"""Demo CLI for the versioned rules engine and audit trail.

`python -m pipeline.rules_demo --case case_001 --as-of 2026-08-01` evaluates
every rule against a real case as of a given date, prints the result, and
appends it to `artifacts/audit_log.jsonl` -- the same append-only log every
other invocation of `evaluate_and_log` writes to, so running this demo
repeatedly grows the log rather than overwriting it (the README's pasted
transcript reflects a single, fresh run).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer

from pipeline.audit import evaluate_and_log

CASES_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "cases.json"

app = typer.Typer(add_completion=False)


@app.command()
def main(
    case: str = typer.Option("case_001", "--case", help="Case id, e.g. case_001."),
    as_of: str = typer.Option(
        "2026-08-01", "--as-of", help="Evaluation date, ISO format (YYYY-MM-DD)."
    ),
) -> None:
    cases = json.loads(CASES_PATH.read_text())["cases"]
    entries = evaluate_and_log(case, date.fromisoformat(as_of), cases)
    for entry in entries:
        print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    app()
