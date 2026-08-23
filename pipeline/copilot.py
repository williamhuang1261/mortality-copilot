"""Explain one modelled case, grounded in the retrieval corpus.

    python -m py.copilot --case 17
    python -m py.copilot --ask "how is follow-up time measured?"

Generation uses a local Ollama model when one is available. When it is not --
which is the default on a clean clone -- the copilot falls back to deterministic
extraction: the same retrieved excerpts and the same citations, assembled by
template instead of by a language model. The fallback exists so that `make demo`
always produces output, and so the citation discipline can be tested without a
model in the loop.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"
CASES_PATH = ROOT / "artifacts" / "cases.json"
PROMPT_PATH = ROOT / "prompts" / "underwriting_note.md"

OLLAMA_MODELS = ("llama3.2:3b", "qwen2.5:3b")
TOP_K = 5

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


@dataclass
class Retrieved:
    text: str
    source: str
    page: int | None
    score: float

    @property
    def citation(self) -> str:
        if self.page is None:
            return f"[source: {self.source}]"
        return f"[source: {self.source}, page {self.page}]"


class Corpus:
    def __init__(self) -> None:
        if not (INDEX_DIR / "corpus.faiss").exists():
            raise typer.Exit(_fail("No index found. Run `make index` first."))
        import faiss
        from sentence_transformers import SentenceTransformer

        payload = json.loads((INDEX_DIR / "chunks.json").read_text())
        self.chunks = payload["chunks"]
        self.index = faiss.read_index(str(INDEX_DIR / "corpus.faiss"))
        self.model = SentenceTransformer(payload["model"])

    def search(self, query: str, k: int = TOP_K) -> list[Retrieved]:
        vector = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(vector, k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            results.append(Retrieved(chunk["text"], chunk["source"],
                                     chunk["page"], float(score)))
        return results


def _fail(message: str) -> int:
    console.print(f"[bold red]{message}[/bold red]")
    return 1


# ------------------------------------------------------------- generation

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


def generate_with_ollama(model: str, prompt: str) -> str:
    result = subprocess.run(["ollama", "run", model], input=prompt,
                            capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ollama failed")
    return result.stdout.strip()


def extractive_note(case: dict, retrieved: list[Retrieved],
                    caveats: list[Retrieved] | None = None) -> str:
    """Deterministic fallback. Same facts, same citations, no model."""
    risk = case["predicted_risk_36mo"]
    drivers = case["top_drivers"]
    raising = [d for d in drivers if d["contribution_log_odds"] > 0]
    lowering = [d for d in drivers if d["contribution_log_odds"] <= 0]

    parts = [
        f"The model estimates a {risk:.1%} probability of death within 36 months "
        f"of examination for {case['case_id']}, placing it in risk decile "
        f"{case['risk_decile']} of 10 for this cohort."
    ]
    if raising:
        parts.append("Characteristics pushing the estimate up, relative to the "
                     "cohort average, are " +
                     _join(d["statement"] for d in raising) + ".")
    if lowering:
        parts.append("Pushing it down: " +
                     _join(d["statement"] for d in lowering) + ".")
    if case["features"].get("income_ratio_imputed"):
        parts.append("The income-to-poverty ratio for this record was not "
                     "reported and has been imputed with the cohort median, so "
                     "it is not a measured value.")
    # The caveat gets its own retrieval (see CAVEAT_QUERY). Scavenging the
    # case-context hits for a limitation sentence does not work: none of the
    # five chunks retrieved for a case necessarily discusses limitations, and
    # keyword-sniffing them picks whatever happens to match.
    source = caveats or retrieved
    if source:
        top = source[0]
        parts.append(f"On the limitations of the underlying data: "
                     f"{_caveat_sentence(top.text)} {top.citation}")
    return " ".join(parts)


CAVEAT_TOKENS = ("limitation", "caution", "should not", "cannot", "may not",
                 "assumed alive", "underestimat", "ascertain", "not all",
                 "subject to", "uncertain", "bias")


def _caveat_sentence(text: str, limit: int = 240) -> str:
    """Prefer a sentence that actually states a caveat.

    The top-ranked chunk of a caveats query is usually the right *document*,
    but its first sentence is often boilerplate introduction. Retrieval picks
    the passage; this picks the sentence within it.
    """
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip())
                 if len(part.strip()) >= 60]
    for sentence in sentences:
        if any(token in sentence.lower() for token in CAVEAT_TOKENS):
            if len(sentence) > limit:
                sentence = sentence[:limit].rsplit(" ", 1)[0] + "..."
            return sentence.rstrip(".") + "."
    return _first_sentence(text, limit)


def _join(items) -> str:
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _first_sentence(text: str, limit: int = 240) -> str:
    """First sentence with enough substance to stand alone.

    Naively taking text.split(". ")[0] returns whatever fragment happens to come
    first -- for the model card that is the section heading, so the note ended
    with the single word "cohort".
    """
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip())]
    candidate = next((s for s in sentences if len(s) >= 60), None)
    if candidate is None:
        candidate = " ".join(sentences)[:limit].strip()
    if len(candidate) > limit:
        candidate = candidate[:limit].rsplit(" ", 1)[0] + "..."
    return candidate.rstrip(".") + "."


def build_prompt(case: dict, retrieved: list[Retrieved]) -> str:
    features = "\n".join(
        f"  - {key.replace('_', ' ')}: {value}"
        for key, value in case["features"].items()
        if key != "income_ratio_imputed"
    )
    if case["features"].get("income_ratio_imputed"):
        features += "\n  - note: the income-to-poverty ratio is IMPUTED, not measured"
    drivers = "\n".join(
        f"  - {d['statement']} ({d['direction']}, "
        f"log-odds {d['contribution_log_odds']:+.3f})"
        for d in case["top_drivers"]
    )
    context = "\n\n".join(
        f"### Excerpt {i} {r.citation}\n{r.text}"
        for i, r in enumerate(retrieved, start=1)
    )
    template = PROMPT_PATH.read_text()
    return (template
            .replace("{{CASE_ID}}", case["case_id"])
            .replace("{{RISK_PCT}}", f"{case['predicted_risk_36mo']:.1%}")
            .replace("{{RISK_DECILE}}", str(case["risk_decile"]))
            .replace("{{FEATURES}}", features)
            .replace("{{DRIVERS}}", drivers)
            .replace("{{CONTEXT}}", context))


CAVEAT_QUERY = (
    "limitations and cautions when using the public-use linked mortality file: "
    "assumed alive status, ascertainment of deaths, underestimation of mortality"
)


def query_from_case(case: dict) -> str:
    drivers = " ".join(d["label"] for d in case["top_drivers"])
    return f"mortality risk factors: {drivers}"


# ------------------------------------------------------------------- CLI

def load_cases() -> list[dict]:
    if not CASES_PATH.exists():
        raise typer.Exit(_fail("artifacts/cases.json not found. Run `make models` first."))
    return json.loads(CASES_PATH.read_text())["cases"]


@app.command()
def main(
    case: int = typer.Option(None, help="Case number from artifacts/cases.json (1-based)."),
    ask: str = typer.Option(None, help="Ask the corpus a question instead."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Force the deterministic fallback."),
) -> None:
    if case is None and ask is None:
        raise typer.Exit(_fail("Give either --case N or --ask \"question\"."))

    corpus = Corpus()

    if ask:
        retrieved = corpus.search(ask)
        console.print(Panel(ask, title="Question", border_style="cyan"))
        for r in retrieved:
            console.print(f"\n[bold]{escape(r.citation)}[/bold]  "
                          f"similarity {r.score:.3f}")
            console.print(escape(_first_sentence(r.text, limit=400)))
        return

    cases = load_cases()
    if not 1 <= case <= len(cases):
        raise typer.Exit(_fail(f"--case must be between 1 and {len(cases)}."))
    record = cases[case - 1]

    retrieved = corpus.search(query_from_case(record))
    caveats = corpus.search(CAVEAT_QUERY, k=3)

    table = Table(title=f"{record['case_id']} — modelled 36-month mortality risk",
                  show_header=True, header_style="bold")
    table.add_column("Driver")
    table.add_column("Contribution (log-odds)", justify="right")
    table.add_column("Direction")
    for d in record["top_drivers"]:
        style = "red" if d["contribution_log_odds"] > 0 else "green"
        table.add_row(d["statement"], f"{d['contribution_log_odds']:+.3f}",
                      f"[{style}]{d['direction']}[/{style}]")

    console.print()
    console.print(Panel(
        f"Estimated risk: [bold]{record['predicted_risk_36mo']:.2%}[/bold]"
        f"   ·   decile {record['risk_decile']}/10"
        f"   ·   prediction is out-of-fold",
        border_style="cyan"))
    console.print(table)

    model = None if no_llm else available_ollama_model()
    if model:
        try:
            note = generate_with_ollama(
                model, build_prompt(record, retrieved + caveats))
            engine = f"ollama · {model}"
        except (RuntimeError, subprocess.SubprocessError) as exc:
            console.print(f"[yellow]Ollama failed ({exc}); using the fallback.[/yellow]")
            note, engine = (extractive_note(record, retrieved, caveats),
                            "deterministic fallback")
    else:
        note, engine = (extractive_note(record, retrieved, caveats),
                        "deterministic fallback")

    # escape(): citations are square-bracketed, which rich would otherwise
    # parse as style markup and drop -- silently deleting the one thing the
    # note is supposed to guarantee.
    console.print(Panel(escape(note), title=f"Case note — {engine}",
                        border_style="green"))
    console.print("[dim]Sources retrieved:[/dim]")
    for r in retrieved + caveats:
        console.print(f"  [dim]{escape(r.citation)}  "
                      f"(similarity {r.score:.3f})[/dim]")
    console.print("\n[dim]Educational demonstration on public survey data. "
                  "Not an underwriting system.[/dim]")


if __name__ == "__main__":
    app()
