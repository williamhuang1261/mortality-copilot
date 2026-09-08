"""A voice interface over the existing tool-calling agent.

    python -m pipeline.voice <audio_in> <audio_out>

Speech in, the same `run_turn()` Extension 8's text and MCP interfaces
already call, speech out. No agent logic lives here: this module is only a
transcribe/synthesize wrapper around `pipeline/agent.py`, the same "reuse,
don't reimplement" pattern `pipeline/mcp_server.py` and `pipeline/api.py`
already follow for `pipeline/tools.py`.

Both directions are fully local and open source:

- **Speech-to-text:** OpenAI's Whisper (`tiny.en`, CPU), loaded once at
  module import so a batch of calls (tests, a multi-file demo) doesn't
  reload the model each time.
- **Text-to-speech:** `pyttsx3`, which drives the host OS's own TTS engine
  (NSSpeechSynthesizer on macOS, SAPI5 on Windows, espeak on Linux) -- no
  model download, no network call.

Whisper's own downloader needs a working TLS trust store; some Python
installs (this one, on first run) ship without one wired up, which surfaces
as a certificate-verification error, not a Whisper bug. Pointing
`SSL_CERT_FILE` at `certifi`'s bundle before the download runs fixes it
without touching the system's certificate store.
"""

from __future__ import annotations

import os
from pathlib import Path

import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import pyttsx3
import typer
import whisper

from pipeline.agent import Session, available_ollama_model, load_data, run_turn

WHISPER_MODEL_NAME = "tiny.en"

app = typer.Typer(add_completion=False, help=__doc__)

_model: "whisper.Whisper | None" = None


def _whisper_model() -> "whisper.Whisper":
    """Load the Whisper model once and cache it at module scope."""
    global _model
    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL_NAME)
    return _model


def transcribe(audio_path: Path) -> str:
    """Speech-to-text over one audio file, using the cached Whisper model."""
    result = _whisper_model().transcribe(str(audio_path))
    return result["text"].strip()


def synthesize(text: str, out_path: Path) -> None:
    """Text-to-speech, written to `out_path` via the host OS's TTS engine."""
    engine = pyttsx3.init()
    engine.save_to_file(text, str(out_path))
    engine.runAndWait()


def voice_turn(session: Session, audio_in: Path, audio_out: Path,
               use_llm: bool = False, model: str | None = None) -> dict:
    """One full voice turn: transcribe `audio_in`, run it through the
    existing agent, synthesize the reply to `audio_out`.

    Returns `{"transcript": ..., "reply": ...}` so callers (tests, the CLI)
    can inspect both the recognized text and the agent's answer.
    """
    transcript = transcribe(audio_in)
    reply = run_turn(session, transcript, use_llm, model)
    synthesize(reply, audio_out)
    return {"transcript": transcript, "reply": reply}


@app.command()
def main(
    audio_in: Path = typer.Argument(..., help="Input audio file (question)."),
    audio_out: Path = typer.Argument(..., help="Output audio file (reply)."),
    llm: bool = typer.Option(False, "--llm",
                              help="Opt into Ollama tool calling."),
) -> None:
    """Run one voice turn from the command line."""
    cases, model_card = load_data()
    session = Session(cases=cases, model_card=model_card)
    model = available_ollama_model() if llm else None

    result = voice_turn(session, audio_in, audio_out, llm, model)
    typer.echo(f"Transcript: {result['transcript']}")
    typer.echo(f"Reply: {result['reply']}")
    typer.echo(f"Reply audio written to: {audio_out}")


if __name__ == "__main__":
    app()
