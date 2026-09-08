"""Tests for pipeline/voice.py's speech-to-text -> agent -> speech-to-text loop.

Runs against the two audio samples committed in artifacts/voice_samples/,
generated once via pyttsx3 itself (documented in the README) so the test is
reproducible without a microphone or a real phone call. Every voice_turn()
assertion also cross-checks the transcribed text against a direct dispatch()
call on that same text, so the voice interface can't silently diverge from
the text interface it wraps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.agent import Session, dispatch, load_data
from pipeline.voice import transcribe, voice_turn

SAMPLES = Path(__file__).resolve().parent.parent / "artifacts" / "voice_samples"


@pytest.fixture
def session():
    cases, model_card = load_data()
    return Session(cases=cases, model_card=model_card)


def test_transcribe_case_lookup_sample():
    transcript = transcribe(SAMPLES / "sample_case1.aiff").lower()
    assert "case" in transcript
    assert "1" in transcript


def test_transcribe_what_if_sample():
    transcript = transcribe(SAMPLES / "sample_whatif_case1.aiff").lower()
    assert "age" in transcript
    assert "80" in transcript


def test_voice_turn_case_lookup_matches_direct_dispatch(session, tmp_path):
    audio_out = tmp_path / "reply.aiff"
    control_session = Session(cases=session.cases, model_card=session.model_card)

    result = voice_turn(session, SAMPLES / "sample_case1.aiff", audio_out)
    direct_reply = dispatch(control_session, result["transcript"])

    assert result["reply"] == f"[deterministic dispatcher] {direct_reply}"
    assert "case_001" in result["reply"]
    assert audio_out.exists()
    assert audio_out.stat().st_size > 0


def test_voice_turn_what_if_matches_direct_dispatch(session, tmp_path):
    dispatch(session, "case 1")  # establish case context, as a real caller would
    control_session = Session(cases=session.cases, model_card=session.model_card,
                               current_case_id=session.current_case_id)
    audio_out = tmp_path / "reply.aiff"

    result = voice_turn(session, SAMPLES / "sample_whatif_case1.aiff", audio_out)
    direct_reply = dispatch(control_session, result["transcript"])

    assert result["reply"] == f"[deterministic dispatcher] {direct_reply}"
    assert "risk" in result["reply"].lower()
    assert audio_out.exists()
    assert audio_out.stat().st_size > 0
