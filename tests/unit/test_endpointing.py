import pytest

from backend.application.ports.stt_port import TranscriptEvent
from backend.application.services.endpointing import (
    SemanticEndpointDetector,
    SilenceEndpointDetector,
    heuristic_completeness,
)


def speak_then_silence(detector, speech_s: float = 1.0, t0: float = 0.0) -> float:  # type: ignore[no-untyped-def]
    t = t0
    while t < t0 + speech_s:
        detector.observe_audio(True, t)
        t += 0.02
    detector.observe_audio(False, t)
    return t - 0.02  # time of last speech frame


def test_silence_detector_commits_after_threshold() -> None:
    d = SilenceEndpointDetector(silence_ms=700)
    last = speak_then_silence(d)
    assert not d.should_commit(last + 0.69)
    assert d.should_commit(last + 0.71)
    assert d.speech_end == pytest.approx(last)


def test_no_commit_without_speech_or_while_speaking() -> None:
    d = SilenceEndpointDetector()
    d.observe_audio(False, 0)
    assert not d.should_commit(10)
    d.observe_audio(True, 11)
    assert not d.should_commit(20)


def test_reset_forgets_speech() -> None:
    d = SilenceEndpointDetector()
    last = speak_then_silence(d)
    d.reset()
    assert not d.should_commit(last + 5)
    assert d.speech_end is None


@pytest.mark.parametrize(
    ("text", "complete"),
    [
        ("I'd like to book an appointment.", True),
        ("Can you call me back?", True),
        ("My landlord is refusing to return my deposit and", False),
        ("It's about my, um", False),
        ("Ich habe eine Frage zu meinem Mietvertrag.", True),
        ("Es geht um meinen Arbeitgeber, weil", False),
        ("Well...", False),
        ("the thing is,", False),
    ],
)
def test_heuristic_completeness(text: str, complete: bool) -> None:
    score = heuristic_completeness(text)
    assert (score >= 0.8) is complete
    if not complete:
        assert score <= 0.2


def test_heuristic_neutral_on_unpunctuated_or_empty() -> None:
    assert heuristic_completeness("") == 0.5
    assert heuristic_completeness("yes") == 0.5


def test_semantic_commits_fast_on_complete_sentence() -> None:
    d = SemanticEndpointDetector(min_silence_ms=250, default_silence_ms=600, max_silence_ms=1500)
    last = speak_then_silence(d)
    d.observe_transcript(TranscriptEvent("Can I get an appointment?", is_final=True), last)
    assert d.required_silence_ms() == 250
    assert d.should_commit(last + 0.26)


def test_semantic_holds_on_trailing_conjunction() -> None:
    d = SemanticEndpointDetector(min_silence_ms=250, default_silence_ms=600, max_silence_ms=1500)
    last = speak_then_silence(d)
    d.observe_transcript(TranscriptEvent("My employer fired me because", is_final=False), last)
    assert not d.should_commit(last + 1.0)
    assert d.should_commit(last + 1.51)


def test_semantic_uses_default_without_transcript_or_when_unsure() -> None:
    d = SemanticEndpointDetector(min_silence_ms=250, default_silence_ms=600, max_silence_ms=1500)
    speak_then_silence(d)
    assert d.required_silence_ms() == 600
    d.observe_transcript(TranscriptEvent("yes", is_final=True), 0)
    assert d.required_silence_ms() == 600
    assert d.transcript == "yes"


def test_semantic_transcript_joins_finals_and_interim() -> None:
    d = SemanticEndpointDetector()
    d.observe_transcript(TranscriptEvent("Hello.", is_final=True), 0)
    d.observe_transcript(TranscriptEvent("I need", is_final=False), 0)
    assert d.transcript == "Hello. I need"
    d.observe_transcript(TranscriptEvent("I need help.", is_final=True), 0)
    assert d.transcript == "Hello. I need help."
    d.reset()
    assert d.transcript == "" and d.speech_end is None
