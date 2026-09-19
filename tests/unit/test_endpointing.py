import pytest

from backend.application.ports.stt_port import TranscriptEvent
from backend.application.services.endpointing import (
    SemanticEndpointDetector,
    SilenceEndpointDetector,
    expects_dictation,
    heuristic_completeness,
)
from backend.domain.value_objects.audio import AudioChunk

FRAME = AudioChunk(b"\x00\x00" * 320)


def speak_then_silence(detector, speech_s: float = 1.0, t0: float = 0.0) -> float:  # type: ignore[no-untyped-def]
    t = t0
    while t < t0 + speech_s:
        detector.observe_audio(FRAME, True, t)
        t += 0.02
    detector.observe_audio(FRAME, False, t)
    return t - 0.02  # time of last speech frame


def test_silence_detector_commits_after_threshold() -> None:
    d = SilenceEndpointDetector(silence_ms=700)
    last = speak_then_silence(d)
    assert not d.should_commit(last + 0.69)
    assert d.should_commit(last + 0.71)
    assert d.speech_end == pytest.approx(last)


def test_no_commit_without_speech_or_while_speaking() -> None:
    d = SilenceEndpointDetector()
    d.observe_audio(FRAME, False, 0)
    assert not d.should_commit(10)
    d.observe_audio(FRAME, True, 11)
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


@pytest.mark.parametrize(
    ("agent", "dictation"),
    [
        ("Thank you. Could you please tell me your full name?", True),
        ("Perfect. What's the best email for the confirmation?", True),
        ("And a phone number we can reach you on?", True),
        ("Wie ist Ihre Telefonnummer?", True),
        ("Darf ich Ihren Namen erfahren?", True),
        ("Thank you, Aaron. Is this regarding employment, tenancy or family law?", False),
        ("Thanks for your email. Is there a deadline coming up?", False),
        ("Goodbye, Aaron.", False),
        ("", False),
    ],
)
def test_expects_dictation_reads_only_the_last_question(agent: str, dictation: bool) -> None:
    assert expects_dictation(agent) is dictation


def test_semantic_waits_longer_after_a_request_for_an_email() -> None:
    """The cut-off from a real call: "Yeah, my email is asad." committed after 250 ms while
    the caller was mid-address."""
    d = SemanticEndpointDetector(min_silence_ms=250, default_silence_ms=600, max_silence_ms=1500)
    d.observe_agent_turn("Perfect. What's the best email for the confirmation?")
    last = speak_then_silence(d)
    d.observe_transcript(TranscriptEvent("Yeah, my email is asad.", is_final=True), last)
    assert not d.should_commit(last + 0.3)
    assert d.should_commit(last + 0.61)

    d.reset()  # the agent's question still frames the caller's next attempt
    last = speak_then_silence(d, t0=10)
    d.observe_transcript(TranscriptEvent("My name is Aaron.", is_final=True), last)
    assert d.required_silence_ms() == 600

    d.observe_agent_turn("Thank you, Aaron. Is this about employment or tenancy?")
    assert d.required_silence_ms() == 250


def test_silence_detector_ignores_agent_context() -> None:
    d = SilenceEndpointDetector(silence_ms=700)
    d.observe_agent_turn("What's your email?")
    last = speak_then_silence(d)
    assert d.should_commit(last + 0.71)
