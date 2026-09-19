import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest

from backend.application.ports.endpoint_detector import EndpointDetector
from backend.application.ports.stt_port import TranscriptEvent
from backend.application.services.endpointing import (
    EndpointerKind,
    SemanticEndpointDetector,
    SilenceEndpointDetector,
    SmartTurnEndpointDetector,
    expects_dictation,
    heuristic_completeness,
)
from backend.domain.value_objects.audio import PCM16_48K_MONO, AudioChunk

FRAME = AudioChunk(b"\x00\x00" * 320)

DETECTORS: dict[EndpointerKind, Callable[[], EndpointDetector]] = {
    EndpointerKind.SILENCE: SilenceEndpointDetector,
    EndpointerKind.SEMANTIC: SemanticEndpointDetector,
    EndpointerKind.SMART_TURN: lambda: SmartTurnEndpointDetector(lambda audio: 0.5),
}


def test_every_endpointer_kind_has_shared_behaviour_tests() -> None:
    assert set(DETECTORS) == set(EndpointerKind)


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


@pytest.mark.parametrize("kind", list(EndpointerKind))
def test_no_commit_without_speech_or_while_speaking(kind: EndpointerKind) -> None:
    d = DETECTORS[kind]()
    d.observe_audio(FRAME, False, 0)
    assert not d.should_commit(10)
    d.observe_audio(FRAME, True, 11)
    assert not d.should_commit(20)


@pytest.mark.parametrize("kind", list(EndpointerKind))
def test_commits_after_sustained_silence_stamping_the_end_of_speech(kind: EndpointerKind) -> None:
    d = DETECTORS[kind]()
    last = speak_then_silence(d)
    assert not d.should_commit(last + 0.1)
    assert d.should_commit(last + 2.0)
    assert d.speech_end == pytest.approx(last)


@pytest.mark.parametrize("kind", list(EndpointerKind))
def test_reset_forgets_speech(kind: EndpointerKind) -> None:
    d = DETECTORS[kind]()
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


def first_commit_after_silence(detector: EndpointDetector, last_speech: float) -> float:
    """Feeds silence frames after `last_speech` as the call session does, returning how many
    seconds of silence passed before the detector committed."""
    t = last_speech + 0.02
    while t < last_speech + 5:
        detector.observe_audio(FRAME, False, t)
        if detector.should_commit(t):
            return t - last_speech
        t += 0.02
    raise AssertionError("never committed")


def test_smart_turn_commits_a_complete_utterance_sooner_than_the_silence_baseline() -> None:
    baseline = SilenceEndpointDetector()
    smart = SmartTurnEndpointDetector(lambda audio: 0.9)
    baseline_wait = first_commit_after_silence(baseline, speak_then_silence(baseline))
    smart_wait = first_commit_after_silence(smart, speak_then_silence(smart))
    assert smart_wait == pytest.approx(0.2, abs=0.021)
    assert smart_wait < baseline_wait - 0.4


def test_smart_turn_holds_a_trailing_clause_until_the_silence_ceiling() -> None:
    d = SmartTurnEndpointDetector(lambda audio: 0.1, ceiling_ms=1500)
    assert first_commit_after_silence(d, speak_then_silence(d)) == pytest.approx(1.5, abs=0.021)


WINDOW_BYTES = 8 * 16_000 * 2
VOICE = AudioChunk(b"\x01\x00" * 320)
OTHER_VOICE = AudioChunk(b"\x03\x00" * 320)
ROOM = AudioChunk(b"\x02\x00" * 320)  # quiet but not digital silence, unlike the padding
# A pause is measured from the last speech frame, so 200 ms of silence is ten frames; the
# threshold sits just under it to keep the count exact against floating-point frame times.
TEN_FRAME_ONSET_MS = 190


class RecordingModel:
    def __init__(self, probability: float = 0.1) -> None:
        self.heard: list[bytes] = []
        self.probability = probability

    def __call__(self, audio: bytes) -> float:
        self.heard.append(audio)
        return self.probability


def feed(
    detector: EndpointDetector, frame: AudioChunk, seconds: float, t: float, speech: bool = False
) -> float:
    for _ in range(round(seconds / 0.02)):
        detector.observe_audio(frame, speech, t)
        detector.should_commit(t)
        t += 0.02
    return t


def test_smart_turn_hears_a_short_turn_left_padded_to_eight_seconds() -> None:
    model = RecordingModel()
    d = SmartTurnEndpointDetector(model, onset_ms=TEN_FRAME_ONSET_MS)
    t = feed(d, ROOM, 3.0, 0.0)  # only the last half second before speech is kept
    t = feed(d, VOICE, 1.0, t, speech=True)
    feed(d, ROOM, 0.5, t)

    [audio] = model.heard
    kept = ROOM.data * 25 + VOICE.data * 50 + ROOM.data * 10
    assert len(audio) == WINDOW_BYTES
    assert audio == bytes(WINDOW_BYTES - len(kept)) + kept


def test_smart_turn_hears_only_the_last_eight_seconds_of_a_long_turn() -> None:
    model = RecordingModel()
    d = SmartTurnEndpointDetector(model, onset_ms=TEN_FRAME_ONSET_MS)
    t = feed(d, VOICE, 1.0, 0.0, speech=True)
    t = feed(d, OTHER_VOICE, 9.0, t, speech=True)
    feed(d, ROOM, 0.3, t)

    [audio] = model.heard
    assert len(audio) == WINDOW_BYTES
    assert audio == OTHER_VOICE.data * 390 + ROOM.data * 10
    assert VOICE.data not in audio


def test_smart_turn_asks_once_per_pause_and_rehears_the_whole_turn_when_speech_resumes() -> None:
    model = RecordingModel(probability=0.1)
    d = SmartTurnEndpointDetector(model, onset_ms=TEN_FRAME_ONSET_MS)
    t = feed(d, VOICE, 1.0, 0.0, speech=True)
    t = feed(d, ROOM, 0.6, t)
    assert len(model.heard) == 1

    t = feed(d, OTHER_VOICE, 0.5, t, speech=True)
    feed(d, ROOM, 0.6, t)
    assert len(model.heard) == 2
    assert model.heard[1].endswith(
        VOICE.data * 50 + ROOM.data * 30 + OTHER_VOICE.data * 25 + ROOM.data * 10
    )


def test_smart_turn_starts_each_turn_from_fresh_audio_after_reset() -> None:
    model = RecordingModel()
    d = SmartTurnEndpointDetector(model, onset_ms=TEN_FRAME_ONSET_MS)
    t = feed(d, VOICE, 1.0, 0.0, speech=True)
    d.reset()
    t = feed(d, OTHER_VOICE, 0.5, t, speech=True)
    feed(d, ROOM, 0.3, t)
    assert VOICE.data not in model.heard[-1]


def test_smart_turn_records_what_each_decision_cost_in_inference_time() -> None:
    ticks = iter([0.0, 0.012, 1.0, 1.009])
    d = SmartTurnEndpointDetector(
        lambda audio: 0.1, onset_ms=TEN_FRAME_ONSET_MS, timer=lambda: next(ticks)
    )
    t = feed(d, VOICE, 0.5, 0.0, speech=True)
    t = feed(d, ROOM, 0.3, t)
    t = feed(d, VOICE, 0.5, t, speech=True)
    feed(d, ROOM, 0.3, t)
    assert list(d.inference_ms) == pytest.approx([12.0, 9.0])


def test_smart_turn_keeps_the_audio_loop_moving_while_the_model_thinks() -> None:
    started, release = threading.Event(), threading.Event()

    def slow_model(audio: bytes) -> float:
        started.set()
        release.wait(5)
        return 0.9

    with ThreadPoolExecutor(max_workers=1) as pool:
        d = SmartTurnEndpointDetector(slow_model, onset_ms=TEN_FRAME_ONSET_MS, executor=pool)
        t = feed(d, VOICE, 0.5, 0.0, speech=True)
        t = feed(d, ROOM, 0.3, t)
        assert started.wait(5)
        assert not d.should_commit(t)  # thinking, and the frames kept arriving

        release.set()
        deadline = time.monotonic() + 5
        # The pause stays at 300 ms, far short of the ceiling, so committing here can only
        # be the model's verdict arriving.
        while not d.should_commit(t) and time.monotonic() < deadline:
            time.sleep(0.005)
        assert d.should_commit(t)
        assert len(d.inference_ms) == 1


def test_smart_turn_falls_back_to_the_ceiling_when_the_model_fails() -> None:
    def broken_model(audio: bytes) -> float:
        raise RuntimeError("no such model file")

    d = SmartTurnEndpointDetector(broken_model, onset_ms=TEN_FRAME_ONSET_MS, ceiling_ms=1500)
    last = speak_then_silence(d)
    assert first_commit_after_silence(d, last) == pytest.approx(1.5, abs=0.021)
    assert d.inference_ms == ()


def test_smart_turn_refuses_audio_it_was_not_trained_on() -> None:
    d = SmartTurnEndpointDetector(lambda audio: 0.9)
    with pytest.raises(ValueError, match="16000"):
        d.observe_audio(AudioChunk(b"\x00\x00" * 960, PCM16_48K_MONO), True, 0.0)
