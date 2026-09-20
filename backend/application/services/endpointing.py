from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from concurrent.futures import Executor, Future
from enum import StrEnum

from backend.application.ports.stt_port import TranscriptEvent
from backend.domain.value_objects.audio import PCM16_16K_MONO, AudioChunk, AudioFormat

log = logging.getLogger(__name__)


class EndpointerKind(StrEnum):
    SEMANTIC = "semantic"
    SILENCE = "silence"
    SMART_TURN = "smart_turn"


class _TranscriptTracker:
    def __init__(self) -> None:
        self.finals: list[str] = []
        self.interim = ""

    def observe(self, event: TranscriptEvent) -> None:
        if event.is_final:
            if event.text.strip():
                self.finals.append(event.text.strip())
            self.interim = ""
        else:
            self.interim = event.text.strip()

    @property
    def text(self) -> str:
        return " ".join([*self.finals, self.interim]).strip()


class _SpeechTracker:
    def __init__(self) -> None:
        self.in_speech = False
        self.heard_speech = False
        self.last_speech: float | None = None

    def observe(self, is_speech: bool, now: float) -> None:
        if is_speech:
            self.in_speech = True
            self.heard_speech = True
            self.last_speech = now
        else:
            self.in_speech = False

    def silence_ms(self, now: float) -> float | None:
        if not self.heard_speech or self.in_speech or self.last_speech is None:
            return None
        return (now - self.last_speech) * 1000


class SilenceEndpointDetector:
    """Baseline: commit after a fixed silence. Too short and it cuts callers off mid-thought;
    too long and every turn pays the full wait. That trade-off is what the semantic
    detector exists to escape."""

    def __init__(self, silence_ms: float = 700) -> None:
        self._silence_ms = silence_ms
        self.reset()

    def reset(self) -> None:
        self._speech = _SpeechTracker()
        self._transcript = _TranscriptTracker()

    def observe_audio(self, frame: AudioChunk, is_speech: bool, now: float) -> None:
        self._speech.observe(is_speech, now)

    def observe_transcript(self, event: TranscriptEvent, now: float) -> None:
        self._transcript.observe(event)

    def should_commit(self, now: float) -> bool:
        silence = self._speech.silence_ms(now)
        return silence is not None and silence >= self._silence_ms

    def observe_agent_turn(self, text: str) -> None:
        return None

    @property
    def speech_end(self) -> float | None:
        return self._speech.last_speech

    @property
    def transcript(self) -> str:
        return self._transcript.text


_INCOMPLETE_TAIL = frozenset(
    """
    and or but because so then the a an to of with for my your our their his her its in on
    at um uh er erm like if when that which who whose while although though since as is am
    are und oder aber weil dass denn der die das den dem des ein eine einen einem mit mein
    meine meinen ihr ihre für zu von bei im am auf wenn als ob also ähm äh hm ist bin sind
    habe
    """.split()  # noqa: SIM905
)
# Things people dictate in chunks, pausing after each: a full stop there is not the end of
# the turn. English and German, matched as word stems in the agent's last question.
_DICTATION_CUES = re.compile(
    r"\b(e-?mail|phone|number|name|spell|address|postcode|zip|date of birth|birthday|"
    r"telefon|nummer|namen?|buchstabier|adresse|anschrift|postleitzahl|geburtsdatum)",
    re.IGNORECASE,
)
_SENTENCES = re.compile(r"[^.!?]+[.!?]*")


def expects_dictation(agent_text: str) -> bool:
    """True when the agent's last question asks for something spoken in chunks. Only the
    last question counts: "Thanks, Aaron. Is this about employment?" asks for nothing."""
    sentences = [s.strip() for s in _SENTENCES.findall(agent_text) if s.strip()]
    questions = [s for s in sentences if s.endswith("?")]
    target = questions[-1] if questions else (sentences[-1] if sentences else "")
    return bool(_DICTATION_CUES.search(target))


_TRAILING_PUNCT = re.compile(r"[\"')\]\s]+$")


def heuristic_completeness(text: str) -> float:
    """Probability-like score that `text` is a finished turn. Cheap enough for frame rate;
    tuned for English and German receptionist dialogue."""
    stripped = _TRAILING_PUNCT.sub("", text.strip())
    if not stripped:
        return 0.5
    if stripped.endswith(("...", "…")):
        return 0.15
    last_char = stripped[-1]
    last_word = re.split(r"\s+", stripped.lower())[-1].strip(".,!?;:")
    if last_word in _INCOMPLETE_TAIL:
        return 0.1
    if last_char in ",;:-–—":
        return 0.2
    if last_char == "?":
        return 0.95
    if last_char in ".!":
        return 0.85
    return 0.5


class SemanticEndpointDetector:
    """Waits less when the words say the caller is done and longer when they trail off.

    The silence required scales with how complete the transcript looks: a clear question
    commits after `min_silence_ms`, a dangling "and my…" holds out to `max_silence_ms`.
    After the agent asks for an email, number or name, a finished-looking sentence still
    gets `default_silence_ms`: "My email is asad." is usually followed by "at hotmail".
    """

    def __init__(
        self,
        min_silence_ms: float = 250,
        default_silence_ms: float = 600,
        max_silence_ms: float = 1500,
        classifier: Callable[[str], float] = heuristic_completeness,
    ) -> None:
        self._min = min_silence_ms
        self._default = default_silence_ms
        self._max = max_silence_ms
        self._classifier = classifier
        self._cached_text: str | None = None
        self._cached_wait = default_silence_ms
        self._dictation = False
        self.reset()

    def reset(self) -> None:
        self._speech = _SpeechTracker()
        self._transcript = _TranscriptTracker()
        self._cached_text = None

    def observe_audio(self, frame: AudioChunk, is_speech: bool, now: float) -> None:
        self._speech.observe(is_speech, now)

    def observe_transcript(self, event: TranscriptEvent, now: float) -> None:
        self._transcript.observe(event)

    def observe_agent_turn(self, text: str) -> None:
        self._dictation = expects_dictation(text)
        self._cached_text = None

    def required_silence_ms(self) -> float:
        text = self._transcript.text
        if text != self._cached_text:
            self._cached_text = text
            self._cached_wait = self._wait_for(text)
        return self._cached_wait

    def should_commit(self, now: float) -> bool:
        silence = self._speech.silence_ms(now)
        return silence is not None and silence >= self.required_silence_ms()

    @property
    def speech_end(self) -> float | None:
        return self._speech.last_speech

    @property
    def transcript(self) -> str:
        return self._transcript.text

    def _wait_for(self, text: str) -> float:
        if not text:
            return self._default
        p = self._classifier(text)
        if p >= 0.8:
            return self._default if self._dictation else self._min
        if p <= 0.2:
            return self._max
        return self._default


MODEL_WINDOW_S = 8.0
"""What Smart Turn v3.2 reads: the last 8 s of the turn, left-padded when it is shorter.
Source: pipecat-ai/smart-turn-v3 inference.py. The adapter pads to the same figure in its
own units (`smart_turn.WINDOW_SAMPLES`); a model with a different window needs both."""
PRE_SPEECH_S = 0.5
"""Room tone kept ahead of the caller's first word, so the model hears the turn start."""


class SmartTurnEndpointDetector:
    """Asks an audio turn-completion model whether the caller has finished, once a short
    pause follows their speech, and commits when it says yes.

    `probability` returns P(turn complete) for 8 s of 16 kHz PCM16; Smart Turn v3.2 takes
    around 10 ms per call, which is half a frame, so it is run on `executor` when one is
    given and the frame loop keeps going while it thinks. It is asked once per pause and
    again on the whole turn if the caller resumes; a pause the model never calls complete
    commits at `ceiling_ms`, so a failing or slow model only costs the silence baseline.
    """

    def __init__(
        self,
        probability: Callable[[bytes], float],
        onset_ms: float = 200,
        ceiling_ms: float = 1500,
        threshold: float = 0.5,
        executor: Executor | None = None,
        timer: Callable[[], float] = time.perf_counter,
        audio_format: AudioFormat = PCM16_16K_MONO,
    ) -> None:
        self._probability = probability
        self._onset_ms = onset_ms
        self._ceiling_ms = ceiling_ms
        self._threshold = threshold
        self._executor = executor
        self._timer = timer
        self._format = audio_format
        self._window_bytes = audio_format.byte_count(MODEL_WINDOW_S)
        self._pre_speech_bytes = audio_format.byte_count(PRE_SPEECH_S)
        self._inference_ms: list[float] = []
        self._failures = 0
        self.reset()

    def reset(self) -> None:
        self._speech = _SpeechTracker()
        self._audio = bytearray()
        self._verdict: float | None = None
        self._asked = False
        self._pending: Future[tuple[float, float]] | None = None

    def observe_audio(self, frame: AudioChunk, is_speech: bool, now: float) -> None:
        if frame.format != self._format:
            raise ValueError(f"Smart Turn needs {self._format}, got {frame.format}")
        self._speech.observe(is_speech, now)
        self._audio += frame.data
        keep = self._window_bytes if self._speech.heard_speech else self._pre_speech_bytes
        if len(self._audio) > keep:
            del self._audio[: len(self._audio) - keep]
        if is_speech:
            self._forget_decision()
            return
        silence = self._speech.silence_ms(now)
        if not self._asked and silence is not None and silence >= self._onset_ms:
            self._asked = True
            self._pending = self._ask(self._window())

    def observe_transcript(self, event: TranscriptEvent, now: float) -> None:
        return None

    def observe_agent_turn(self, text: str) -> None:
        return None

    def should_commit(self, now: float) -> bool:
        silence = self._speech.silence_ms(now)
        if silence is None:
            return False
        if silence >= self._ceiling_ms:
            return True
        self._collect_decision()
        return self._verdict is not None and self._verdict > self._threshold

    @property
    def speech_end(self) -> float | None:
        return self._speech.last_speech

    @property
    def inference_ms(self) -> tuple[float, ...]:
        """How long the model took over each decision this call, in arrival order."""
        return tuple(self._inference_ms)

    @property
    def failures(self) -> int:
        """Decisions the model could not answer. Each one costs only the silence ceiling,
        so a model that fails every time still answers calls — as a silence endpointer.
        The benchmark publishes this count so a run labelled smart_turn that never asked
        the model says so."""
        return self._failures

    def _window(self) -> bytes:
        return bytes(self._window_bytes - len(self._audio)) + bytes(self._audio)

    def _ask(self, audio: bytes) -> Future[tuple[float, float]]:
        if self._executor is not None:
            return self._executor.submit(self._timed_probability, audio)
        decided: Future[tuple[float, float]] = Future()
        try:
            decided.set_result(self._timed_probability(audio))
        except Exception as exc:
            decided.set_exception(exc)
        return decided

    def _timed_probability(self, audio: bytes) -> tuple[float, float]:
        started = self._timer()
        return self._probability(audio), (self._timer() - started) * 1000

    def _collect_decision(self) -> None:
        if self._pending is None or not self._pending.done():
            return
        decided, self._pending = self._pending, None
        try:
            self._verdict, inference_ms = decided.result()
        except Exception:
            self._failures += 1
            log.warning(
                "Smart Turn inference failed; holding for the silence ceiling", exc_info=True
            )
            return
        self._inference_ms.append(inference_ms)

    def _forget_decision(self) -> None:
        """The caller is speaking again: whatever the model was asked about is stale."""
        if self._pending is not None:
            self._pending.cancel()
        self._pending = None
        self._asked = False
        self._verdict = None
