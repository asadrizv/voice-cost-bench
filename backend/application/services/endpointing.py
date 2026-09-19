from __future__ import annotations

import re
from collections.abc import Callable

from backend.application.ports.stt_port import TranscriptEvent
from backend.domain.value_objects.audio import AudioChunk


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
