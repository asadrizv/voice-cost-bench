"""Renders fixture audio from the scripts in fixtures/: conversations, the WER corpus, and the
browser-test microphone files. Output is generated, never committed: macOS `say` voices may
not be redistributed, so each machine renders its own. Uses `say` on macOS and `espeak-ng`
elsewhere (CI); content and timing match, only the voice differs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
E2E = ROOT / "frontend" / "e2e"
ESPEAK_VOICE = {"en": "en-us", "de": "de"}


def _read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels()).mean(axis=1).astype("<i2")
        return audio, w.getframerate()


def _write(path: Path, audio: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(audio.astype("<i2").tobytes())


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio
    positions = np.arange(0, len(audio), src / dst)
    return np.interp(positions, np.arange(len(audio)), audio).astype("<i2")


def render(text: str, language: str, voice: str | None, out: Path, rate: int = 16_000) -> None:
    """Mono PCM16 at `rate`. `voice` is a macOS voice name; ignored by espeak-ng."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.wav"
        if shutil.which("say"):
            args = ["say", "-o", str(raw), "--file-format=WAVE", f"--data-format=LEI16@{rate}"]
            subprocess.run([*args, *(["-v", voice] if voice else []), text], check=True)
        elif shutil.which("espeak-ng"):
            voice_id = ESPEAK_VOICE.get(language, language)
            subprocess.run(["espeak-ng", "-v", voice_id, "-w", str(raw), text], check=True)
        else:
            sys.exit("needs macOS `say` or `espeak-ng` (apt install espeak-ng)")
        audio, src_rate = _read(raw)
    _write(out, _resample(audio, src_rate, rate), rate)


def conversations(force: bool) -> None:
    spec = yaml.safe_load((FIXTURES / "conversations.yaml").read_text())
    for name, conv in spec["conversations"].items():
        for i, text in enumerate(conv["turns"]):
            out = FIXTURES / "audio" / name / f"{i:02d}.wav"
            if force or not out.exists():
                render(text, conv["language"], conv.get("voice"), out)
        print(f"{name}: {len(conv['turns'])} turns")


def wer_corpus(force: bool) -> None:
    spec = yaml.safe_load((FIXTURES / "wer" / "sentences.yaml").read_text())
    for lang, block in spec.items():
        count = 0
        for v_index, voice in enumerate(block["voices"]):
            for s_index, text in enumerate(block["sentences"]):
                out = FIXTURES / "audio" / "wer" / lang / f"{s_index:03d}_v{v_index}.wav"
                if force or not out.exists():
                    render(text, lang, voice, out)
                count += 1
        print(f"wer/{lang}: {count} utterances")


def e2e_microphones(force: bool) -> None:
    """Chromium plays these as the microphone in the Playwright tests (48 kHz)."""
    rate = 48_000

    def silence(seconds: float) -> np.ndarray:
        return np.zeros(int(rate * seconds), dtype="<i2")

    caller = E2E / "caller.wav"
    if force or not caller.exists():
        parts = [silence(6.0)]  # let the greeting play first
        for i in (0, 1):
            audio, src = _read(FIXTURES / "audio" / "intake_en" / f"{i:02d}.wav")
            parts += [_resample(audio, src, rate), silence(8.0)]
        _write(caller, np.concatenate(parts), rate)

    wrong = E2E / "wrong_number.wav"
    if force or not wrong.exists():
        with tempfile.TemporaryDirectory() as tmp:
            phrase = Path(tmp) / "phrase.wav"
            render(
                "Oh sorry, I think I have the wrong number. Bye!", "en", "Samantha", phrase, rate
            )
            audio, _ = _read(phrase)
        # Says goodbye once, then stays silent: only the agent can end this call.
        _write(wrong, np.concatenate([silence(6.0), audio, silence(40.0)]), rate)
    print("e2e: caller.wav, wrong_number.wav")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["conversations", "wer", "e2e", "all"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.what in ("conversations", "e2e", "all"):
        conversations(args.force)
    if args.what in ("e2e", "all"):
        e2e_microphones(args.force)
    if args.what in ("wer", "all"):
        wer_corpus(args.force)


if __name__ == "__main__":
    main()
