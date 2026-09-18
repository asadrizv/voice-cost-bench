"""Renders fixture audio with macOS `say`: conversations (committed) and the WER corpus
(regenerated on demand). 16 kHz mono PCM16, the format every STT adapter receives."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"


def render(text: str, voice: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "say",
            "-v",
            voice,
            "-o",
            str(out),
            "--file-format=WAVE",
            "--data-format=LEI16@16000",
            text,
        ],
        check=True,
    )


def conversations(force: bool) -> None:
    spec = yaml.safe_load((FIXTURES / "conversations.yaml").read_text())
    for name, conv in spec["conversations"].items():
        for i, text in enumerate(conv["turns"]):
            out = FIXTURES / "audio" / name / f"{i:02d}.wav"
            if force or not out.exists():
                render(text, conv["voice"], out)
        print(f"{name}: {len(conv['turns'])} turns")


def wer_corpus(force: bool) -> None:
    spec = yaml.safe_load((FIXTURES / "wer" / "sentences.yaml").read_text())
    for lang, block in spec.items():
        count = 0
        for v_index, voice in enumerate(block["voices"]):
            for s_index, text in enumerate(block["sentences"]):
                out = FIXTURES / "audio" / "wer" / lang / f"{s_index:03d}_v{v_index}.wav"
                if force or not out.exists():
                    render(text, voice, out)
                count += 1
        print(f"wer/{lang}: {count} utterances")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["conversations", "wer", "all"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if shutil.which("say") is None:
        sys.exit("needs macOS `say`; on Linux, use the committed conversation fixtures")
    if args.what in ("conversations", "all"):
        conversations(args.force)
    if args.what in ("wer", "all"):
        wer_corpus(args.force)


if __name__ == "__main__":
    main()
