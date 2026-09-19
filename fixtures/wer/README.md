# WER corpus

`sentences.yaml` holds the reference transcripts: English and German receptionist-call
sentences, heavy on legal vocabulary (Kündigungsschutzklage, Nebenkostenabrechnung,
Eigenbedarfskündigung...). `make wer-audio` renders each sentence with several macOS voices
into `fixtures/audio/wer/` (not committed; ~20 MB). On Linux, espeak-ng stands in for `say`.

**These recordings are synthetic.** TTS audio is cleaner than a phone line and has no real
regional accent, so WER measured on it is optimistic and must not be published. It is
useful for regression (did a Whisper setting make things worse?) and for comparing the two
STT stacks against each other on identical input.

For the published German WER claim, replace `fixtures/audio/wer/de/` with recordings of
real speakers (Bavarian, Saxon, Swiss German, Austrian, plus L2 speakers), keep the same
file names, and rerun `make wer`. The harness does not care where the WAVs came from.
