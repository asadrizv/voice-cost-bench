"""Smart Turn v3.2: a Whisper-Tiny encoder with a linear head that scores whether a turn
is finished, run on CPU through ONNX Runtime. Weights and code are BSD-2-Clause.

Source: https://huggingface.co/pipecat-ai/smart-turn-v3. Its inference.py names a
`smart-turn-v3.1.onnx` the repository does not contain; the int8 CPU build below is the
file that is actually published.
"""

from __future__ import annotations

import functools
import logging
from concurrent.futures import Executor, ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from huggingface_hub import hf_hub_download

log = logging.getLogger(__name__)

MODEL_REPO = "pipecat-ai/smart-turn-v3"
MODEL_FILE = "smart-turn-v3.2-cpu.onnx"
MODEL_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
"""Pinned so a benchmark run records weights that cannot change under it."""

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 8 * SAMPLE_RATE
WINDOW_BYTES = WINDOW_SAMPLES * 2
N_FFT = 400
HOP_LENGTH = 160
MEL_BINS = 80
MEL_FRAMES = 800
INTRA_OP_THREADS = 4
"""Four threads put a decision at ~9 ms on an M-series Mac; one thread takes ~30 ms, more
than an audio frame. Kept below the core count so concurrent calls still get a share."""


class SmartTurnModel:
    """P(the caller has finished) for the last 8 s of 16 kHz mono PCM16, as a float in
    [0, 1]. Shorter audio is left-padded and longer audio keeps its end, as the model was
    trained. Safe to call from several threads: ONNX Runtime sessions are re-entrant."""

    def __init__(self, model_path: str) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = INTRA_OP_THREADS
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )

    def __call__(self, audio: bytes) -> float:
        samples = np.frombuffer(audio[-WINDOW_BYTES:], dtype=np.int16).astype(np.float32) / 32768
        samples = np.pad(samples, (WINDOW_SAMPLES - len(samples), 0))
        [[[probability]]] = self._session.run(None, {"input_features": log_mel_features(samples)})
        return float(probability)


def weights(offline: bool = False) -> str:
    """The weights file, downloaded into the Hugging Face cache the first time. With
    `offline`, raises rather than downloading when the cache does not hold them."""
    return hf_hub_download(
        MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION, local_files_only=offline
    )


@functools.cache
def load_model(offline: bool = False) -> SmartTurnModel:
    """Loads the model on the first call that selects Smart Turn, never at import."""
    path = weights(offline)
    log.info("Smart Turn v3.2 loaded from %s", path)
    return SmartTurnModel(path)


def probability(audio: bytes) -> float:
    """Loads the model on first use, on whichever thread runs the decision. Loading it
    where the detector is built would stall the agent's audio loop (147 ms measured for
    the import alone, plus the download on a host that has no cache yet)."""
    return load_model().probability(audio)


@functools.cache
def pool() -> Executor:
    """Where decisions run so the audio loop keeps reading frames while the model thinks.
    Two workers: one decision per caller pause is far below what they can absorb."""
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="smart-turn")


def log_mel_features(samples: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """The Whisper log-mel spectrogram the model was trained on, as
    `WhisperFeatureExtractor(chunk_length=8)` with `do_normalize=True` produces it: one
    batch of 80 mel bins over 800 frames."""
    normalised = (samples - samples.mean()) / np.sqrt(samples.var() + 1e-7)
    padded = np.pad(normalised, N_FFT // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(N_FFT + 1)[:-1], axis=-1)) ** 2
    mel = np.maximum(spectrum @ _mel_filters(), 1e-10)
    log_mel = np.log10(mel).T[:, :MEL_FRAMES]
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    return ((log_mel + 4.0) / 4.0)[np.newaxis].astype(np.float32)


@functools.cache
def _mel_filters() -> npt.NDArray[np.float64]:
    """Slaney-scaled triangular mel filter bank, matching transformers' `mel_filter_bank`
    with `norm="slaney"` and `mel_scale="slaney"`."""
    fft_frequencies = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    # 8 kHz sits above the Slaney scale's 1 kHz knee, so the top edge takes its log branch.
    top_mel = _MEL_LOG_MEL + float(np.log(SAMPLE_RATE / 2 / _MEL_LOG_HERTZ)) * _MEL_LOG_STEP
    edges = _mel_to_hertz(np.linspace(0, top_mel, MEL_BINS + 2))
    slopes = edges[np.newaxis, :] - fft_frequencies[:, np.newaxis]
    widths = np.diff(edges)
    falling = -slopes[:, :-2] / widths[:-1]
    rising = slopes[:, 2:] / widths[1:]
    filters = np.maximum(0.0, np.minimum(falling, rising))
    return filters * (2.0 / (edges[2:] - edges[:MEL_BINS]))[np.newaxis, :]


_MEL_LOG_HERTZ = 1000.0
_MEL_LOG_MEL = 15.0
_MEL_LOG_STEP = 27.0 / np.log(6.4)


def _mel_to_hertz(mel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    linear = mel * 200.0 / 3.0
    logarithmic = _MEL_LOG_HERTZ * np.exp((mel - _MEL_LOG_MEL) / _MEL_LOG_STEP)
    return np.where(mel < _MEL_LOG_MEL, linear, logarithmic)
