# GPU node

The self-hosted pipeline: vLLM (LLM), `whisper_service` (streaming STT over WebSocket) and
`kokoro_service` (streaming TTS over HTTP), all on **one** card. Plain HTTP/WebSocket, so the
deployment target stays open.

| Service | Port | Health that exercises the model |
|---|---|---|
| vLLM | 8000 | `GET /health` + the API's `/health/deep?pipeline=selfhosted` (one-token generation) |
| Whisper | 8001 | `GET /health/deep` |
| Kokoro | 8002 | `GET /health/deep` |
| vLLM realtime (Voxtral) | 8003 | `GET /health`, and the STT service's `/health/deep` through it |
| vLLM-Omni (Qwen3-TTS) | 8004 | `GET /health`, and the TTS service's `/health/deep` through it |
| DCGM exporter | 9400 | `/metrics` (VMs only) |

The last two run only when a CUDA speech engine is selected, and they do not fit beside the
LLM at its shipped share: read **GPU memory on one card** below before enabling either.

## RunPod (build and benchmark)

Iterate on **Community Cloud**; run every published benchmark on **Secure Cloud**. A noisy
neighbour silently poisons p95.

1. Create a pod: 1x L40S, a PyTorch/CUDA 12.4 template, 60 GB volume, expose HTTP ports
   8000, 8001, 8002 (and 8003/8004 for the CUDA speech engines). To serve another model,
   set `SERVING_CONFIG` (a file name in `config/serving/`) in the pod's environment, so
   vLLM and the harness read the same one.
2. In the pod: `git clone <this repo> /workspace/voice-cost-bench && bash
   /workspace/voice-cost-bench/gpu/runpod/start.sh`. It installs espeak-ng, the services,
   and the harness with its `gpu` extra (NVML telemetry), and downloads the weights to
   `HF_HOME` (default `/workspace/hf-cache`, on the volume) before starting anything.
3. Run the harness **on the pod too** (colocation is the point, and it keeps the network
   out of the numbers): `cd /workspace/voice-cost-bench && make benchmark`.
4. Stop the pod when done. Nothing here needs it left running.

Pods have no Docker daemon, which is why `runpod/start.sh` exists alongside
`docker-compose.gpu.yml`.

## Any GPU VM with Docker (client-style deployment)

```bash
docker compose -f gpu/docker-compose.gpu.yml up -d   # SERVING_CONFIG=<file in config/serving/>
# with the CUDA speech engines, each behind its own profile:
WHISPER_BACKEND=voxtral-vllm TTS_ENGINES=kokoro,qwen3-tts-vllm \
  docker compose -f gpu/docker-compose.gpu.yml --profile voxtral --profile qwen3-tts up -d
```

The speech servers hold the weights; `whisper_service` and `kokoro_service` only speak to
them, and restart until they answer rather than declaring a dependency on a profile that
may be off.

## Pointing the local stack at the GPU node

`make up GPU_HOST=<ip or hostname>` (or `GPU_HOST` in `.env`) sends the API and agent
containers' self-hosted traffic there; the default is the machine running compose.
`make gpu-targets GPU_HOST=<ip or hostname>` points Prometheus at the same node.

## STT engines

`WHISPER_BACKEND` (default `faster-whisper`) says which backend `whisper_service` loads; an
id it cannot run is refused at startup, `GET /v1/info` reports the engine it loaded, and
every engine listed in `TRANSCRIBERS` needs an entry in `config/components.yaml` or the API
refuses to start. The WebSocket protocol is the same whichever is loaded.

| `WHISPER_BACKEND` | Engine reported | Model | Runtime |
|---|---|---|---|
| `faster-whisper` | `faster-whisper` | `WHISPER_MODEL`, default `large-v3-turbo` | faster-whisper on CUDA |
| `mlx` | `mlx` | `WHISPER_MLX_REPO` | mlx-whisper, Apple Silicon only |
| `voxtral` | `voxtral` | `VOXTRAL_MLX_REPO` | mlx-audio, Apple Silicon only |
| `voxtral-vllm` | `voxtral` | `VOXTRAL_VLLM_MODEL` | vLLM realtime API on CUDA, at `VOXTRAL_VLLM_URL` |

Whisper re-transcribes the utterance buffer for every result; Voxtral decodes as the caller
speaks, at a 480 ms transcription delay, and detects the language itself. Its 4-bit weights
are ~3.1 GB, downloaded on first load, so a deployment that leaves `WHISPER_BACKEND` alone
never fetches them.

The two Voxtral backends are one engine on two runtimes, so both report `voxtral` at
`/v1/info` and both are described by the one catalogue entry. The reported *model* names
the runtime -- `mistralai/Voxtral-Mini-4B-Realtime-2602 (vLLM realtime)` -- because that
entry's `version` is fixed configuration, and a benchmark run has to say which runtime
produced its numbers. `voxtral-vllm` holds nothing on the card itself: it opens a WebSocket
to vLLM's `/v1/realtime` per utterance, sends base64 PCM16 at 16 kHz as the caller speaks,
and commits with `final: true` on a flush. If that server is unreachable the service fails
its warm-up and refuses to start, rather than answering every flush with silence. The 480 ms
transcription delay above is the mlx-audio path's setting: a vLLM realtime session carries
only a model, so the CUDA path runs at whatever delay its server was started with. It does
serve `WHISPER_WORKERS` calls at once, where the Apple Silicon engine serves one: that
engine's decoder state lives on the thread its model was built on, and a socket's does not.

### Voxtral against Whisper, measured (#29)

`tests/integration/test_voxtral_stt_model.py` drives both Apple Silicon engines through
this service's socket at caller pace and prints these; rerun it rather than trusting the
table. One M5 Pro, `whisper-large-v3-turbo` against `Voxtral-Mini-4B-Realtime-2602-4bit`,
on the six scripted intake turns. **The fixture audio is synthetic**, so the error rates
compare the two engines and are not publishable numbers (`fixtures/wer/README.md`).

| | mean WER | median time to final | worst |
|---|---|---|---|
| English, `mlx` | 0.065 | 186 ms | 296 ms |
| English, `voxtral` | 0.065 | 511 ms | 653 ms |
| German, `mlx` | 0.143 | 179 ms | 284 ms |
| German, `voxtral` | 0.143 | 516 ms | 747 ms |

Turn for turn the two transcripts differ only in spelling out an e-mail address, so on this
corpus Voxtral buys no accuracy, and it answers a flush roughly 330 ms later because closing
its session drains the transcription delay it deliberately runs behind. What it does buy is
a bounded cost per second of speech: Whisper's final grows with the utterance (182 ms at
4.3 s, 500 ms at 16.9 s) while Voxtral's barely moves (456 ms, 683 ms), and its interims are
a running transcript rather than a whole re-transcription. For receptionist turns of a few
seconds, Whisper is the faster engine on this hardware. The same comparison on CUDA needs a
GPU to run the `voxtral-vllm` backend on (#10).

## TTS engines

`kokoro_service` routes each request to an engine by its voice id: `<engine>:<voice>` goes
to that engine, a bare id to Kokoro. `TTS_ENGINES` (default `kokoro`) says which engines a
process loads; `GET /v1/info` reports them, and every engine listed in `SYNTHESIZERS` needs
an entry in `config/components.yaml` or the API refuses to start.

| `TTS_ENGINES` entry | Engine reported | Voices | Runtime |
|---|---|---|---|
| `kokoro` | `kokoro` | Kokoro's own ids, e.g. `af_heart` | `kokoro` on CPU or CUDA |
| `qwen3-tts` | `qwen3-tts` | `qwen3-tts:<id>` from `kokoro_service/voices.yaml` | mlx-audio, Apple Silicon only |
| `qwen3-tts-vllm` | `qwen3-tts` | the same ids | vLLM-Omni speech API on CUDA, at `QWEN3_TTS_VLLM_URL` |

Qwen3-TTS gives the German persona a German voice (`qwen3-tts:clara_de`). The voice is a
written description fed to the VoiceDesign model, not a cloned recording, so no speaker
consent is involved; edit `voices.yaml` to change how Clara sounds. Its weights are ~3 GB,
so a deployment that leaves `TTS_ENGINES` alone never downloads them.

As with Voxtral, the two Qwen3-TTS backends are one engine on two runtimes: both report
`qwen3-tts`, and the reported model names the runtime (`... (vLLM-Omni)`). A voice names the
engine that speaks it, so a process may load one runtime of an engine, not both; naming both
in `TTS_ENGINES` is refused at startup. `qwen3-tts-vllm` posts to `/v1/audio/speech` with
`stream_format: "audio"` and reads the raw PCM16 back at 24 kHz, and refuses a `speed` other
than 1.0, which vLLM-Omni does not stream.

## GPU memory on one card

**The CUDA speech engines do not fit beside the LLM on one L40S at its shipped share.**
They are served by vLLM processes, and a vLLM process does not take what its model weighs:
it reserves the fraction of the card you give it, up front, whether or not a call is in
flight. The budget in `config/components.yaml` counts model footprints, so it understates
what these runtimes hold.

On a 48 GiB L40S, with the shares this repo ships:

| What | Share | GiB |
|---|---|---|
| LLM, `gpu-memory-utilization: 0.72` in `config/serving/qwen-9b-l40s.yaml` | 0.72 | 34.56 |
| `voxtral-vllm` server, `VOXTRAL_GPU_FRACTION` | 0.34 | 16.32 |
| `qwen3-tts-vllm` server, `QWEN3_TTS_GPU_FRACTION` | 0.30 | 14.40 |
| `kokoro_service`, when Kokoro is loaded too | -- | ~1.0 |
| **Total** | | **~66.3, on a 48 GiB card** |

The two defaults are not padding: vLLM's own Voxtral recipe asks for a GPU with **>= 16 GiB**
for the bf16 weights, and vLLM-Omni's bundled `qwen3_tts.yaml` gives its first stage 0.3.
Something has to give, and the choice belongs to whoever runs the benchmark:

- **Shrink the LLM.** Both speech servers plus Kokoro leave about 16 GiB, which a 9B model
  in bf16 does not fit into; it would have to be a quantised or smaller checkpoint, and that
  is a different LLM in the comparison, not the same one served differently.
- **Serve speech from a second card**, and say so beside any cost number: the L40S hour in
  `config/rates.yaml` prices one card.
- **Run one CUDA speech engine, not both.** Voxtral with Kokoro's English voices, or
  Whisper with Qwen3-TTS German, each fits with an LLM share around 0.5.

This is the same wall #30's budget already hit for the Apple Silicon numbers: Qwen3-TTS
(5.48 GiB) beside Voxtral (9.98 GiB) and the LLM's 0.72 share come to 50.02 GiB, 2.02 GiB
over the card. Serving those two engines through vLLM does not narrow that gap -- it widens
it, because each server reserves a fraction rather than its weights. None of these figures
has been measured on an L40S (#10).

## Known gaps

- **Neither CUDA speech engine has been run on a GPU.** Both backends are contract-tested
  against servers that speak the documented vLLM and vLLM-Omni wire formats, and the compose
  file and `runpod/start.sh` are checked only by `docker compose config` and `bash -n`. No
  image has been pulled, no server started, no audio synthesised or transcribed on CUDA.
  Latency, quality and the memory figures above are all unverified.
- **The vLLM-Omni request shape is confirmed from documentation, not from a running server.**
  `stream_format: "audio"` returning raw PCM, the VoiceDesign `instructions` field and the
  capitalised `language` vocabulary come from vLLM-Omni's `docs/serving/speech_api.md` and
  its own client; the byte order of that PCM is stated nowhere and is read here as
  little-endian. `--deploy-config vllm_omni/deploy/qwen3_tts.yaml` is a repo-relative path in
  the upstream example, and whether it resolves inside `vllm/vllm-omni:v0.28.0` is untested.
- **A mid-stream failure of either server is only partly recoverable.** The TTS service pulls
  the first PCM chunk before committing a 200, so an engine that fails to start generating is
  refused with a status; one that dies after the first chunk can only truncate the stream,
  because the raw form carries no error frame. On the STT side an error event fails the
  utterance and breaks the caller's socket rather than answering the flush with a partial
  transcript.
