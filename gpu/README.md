# GPU node

The self-hosted pipeline: vLLM (LLM), `whisper_service` (streaming STT over WebSocket) and
`kokoro_service` (streaming TTS over HTTP), all on **one** card. Plain HTTP/WebSocket, so the
deployment target stays open.

| Service | Port | Health that exercises the model |
|---|---|---|
| vLLM | 8000 | `GET /health` + the API's `/health/deep?pipeline=selfhosted` (one-token generation) |
| Whisper | 8001 | `GET /health/deep` |
| Kokoro | 8002 | `GET /health/deep` |
| DCGM exporter | 9400 | `/metrics` (VMs only) |

## RunPod (build and benchmark)

Iterate on **Community Cloud**; run every published benchmark on **Secure Cloud**. A noisy
neighbour silently poisons p95.

1. Create a pod: 1x L40S, a PyTorch/CUDA 12.4 template, 60 GB volume, expose HTTP ports
   8000, 8001, 8002. To serve another model, set `SERVING_CONFIG` (a file name in
   `config/serving/`) in the pod's environment, so vLLM and the harness read the same one.
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
```

## Pointing the local stack at the GPU node

`make up GPU_HOST=<ip or hostname>` (or `GPU_HOST` in `.env`) sends the API and agent
containers' self-hosted traffic there; the default is the machine running compose.
`make gpu-targets GPU_HOST=<ip or hostname>` points Prometheus at the same node.

## TTS engines

`kokoro_service` routes each request to an engine by its voice id: `<engine>:<voice>` goes
to that engine, a bare id to Kokoro. `TTS_ENGINES` (default `kokoro`) says which engines a
process loads; `GET /v1/info` reports them, and every engine listed in `SYNTHESIZERS` needs
an entry in `config/components.yaml` or the API refuses to start.

| Engine | Voices | Runtime |
|---|---|---|
| `kokoro` | Kokoro's own ids, e.g. `af_heart` | `kokoro` on CPU or CUDA |
| `qwen3-tts` | `qwen3-tts:<id>` from `kokoro_service/voices.yaml` | mlx-audio, Apple Silicon only |

Qwen3-TTS gives the German persona a German voice (`qwen3-tts:clara_de`). The voice is a
written description fed to the VoiceDesign model, not a cloned recording, so no speaker
consent is involved; edit `voices.yaml` to change how Clara sounds. Its weights are ~3 GB,
so a deployment that leaves `TTS_ENGINES` alone never downloads them.

## Known gaps

- **Qwen3-TTS runs on Apple Silicon only.** `qwen-tts` does not stream and pins an older
  transformers, so streaming comes from mlx-audio, which needs Metal. The CUDA backend
  (vLLM-Omni) is #32; until it lands, a GPU node can only serve German through Kokoro's
  English voices.
