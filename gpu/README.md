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
   8000, 8001, 8002.
2. In the pod: `git clone <this repo> /workspace/voice-cost-bench && bash
   /workspace/voice-cost-bench/gpu/runpod/start.sh`
3. Run the harness **on the pod too** (colocation is the point, and it keeps the network
   out of the numbers): `curl -LsSf https://astral.sh/uv/install.sh | sh && make benchmark`.
4. Stop the pod when done. Nothing here needs it left running.

Pods have no Docker daemon, which is why `runpod/start.sh` exists alongside
`docker-compose.gpu.yml`.

## Any GPU VM with Docker (client-style deployment)

```bash
docker compose -f gpu/docker-compose.gpu.yml up -d
```

## Known gaps

- **Kokoro has no German voice.** The German persona falls back to an English Kokoro voice
  on the self-hosted pipeline, which is not a fair comparison. For German, swap
  `kokoro_service` for a multilingual model (Chatterbox Multilingual or Orpheus) behind the
  same `/v1/synthesize` contract and record the model in provenance.
