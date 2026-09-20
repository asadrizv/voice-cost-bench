# Running the L40S benchmark (#10)

The numbers this repo publishes are estimates until this runs. One session produces
`results/benchmark.json`: per-level cost and latency, a row per timed turn, the breaking
point, the utilisation curve, and provenance naming every component and weight.

Budget **about an hour of pod time, ~$1.10** at the $1.09/hour L40S Secure Cloud rate in
`config/rates.yaml`. The sweep itself is 10-18 minutes; the rest is downloads and start-up.

## 1 — The pod

RunPod **Secure Cloud**, one **L40S** (48 GiB), a CUDA 12.4+ PyTorch template.

- **Volume:** 100 GB at `/workspace`. The container disk is wiped on stop; the volume is not,
  so a second session skips the downloads.
- **Ports:** 8000-8002. Add 8003/8004 only if you run a CUDA speech engine (you do not need
  them for this run — see *What this run does not cover*).
- **Environment:** `HF_TOKEN` if any weight is gated. Nothing else is required; the defaults
  in `start.sh` are the configuration this benchmark is about.

## 2 — Start the stack

```bash
cd /workspace && git clone https://github.com/asadrizv/voice-cost-bench && cd voice-cost-bench
bash gpu/runpod/start.sh
```

It installs, downloads every weight **before** starting anything, then brings up vLLM
(port 8000), the STT service (8001) and the TTS service (8002). It prints `every selected
service warm` when all three answer their health checks.

Every wait is bounded by `READY_TIMEOUT_S` (default 900 s). If a server dies, the script
exits and prints the last 40 lines of that server's log rather than hanging on a health
check that will never pass — the logs are `/workspace/{vllm,whisper,kokoro}.log`.

**Check before spending the sweep:** the GPU budget this stack expects to fit.

```bash
curl -s localhost:8001/v1/info; curl -s localhost:8002/v1/info
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

`gpu_fraction` should be `null` for both (faster-whisper and Kokoro hold their weights;
nothing reserves a share). Used memory should be roughly the LLM's 0.72 of 48 GiB plus about
4 GiB — near 39 GiB, leaving headroom. If it is at the card's limit before a single call,
stop: the sweep will measure queueing, not the pipeline.

## 3 — The baseline, then the sweep

The sweep embeds the API pipeline as its comparison, so the baseline runs first. It needs
`OPENAI_API_KEY`, `DEEPGRAM_API_KEY` and `ELEVENLABS_API_KEY`, and it spends real vendor
money (three minutes of calls, cents). It does not touch the GPU, so you can run it on any
machine and copy `results/api_baseline.json` to the pod instead.

```bash
make baseline    # API pipeline, one call at a time, 180 s -> results/api_baseline.json
make benchmark   # self-hosted sweep 1/5/10/20/40, on until p95 > 900 ms -> results/benchmark.json
```

`make benchmark` passes `--forward`, which posts live metrics to an API server for Grafana.
No API runs on the pod and the forwarder swallows the failures, so this is harmless; drop
the flag if you want the log quiet.

## 4 — What to bring home

```bash
tar czf /workspace/benchmark-$(date +%F).tgz results/ /workspace/*.log
```

`results/benchmark.json` carries:

- `provenance` — every component with vendor, model, version, region and licence, as the
  services reported them; the weights' revisions; the GPU the harness read through NVML;
  and `gpu_memory_budget` with the arithmetic behind its verdict
- `levels[]` — per concurrency: cost per minute by stage and by carrier, latency percentiles
  per stage, caller-observed delay, WER, and **`turn_rows`**, one row per timed turn with its
  own latency, cost, usage and the concurrency it was billed against
- `breaking_point` and `utilisation_curve`

Check `harness_valid` on every level before publishing. It is false when the harness itself
fell behind real time, and a level that ran late measures the harness, not the stack.

## 5 — What this run does not cover

- **The CUDA speech engines.** `voxtral-vllm` and `qwen3-tts-vllm` are served by vLLM
  processes that reserve a fraction of the card each, and the budget check computes that
  they do not fit beside the LLM on one L40S — see *GPU memory on one card* in `README.md`.
  Selecting them here would measure an over-committed card. They need their own session, on
  a bigger card or with one engine at a time.
- **German WER.** The fixture corpus is read by English speakers (#9).
- **Anything about a second card**, which the L40S hourly rate does not price.
