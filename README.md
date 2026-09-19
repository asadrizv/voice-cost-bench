# Voice Cost Bench

One law-firm receptionist conversation, two interchangeable pipelines, live **cost per
minute** and **latency** for both:

| | STT | LLM | TTS |
|---|---|---|---|
| `api` | Deepgram Nova-3 | GPT-4o-mini | ElevenLabs Flash v2.5 |
| `selfhosted` | faster-whisper large-v3-turbo | Qwen3.5-9B on vLLM | Kokoro, or Qwen3-TTS for German |

The self-hosted pipeline runs all three models on one L40S. The load harness finds the
concurrency where its p95 latency breaks, and reports cost at stated utilisation.

## Quick start (local, API pipeline)

```bash
cp .env.example .env        # add DEEPGRAM_API_KEY, OPENAI_API_KEY, ELEVENLABS_API_KEY
make up                     # Postgres, LiveKit (dev), API, agent, frontend, Prometheus, Grafana
open http://localhost:3000
```

No keys yet? Set `SIMULATE_PROVIDERS=true` in `.env`: both pipelines then run on a
simulated GPU model and the whole stack works end to end. You hear a tone instead of a
voice, and every cost is fiction.

**Fully local, real voices, no keys (Apple Silicon):** the self-hosted pipeline runs on the
Mac itself: Qwen 3.5 9B via Ollama, Whisper via MLX, Kokoro on CPU and Qwen3-TTS via MLX.

```bash
brew install ollama livekit && ollama pull qwen3.5:9b
uv sync --extra local
```

Then run `ollama serve`, `livekit-server --dev`, the Whisper and Kokoro services
(`WHISPER_BACKEND=mlx uv run --extra local uvicorn gpu.whisper_service.app:app --factory
--port 8001`, and `TTS_ENGINES=kokoro,qwen3-tts uv run --extra local uvicorn
gpu.kokoro_service.app:app --factory --port 8002`, which downloads ~3 GB of Qwen3-TTS
weights the first time), and the API and agent with
`LLM_SERVER=ollama VLLM_BASE_URL=http://localhost:11434/v1 VLLM_MODEL=qwen3.5:9b
SELFHOSTED_ON_LOCAL_MACHINE=true`. Latency on a laptop is not a benchmark figure; the call
screen says so.

Without Docker: `make test-db migrate`, then `make dev-api`, `make dev-agent` and
`make dev-frontend` in three terminals, with a LiveKit server or a LiveKit Cloud project in
`.env`.

## 5-minute demo script

1. **Open http://localhost:3000.** The pipeline toggle reads *API stack*. Say what's
   running: Deepgram, GPT-4o-mini and ElevenLabs, the stack most people would assemble.
2. **Click Call and play the caller.** "Hi, I need to speak to someone about my landlord."
   Clara asks for your name, the matter, urgency, then offers Tuesday at ten.
   - Point at **Cost per minute** (the hero number) and **Where it went**: TTS dominates
     the API bill.
   - Point at **Latency**: the *Perceived delay* row runs from when you stop talking to
     when Clara starts. That is what a caller feels, and it includes the endpointing
     decision that time-to-first-token hides.
   - Talk over Clara mid-sentence. She stops (barge-in), and the turn is marked
     *interrupted*.
   - Say goodbye. Clara says goodbye back and hangs up herself; talk over her goodbye and
     she stays on the line.
3. **End the call and switch the toggle to *Self-hosted GPU*.** Run the same conversation.
   The stage breakdown becomes GPU share plus telephony. The per-minute number here is for
   *one* call on a whole GPU, which is the worst case, so say that out loud.
4. **History** (`/calls`): both calls side by side. Click one for the **cost attribution**
   table: units × rate = cost for every stage, including the GPU divisor. The same data
   is a public endpoint: `GET /calls/{id}/cost`.
5. **Benchmark** (`/benchmark`): cost per minute and p95 against concurrency on one L40S.
   Lead with the **breaking point**, then the cheapest level within budget, then the
   utilisation table: at 25% utilisation the GPU share is 4x, and at low volume
   self-hosting can lose. That is the honest version, and it answers the CTO's first
   question before they ask it.
6. Optional: **Grafana** at http://localhost:3001 shows the call live (dashboard
   *Voice Cost Bench*).

Reset between runs with the **Reset** button. It clears the panels; nothing is deleted.

## Producing the benchmark

Run on the GPU node, so the network stays out of the numbers (see `gpu/README.md`):

```bash
make baseline      # API pipeline, 1 call at a time, 3 min → results/api_baseline.json
make benchmark     # self-hosted: 1/5/10/20/40 concurrent, then onward until p95 > 900 ms
make wer-audio wer # STT word error rate, English and German, both pipelines
```

`results/benchmark.json` is what `/benchmark` renders. Every run carries a provenance
block: git SHA and dirty flag, model and pinned revision, vLLM version, serving-config hash,
GPU SKU/region/price, rate-card date and hash, fixture audio hash, persona hash, endpointer.
A level whose harness fell behind real time is flagged invalid rather than reported.
Each level also reports caller-observed delay (p50/p95/p99), timed from the fixture audio
and the agent's audio rather than our own endpointer, so it compares with Openbenchmarks'
TTFAB; the benchmark page says how it is measured.

`make benchmark-sim` runs the same sweep offline against a queueing model of the GPU. It
exercises the harness; its output is stamped `simulated` and the UI says so.

## Architecture

Clean architecture: `domain` (pure) ← `application` (ports, use cases) ← `infrastructure`
(adapters) ← `interfaces` (HTTP, LiveKit agent, CLI). No provider SDK is imported outside
`infrastructure/`. `interfaces/container.py` is the only place that picks adapters.

- **`CallSession`** runs a call end to end: audio in, VAD, endpointing, STT, LLM, TTS,
  audio out, barge-in. The LiveKit agent and the load harness drive the *same* object, so
  the benchmark measures what a caller hears.
- **`ConcurrencySupervisor`** is the single count of active GPU calls. It rejects calls
  above the ceiling and is the only source of the GPU cost divisor. Each call accrues
  1/n of every second, so shares always sum to the busy time. The agent runs jobs as
  threads so all calls share one supervisor.
- **Endpointing** is a port with three implementations: fixed silence (baseline);
  semantic, which waits 250 ms after a finished sentence and up to 1.5 s after a dangling
  "and my…"; and Smart Turn v3.2, an 8 MB audio model (BSD-2) that scores the caller's
  last 8 seconds after a 200 ms pause and holds to the same 1.5 s ceiling when it says the
  turn is unfinished. Its weights download on first use and its per-decision inference
  time is reported by the harness.
- **Prices** live only in `config/rates.yaml`. Sampling parameters and the no-thinking
  switch live in the persona. vLLM flags live in `config/serving/<model>-<gpu>.yaml`.

## No call audio is stored

Caller and agent audio exist only in memory for the length of a call. The database keeps
transcripts, timings and costs, and live events carry the same. No audio goes to disk,
the database or the event stream. In Germany, recording someone's spoken words without
consent is a criminal offence (§201 StGB). Transcribing live without keeping the audio is
the position a law firm can defend.

Two tests keep this true: a full call through `CallSession` must leave no audio in the
repository or in emitted events, and the schema test fails if any table gains a binary
column. On the `api` pipeline, audio is streamed to Deepgram and ElevenLabs, so their
retention terms also apply.

## Tests

```bash
make test    # unit + contract + integration + Postgres; 90% coverage gate on domain/application
make lint    # ruff, mypy --strict (domain, application), tsc
make e2e     # Playwright: Chromium's fake mic plays a fixture WAV into a real call
make test-paid  # one real request per paid provider (cents)
```

Contract suites run the same tests against both implementations of each port: Deepgram
against a server speaking its protocol from fixtures, Whisper and both TTS engines against
the real service code with the models stubbed, and OpenAI and vLLM over recorded SSE. Test
runs cost nothing. Tests that need real weights skip unless those weights are already
cached, so a clean checkout downloads nothing.

## Caveats to state in any client conversation

- **Rates were last verified 2026-09-17** (`config/rates.yaml`). Re-check before quoting;
  L40S prices move fast.
- **Telephony is in both pipelines** and becomes the largest line item once the GPU is
  shared. It's identical on both sides, so it narrows the percentage saving. The selected
  carrier in `config/rates.yaml` (Twilio, $0.014/min) prices it; the benchmark also shows
  cost per minute under each other listed carrier. Telnyx and sipgate are unverified.
- **German self-hosted TTS runs on Apple Silicon only.** Kokoro has no German voice, so
  the German persona uses Qwen3-TTS, which streams through mlx-audio. A CUDA backend is
  still to come (#32), so a German comparison on the L40S is not yet fair.
- **The WER corpus is synthetic speech** (`fixtures/wer/README.md`). It's fine for
  regression and head-to-head comparison, not for a published German WER.
- **Quote loaded cost at a stated utilisation and concurrency**, never a bare per-minute
  figure (see the utilisation table).
