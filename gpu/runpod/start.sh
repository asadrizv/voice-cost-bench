#!/usr/bin/env bash
# RunPod pod entry point: vLLM, Whisper and Kokoro in one container on one GPU, which is
# the colocation the benchmark is about. Expose ports 8000-8002 (HTTP) in the pod template,
# plus 8003/8004 when a CUDA speech engine is selected (see gpu/README.md).
# Pods can't run DCGM exporter; GPU numbers come from NVML via the harness instead.
set -euo pipefail
cd /workspace/voice-cost-bench

# The container disk is wiped when the pod stops; the /workspace volume is not.
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
serving="config/serving/${SERVING_CONFIG:-qwen-9b-l40s.yaml}"
if [[ ! -f "$serving" ]]; then
  echo "SERVING_CONFIG=${SERVING_CONFIG:-}: no file at $serving; name a file in config/serving/" >&2
  exit 1
fi

# A pod bills by the second, so nothing here waits forever: a server that died on a bad
# flag or an OOM never answers /health, and the reason is in its log rather than on stdout.
ready() {
  local name="$1" url="$2" log="$3" deadline=$((SECONDS + ${READY_TIMEOUT_S:-900}))
  until curl -sf "$url" >/dev/null; do
    if (( SECONDS > deadline )); then
      echo "$name did not answer $url within ${READY_TIMEOUT_S:-900}s; last of $log:" >&2
      tail -n 40 "$log" >&2 || true
      exit 1
    fi
    sleep 5
  done
  echo "$name ready"
}

whisper_backend="${WHISPER_BACKEND:-faster-whisper}"
tts_engines="${TTS_ENGINES:-kokoro}"
voxtral_model="${VOXTRAL_VLLM_MODEL:-mistralai/Voxtral-Mini-4B-Realtime-2602}"
qwen3_tts_model="${QWEN3_TTS_VLLM_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign}"

if ! command -v espeak-ng >/dev/null; then
  apt-get update -qq && apt-get install -y -qq --no-install-recommends espeak-ng
fi
python3 -m pip install -q -r gpu/whisper_service/requirements.txt -r gpu/kokoro_service/requirements.txt
python3 -m pip install -q "vllm==0.29.0" uv
# The harness runs on the pod too (`make benchmark`), and reads the GPU through NVML.
uv sync --frozen --extra gpu

if [[ "$whisper_backend" == voxtral-vllm ]]; then
  python3 -m pip install -q "mistral-common[audio]>=1.9.0"
fi
if [[ "$tts_engines" == *qwen3-tts-vllm* ]]; then
  # vllm-omni pins the vLLM release it is built on, and this pod pins another for the LLM,
  # so the omni server gets an environment of its own.
  python3 -m venv /workspace/omni-venv
  /workspace/omni-venv/bin/pip install -q "vllm-omni==0.28.0"
fi

# Download every model before starting any, so no health wait below includes a download.
python3 - "$serving" "$whisper_backend" "$tts_engines" "$voxtral_model" "$qwen3_tts_model" <<'PY'
import os
import sys

import yaml
from faster_whisper import download_model
from huggingface_hub import snapshot_download
from kokoro import KPipeline

serving_path, whisper_backend, tts_engines, voxtral_model, qwen3_tts_model = sys.argv[1:6]
with open(serving_path) as serving_file:
    serving = yaml.safe_load(serving_file)
snapshot_download(serving["model"], revision=serving.get("revision"))
if whisper_backend == "voxtral-vllm":
    snapshot_download(voxtral_model)
else:
    download_model(os.environ.get("WHISPER_MODEL", "large-v3-turbo"))
if "qwen3-tts-vllm" in tts_engines:
    snapshot_download(qwen3_tts_model)
if "kokoro" in tts_engines:
    KPipeline(lang_code="a")
PY

vllm serve --config "$serving" --port 8000 > /workspace/vllm.log 2>&1 &
# Start the small models after vLLM has claimed its gpu-memory-utilization share.
ready llm http://localhost:8000/health /workspace/vllm.log

# Each speech server claims a share of the same card; what is left for the LLM is what
# SERVING_CONFIG asks for, so these fractions and that file are set against each other.
# vLLM-Omni's deploy config gives 0.3 of the card to each of two stages besides.
if [[ "$whisper_backend" == voxtral-vllm ]]; then
  vllm serve "$voxtral_model" --tokenizer-mode mistral --enforce-eager --port 8003 \
    --gpu-memory-utilization "${VOXTRAL_GPU_FRACTION:-0.34}" > /workspace/voxtral.log 2>&1 &
  ready voxtral http://localhost:8003/health /workspace/voxtral.log
fi
if [[ "$tts_engines" == *qwen3-tts-vllm* ]]; then
  /workspace/omni-venv/bin/vllm serve "$qwen3_tts_model" --omni --trust-remote-code \
    --enforce-eager --port 8004 --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --gpu-memory-utilization "${QWEN3_TTS_GPU_FRACTION:-0.30}" > /workspace/qwen3-tts.log 2>&1 &
  ready qwen3-tts http://localhost:8004/health /workspace/qwen3-tts.log
fi

uvicorn gpu.whisper_service.app:app --factory --host 0.0.0.0 --port 8001 > /workspace/whisper.log 2>&1 &
uvicorn gpu.kokoro_service.app:app --factory --host 0.0.0.0 --port 8002 > /workspace/kokoro.log 2>&1 &
ready stt http://localhost:8001/health/deep /workspace/whisper.log
ready tts http://localhost:8002/health/deep /workspace/kokoro.log
echo "every selected service warm"
wait
