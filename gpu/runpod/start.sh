#!/usr/bin/env bash
# RunPod pod entry point: vLLM, Whisper and Kokoro in one container on one GPU, which is
# the colocation the benchmark is about. Expose ports 8000-8002 (HTTP) in the pod template.
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

if ! command -v espeak-ng >/dev/null; then
  apt-get update -qq && apt-get install -y -qq --no-install-recommends espeak-ng
fi
python3 -m pip install -q -r gpu/whisper_service/requirements.txt -r gpu/kokoro_service/requirements.txt
python3 -m pip install -q "vllm==0.29.0" uv
# The harness runs on the pod too (`make benchmark`), and reads the GPU through NVML.
uv sync --frozen --extra gpu

# Download every model before starting any, so no health wait below includes a download.
python3 - "$serving" <<'PY'
import os
import sys

import yaml
from faster_whisper import download_model
from huggingface_hub import snapshot_download
from kokoro import KPipeline

with open(sys.argv[1]) as serving_file:
    serving = yaml.safe_load(serving_file)
snapshot_download(serving["model"], revision=serving.get("revision"))
download_model(os.environ.get("WHISPER_MODEL", "large-v3-turbo"))
KPipeline(lang_code="a")
PY

vllm serve --config "$serving" --port 8000 > /workspace/vllm.log 2>&1 &
# Start the small models after vLLM has claimed its gpu-memory-utilization share.
until curl -sf localhost:8000/health >/dev/null; do sleep 5; done
uvicorn gpu.whisper_service.app:app --factory --host 0.0.0.0 --port 8001 > /workspace/whisper.log 2>&1 &
uvicorn gpu.kokoro_service.app:app --factory --host 0.0.0.0 --port 8002 > /workspace/kokoro.log 2>&1 &
until curl -sf localhost:8001/health/deep >/dev/null && curl -sf localhost:8002/health/deep >/dev/null; do sleep 3; done
echo "all three services warm"
wait
