#!/usr/bin/env bash
# RunPod pod entry point: vLLM, Whisper and Kokoro in one container on one GPU, which is
# the colocation the benchmark is about. Expose ports 8000-8002 (HTTP) in the pod template.
# Pods can't run DCGM exporter; GPU numbers come from NVML via the harness instead.
set -euo pipefail
cd /workspace/voice-cost-bench

python3 -m pip install -q -r gpu/whisper_service/requirements.txt -r gpu/kokoro_service/requirements.txt
python3 -m pip install -q "vllm==0.29.0"

vllm serve --config "config/serving/${SERVING_CONFIG:-qwen-9b-l40s.yaml}" --port 8000 > /workspace/vllm.log 2>&1 &
# Start the small models after vLLM has claimed its gpu-memory-utilization share.
until curl -sf localhost:8000/health >/dev/null; do sleep 5; done
uvicorn gpu.whisper_service.app:app --factory --host 0.0.0.0 --port 8001 > /workspace/whisper.log 2>&1 &
uvicorn gpu.kokoro_service.app:app --factory --host 0.0.0.0 --port 8002 > /workspace/kokoro.log 2>&1 &
until curl -sf localhost:8001/health/deep >/dev/null && curl -sf localhost:8002/health/deep >/dev/null; do sleep 3; done
echo "all three services warm"
wait
