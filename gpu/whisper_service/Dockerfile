FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY whisper_service/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
# Bake the weights into the image: a cold download at start would dominate the first call.
# The voxtral-vllm backend has none to bake; its weights live in the vLLM realtime server.
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')"
COPY __init__.py gpu/__init__.py
COPY whisper_service gpu/whisper_service
EXPOSE 8001
CMD ["uvicorn", "gpu.whisper_service.app:app", "--factory", "--host", "0.0.0.0", "--port", "8001"]
