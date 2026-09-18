FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip espeak-ng && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY kokoro_service/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 -c "from kokoro import KPipeline; KPipeline(lang_code='a')"
COPY __init__.py gpu/__init__.py
COPY kokoro_service gpu/kokoro_service
EXPOSE 8002
CMD ["uvicorn", "gpu.kokoro_service.app:app", "--factory", "--host", "0.0.0.0", "--port", "8002"]
