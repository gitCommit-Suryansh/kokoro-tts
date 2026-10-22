FROM python:3.12-slim

WORKDIR /app

# libsndfile1 is required by soundfile; curl to fetch model weights at build time
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fetch Kokoro v1.0 model weights (~311MB) and voice pack (~27MB) at build time.
# Swap kokoro-v1.0.onnx for kokoro-v1.0.fp16.onnx below for a faster/slightly-lower-quality build.
RUN mkdir -p /models && \
    curl -L -o /models/kokoro-v1.0.fp16.onnx \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx && \
    curl -L -o /models/voices-v1.0.bin \
      https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

COPY app/ ./app/

ENV KOKORO_MODEL_PATH=/models/kokoro-v1.0.fp16.onnx
ENV KOKORO_VOICES_PATH=/models/voices-v1.0.bin

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
