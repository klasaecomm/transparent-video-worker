# Transparent product-video worker: FastAPI + ffmpeg(libvpx-vp9) + rembg.
FROM python:3.11-slim

# ffmpeg (with libvpx for VP9 alpha) + runtime libs for onnxruntime/opencv used by rembg
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the rembg model into the image so the pod starts ready (no cold fetch).
ENV REMBG_MODEL=u2netp
RUN python -c "from rembg import new_session; new_session('u2netp')"

COPY app.py .
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "120"]
