# ─── Stage 1: Build ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY app/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ─── Stage 2: Copy project ────────────────────────────────────────────────────
# Source code
COPY src/   /app/src/
COPY app/   /app/app/
COPY models/ /app/models/

# Sample images for Gradio examples (small subset)
COPY dataset/images/Fractured/IMG0000019.jpg    /app/dataset/images/Fractured/
COPY dataset/images/Fractured/IMG0000025.jpg    /app/dataset/images/Fractured/
COPY dataset/images/Fractured/IMG0000044.jpg    /app/dataset/images/Fractured/
COPY dataset/images/Non_fractured/IMG0000000.jpg /app/dataset/images/Non_fractured/
COPY dataset/images/Non_fractured/IMG0000001.jpg /app/dataset/images/Non_fractured/
COPY dataset/images/Non_fractured/IMG0000002.jpg /app/dataset/images/Non_fractured/

# ─── Runtime ──────────────────────────────────────────────────────────────────
EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

WORKDIR /app/app
CMD ["python", "app.py"]
