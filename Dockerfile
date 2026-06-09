FROM python:3.11-slim AS cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WHISPER_DOWNLOAD_ROOT=/models

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app.py ./app.py
COPY templates ./templates

RUN mkdir -p /app/uploads /models

EXPOSE 5555

CMD ["python", "app.py"]


FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WHISPER_DEVICE=cuda \
    WHISPER_DOWNLOAD_ROOT=/models

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r requirements.txt

COPY app.py ./app.py
COPY templates ./templates

RUN mkdir -p /app/uploads /models

EXPOSE 5555

CMD ["python", "app.py"]
