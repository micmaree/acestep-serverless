# ============================================================
# ACE-Step XL serverless worker for RunPod
# Based on ACE-Step's official Dockerfile (CUDA 12.8 + uv sync frozen)
# Adds handler.py with runpod SDK to wrap acestep-api as a serverless worker.
# Models live on attached network volume (ACESTEP_CHECKPOINT_DIR=/runpod-volume/...)
# so the image stays slim (~3GB vs ~22GB if models baked in).
# ============================================================

ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.11

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

# ===== System deps =====
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common build-essential \
        git curl wget \
        libsndfile1 libsndfile1-dev ffmpeg \
        libffi-dev libssl-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# ===== uv (fast pip) =====
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ===== Clone ACE-Step-1.5 + uv sync (exact pinned versions) =====
WORKDIR /app
RUN git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git . \
    && uv sync --frozen --no-dev --python python3.11

# ===== Add the RunPod handler =====
RUN /bin/sh -c '. /app/.venv/bin/activate && uv pip install runpod hf_transfer'

COPY handler.py /app/handler.py

# ===== Runtime config =====
# Models live on the attached network volume (RunPod mounts at /runpod-volume).
ENV ACESTEP_CHECKPOINT_DIR=/runpod-volume/models/acestep
ENV ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo
ENV ACESTEP_QUANTIZATION=none
ENV ACESTEP_OFFLOAD_TO_CPU=false
ENV ACESTEP_INIT_LLM=true
ENV ACESTEP_LM_BACKEND=pt
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME=/runpod-volume/.hf_cache

# RunPod serverless entry — handler.py reads stdin for jobs
CMD ["/app/.venv/bin/python", "/app/handler.py"]
