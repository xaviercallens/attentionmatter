# Adaptive Attention Token Reduction PoC — GPU Benchmark Container
# Base: NVIDIA CUDA 12.1 + Python 3.11 (Ubuntu 22.04)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip python3.11-dev \
    git curl wget && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Working directory
WORKDIR /workspace

# Install Python dependencies
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Pre-download embedding model (small, ~90MB)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy source code
COPY src/ ./src/
COPY run_poc.py .
COPY benchmark.sh .

# Default: run the full benchmark
CMD ["bash", "benchmark.sh"]
