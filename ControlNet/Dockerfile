FROM docker.io/nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /workspace/requirements.txt

# Copy ControlNet code
COPY . /workspace

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "/workspace/scripts/infer.py"]