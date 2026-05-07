FROM docker.io/nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list \
 && sed -i 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' /etc/apt/sources.list \
 && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 python3-pip git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r /workspace/requirements.txt

# Copy ControlNet code
COPY . /workspace

ENV PYTHONUNBUFFERED=1

# ENTRYPOINT ["python3", "/workspace/translate_simulation_pics.py"]