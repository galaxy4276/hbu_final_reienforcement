# AWS SageMaker Custom Container for Atari RL Training
# Based on AWS Deep Learning Container for PyTorch

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker

LABEL maintainer="atari-rl-team"
LABEL description="Atari RL Training Container with Ray RLlib"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for Atari
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install Atari ROMs
RUN pip install autorom[accept-rom-license] && \
    AutoROM --accept-license

# Set working directory
WORKDIR /opt/ml/code

# Copy training scripts
COPY train_expert.py .
COPY record_data.py .
COPY train_offline.py .

# Default command
CMD ["python", "train_expert.py"]
