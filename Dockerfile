# Use a PyTorch CUDA development image as base
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies & build tools for KenLM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libboost-system-dev \
    libboost-thread-dev \
    libboost-program-options-dev \
    libboost-test-dev \
    libeigen3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Clone and compile KenLM binaries
RUN git clone https://github.com/kpu/kenlm.git /opt/kenlm && \
    mkdir -p /opt/kenlm/build && \
    cd /opt/kenlm/build && \
    cmake .. && \
    make -j$(nproc)

# Add KenLM bin folder to system PATH
ENV PATH="/opt/kenlm/build/bin:${PATH}"

# Install Python requirements
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    transformers==4.44.2 \
    datasets==2.21.0 \
    accelerate==0.34.2 \
    torchaudio \
    jiwer==3.0.4 \
    evaluate==0.4.2 \
    soundfile \
    librosa==0.10.2.post1 \
    pyctcdecode==0.5.0 \
    https://github.com/kpu/kenlm/archive/master.zip

# Set default working directory
WORKDIR /workspace

# Copy prepared training dataset CSVs and audio files directly into the container
COPY cim-wav2vec2-train.csv /workspace/cim-wav2vec2-train.csv
COPY cim-wav2vec2-valid.csv /workspace/cim-wav2vec2-valid.csv
COPY cim-wav2vec2-test.csv /workspace/cim-wav2vec2-test.csv
COPY sentence_audio /workspace/sentence_audio
COPY colab-script-rips/trainer_w2v2_local.py /workspace/trainer_w2v2_local.py

# Pre-download the base XLS-R model checkpoint to cache it in the image
RUN python3 -c "from transformers import Wav2Vec2ForCTC; Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-xlsr-53')"



