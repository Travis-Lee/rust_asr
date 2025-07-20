FROM --platform=linux/amd64 ubuntu:22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \  
    && dpkg-reconfigure -f noninteractive tzdata && \
    apt-get install -y \
    wget \
    vim \
    git \
    iputils-ping \
    cmake \
    xarclock \
    g++ \
    gcc \
    make \
    ffmpeg \
    espeak \
    libsndfile1 \
    curl \
    python3-dev \
    python3-venv \
    python3-pip \
    python3-tk \
    protobuf-compiler \
    libprotobuf-dev \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust（using rustup）
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

#Setting Environment Variables
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --upgrade pip
RUN pip install --no-cache-dir \
    tqdm \
    python-Levenshtein \
    librosa \
    onnx \
    onnx-simplifier \
    onnxruntime \
    numpy \
    pydub 

RUN python3 -c "import onnx; print('✅ ONNX Install Success，Version:', onnx.__version__)"

# Download ONNX Runtime 1.16.0 Linux Version 
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz && \
    tar -xzf onnxruntime-linux-x64-1.16.0.tgz && \
    rm onnxruntime-linux-x64-1.16.0.tgz

ENV LD_LIBRARY_PATH=/onnxruntime-linux-x64-1.16.0/lib

 
# Setting Working Directory
WORKDIR /RustASR

COPY .vimrc /root/.vimrc
RUN chmod 644 /root/.vimrc 
 
# Update pip and Install PyTorch
RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install matplotlib 
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Testing Install 
RUN python3 -c "import torch; print(f'PyTorch Version: {torch.__version__}')"
