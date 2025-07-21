# Rust ASR Project Guide

## 🐳 Build Docker Image

```bash
docker build --no-cache --platform=linux/arm64 -t YourDockerAccountName/DockerName .
```

## 🚪 Run into a Container and Mount Project

```bash
docker run -it \
  --name YourContainerName \
  -v $PWD/../rust_asr:/rust_asr \
  --net=host \
  -w /rust_asr \
  YourDockerAccountName/DockerName /bin/bash
```

## 🧪 Run the Rust ASR Demo

### 1. Convert PyTorch Model to ONNX

```bash
cd onnx
./bash/run.sh
```

### 2. Download Test Data

```bash
cd data
./get_hello_world_wave.py
```

## 🛠️ Build and Run Rust Inference Demo

```bash
./run_asr.sh
```

## ✅ Check Code Format Before Commit (Using pre-commit)

### 1. Install pre-commit

```bash
pip install pre-commit
```

### 2. Install Hook into .git/hooks/pre-commit

```bash
pre-commit install
```

### 3. Check All Files

```bash
pre-commit run --all-files
```
