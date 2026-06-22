#!/usr/bin/env bash
set -euo pipefail

# Build llama-cpp-python wheel for Linux NVIDIA Consumer GPUs (CUDA backend, x86_64)
# Target: RTX 20xx/30xx/40xx, GTX 1660 Ti (Turing, Ampere, Ada Lovelace)
# CUDA Compute Capabilities: 75, 86, 89

LLAMA_CPP_VERSION="${1:-0.3.19}"
OUTPUT_DIR="${2:-./wheels}"
CUDA_VERSION="${3:-12.6}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Must be x86_64
ARCH="$(uname -m)"
if [[ "$ARCH" != "x86_64" && "$ARCH" != "amd64" ]]; then
  echo "❌ This script must run on x86_64 architecture"
  echo "   Current architecture: $ARCH"
  exit 1
fi

echo "🎮 Building llama-cpp-python ${LLAMA_CPP_VERSION} wheel (Linux NVIDIA Consumer)"
echo "   Backend: CUDA ${CUDA_VERSION}"
echo "   Platform: linux/amd64"
echo "   GPU Support: RTX 20xx/30xx/40xx, GTX 1660 Ti"
echo "   CUDA Architectures: 75, 86, 89"
echo "   Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR_ABS="$(cd "$OUTPUT_DIR" && pwd)"

# Build wheel in container with CUDA
echo "⏳ Building wheel with CUDA ${CUDA_VERSION} backend (15-25 minutes)..."
echo ""

podman run --rm \
  --platform linux/amd64 \
  -v "$OUTPUT_DIR_ABS:/output" \
  -e "LLAMA_CPP_VERSION=$LLAMA_CPP_VERSION" \
  -e "CUDA_VERSION=$CUDA_VERSION" \
  docker.io/nvidia/cuda:${CUDA_VERSION}.1-devel-ubi9 \
  bash -c '
    set -euo pipefail

    echo "Installing build dependencies..."
    dnf -y install --setopt=install_weak_deps=False \
      python3 python3-pip python3-devel \
      gcc gcc-c++ cmake ninja-build make pkgconf-pkg-config

    dnf -y install ccache 2>/dev/null || echo "ccache not available, continuing without it"

    echo "Upgrading pip..."
    pip3 install --upgrade pip setuptools wheel

    echo "Building llama-cpp-python ${LLAMA_CPP_VERSION} with CUDA ${CUDA_VERSION} backend..."
    echo "Target GPUs: RTX 20xx/30xx/40xx, GTX 1660 Ti"
    echo "CUDA Compute Capabilities: 75, 86, 89"
    CMAKE_ARGS="\
      -DGGML_CUDA=on \
      -DCMAKE_CUDA_ARCHITECTURES=75;86;89 \
      -DGGML_STATIC=off \
      -DBUILD_SHARED_LIBS=on \
    " FORCE_CMAKE=1 \
      pip3 wheel --no-binary=llama-cpp-python \
        "llama-cpp-python==${LLAMA_CPP_VERSION}" \
        -w /output

    echo "✅ Wheel built successfully"
  '

WHEEL_FILE="$(ls -1 "$OUTPUT_DIR_ABS"/llama_cpp_python-*-linux_x86_64.whl 2>/dev/null | head -n1)"

if [[ ! -f "$WHEEL_FILE" ]]; then
  echo "❌ Wheel file not found in $OUTPUT_DIR_ABS"
  exit 1
fi

WHEEL_NAME="$(basename "$WHEEL_FILE")"
WHEEL_SIZE="$(du -h "$WHEEL_FILE" | cut -f1)"

echo ""
echo "✅ Linux NVIDIA Consumer wheel built successfully!"
echo "   File: $WHEEL_NAME"
echo "   Size: $WHEEL_SIZE"
echo "   Path: $WHEEL_FILE"
echo "   Backend: CUDA ${CUDA_VERSION} (RTX 20xx/30xx/40xx)"
echo ""
echo "📤 Next step: Upload to GitHub release"
echo "   Run: $SCRIPT_DIR/upload_wheel.sh $LLAMA_CPP_VERSION $WHEEL_FILE cuda-consumer"
