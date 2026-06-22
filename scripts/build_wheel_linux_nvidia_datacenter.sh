#!/usr/bin/env bash
set -euo pipefail

# Build llama-cpp-python wheel for Linux NVIDIA Datacenter GPUs (CUDA backend, x86_64)
# Target: V100, A100, H100 (Volta, Ampere, Hopper datacenter GPUs)
# CUDA Compute Capabilities: 70, 80, 90

LLAMA_CPP_VERSION="${1:-0.3.19}"
OUTPUT_DIR="${2:-./wheels}"
CUDA_VERSION="${3:-12.6}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-detect platform - native build only
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64)
    WHEEL_PLATFORM="linux_x86_64"
    CONTAINER_PLATFORM="linux/amd64"
    ;;
  arm64|aarch64)
    WHEEL_PLATFORM="linux_aarch64"
    CONTAINER_PLATFORM="linux/arm64"
    ;;
  *)
    echo "❌ Unsupported architecture: $ARCH"
    exit 1
    ;;
esac

echo "🏢 Building llama-cpp-python ${LLAMA_CPP_VERSION} wheel (Linux NVIDIA Datacenter)"
echo "   Backend: CUDA ${CUDA_VERSION}"
echo "   Platform: ${CONTAINER_PLATFORM}"
echo "   GPU Support: V100, A100, H100, Grace Hopper (DGX, cloud)"
echo "   CUDA Architectures: 70, 80, 90"
echo "   Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR_ABS="$(cd "$OUTPUT_DIR" && pwd)"

# Build wheel in container with CUDA
echo "⏳ Building wheel with CUDA ${CUDA_VERSION} backend (15-25 minutes)..."
echo ""

podman run --rm \
  --platform "$CONTAINER_PLATFORM" \
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
    echo "Target GPUs: V100, A100, H100 (datacenter/cloud)"
    echo "CUDA Compute Capabilities: 70, 80, 90"
    CMAKE_ARGS="\
      -DGGML_CUDA=on \
      -DCMAKE_CUDA_ARCHITECTURES=70;80;90 \
      -DGGML_STATIC=off \
      -DBUILD_SHARED_LIBS=on \
    " FORCE_CMAKE=1 \
      pip3 wheel --no-binary=llama-cpp-python \
        "llama-cpp-python==${LLAMA_CPP_VERSION}" \
        -w /output

    echo "✅ Wheel built successfully"
  '

WHEEL_FILE="$(ls -1 "$OUTPUT_DIR_ABS"/llama_cpp_python-*-${WHEEL_PLATFORM}.whl 2>/dev/null | head -n1)"

if [[ ! -f "$WHEEL_FILE" ]]; then
  echo "❌ Wheel file not found in $OUTPUT_DIR_ABS"
  exit 1
fi

WHEEL_NAME="$(basename "$WHEEL_FILE")"
WHEEL_SIZE="$(du -h "$WHEEL_FILE" | cut -f1)"

echo ""
echo "✅ Linux NVIDIA Datacenter wheel built successfully!"
echo "   File: $WHEEL_NAME"
echo "   Size: $WHEEL_SIZE"
echo "   Path: $WHEEL_FILE"
echo "   Backend: CUDA ${CUDA_VERSION} (V100, A100, H100)"
echo ""
echo "📤 Next step: Upload to GitHub release"
echo "   Run: $SCRIPT_DIR/upload_wheel.sh $LLAMA_CPP_VERSION $WHEEL_FILE cuda-datacenter"
