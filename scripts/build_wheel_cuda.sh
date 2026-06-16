#!/usr/bin/env bash
set -euo pipefail

# Build llama-cpp-python wheel with CUDA backend (for Linux NVIDIA x86_64)
# CUDA wheels are x86_64 only - ARM64 NVIDIA support is rare (Jetson embedded)
# For ARM64 use Vulkan wheel instead: ./scripts/build_wheel_vulkan.sh

LLAMA_CPP_VERSION="${1:-0.3.19}"
OUTPUT_DIR="${2:-./wheels}"
CUDA_VERSION="${3:-12.6}"
TARGET_PLATFORM="${4:-}"  # Optional: linux/arm64, linux/amd64 for cross-compilation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CUDA wheels are x86_64 only
# ARM64 CUDA exists but is rare (Jetson) and CUDA Docker images don't support it well
if [[ -n "$TARGET_PLATFORM" ]]; then
  case "$TARGET_PLATFORM" in
    linux/amd64|linux/x86_64)
      CONTAINER_PLATFORM="linux/amd64"
      WHEEL_PLATFORM="linux_x86_64"
      ;;
    linux/arm64|linux/aarch64)
      echo "❌ CUDA wheels for ARM64 are not supported"
      echo "   ARM64 NVIDIA GPUs are rare (mainly Jetson embedded)"
      echo "   For ARM64, use Vulkan wheel instead:"
      echo "   ./scripts/build_wheel_vulkan.sh $LLAMA_CPP_VERSION $OUTPUT_DIR linux/arm64"
      exit 1
      ;;
    *)
      echo "❌ Unsupported platform: $TARGET_PLATFORM"
      exit 1
      ;;
  esac
  echo "🔀 Building for platform: $CONTAINER_PLATFORM"
else
  # No platform specified - always build x86_64 CUDA wheel
  # Podman/Docker will handle emulation if running on ARM64 host
  WHEEL_PLATFORM="linux_x86_64"
  CONTAINER_PLATFORM="linux/amd64"

  ARCH="$(uname -m)"
  if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
    echo "ℹ️  Building x86_64 CUDA wheel on ARM64 host (using Podman VM emulation)"
  fi
fi

echo "🔧 Building llama-cpp-python ${LLAMA_CPP_VERSION} wheel (CUDA backend)"
echo "   Output: $OUTPUT_DIR"
echo "   Platform: ${CONTAINER_PLATFORM}"
echo "   CUDA: ${CUDA_VERSION}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR_ABS="$(cd "$OUTPUT_DIR" && pwd)"

# Build wheel in container with CUDA
echo "⏳ Building wheel with CUDA ${CUDA_VERSION} backend (15-30 minutes)..."
echo ""

podman run --rm \
  --platform "$CONTAINER_PLATFORM" \
  -v "$OUTPUT_DIR_ABS:/output" \
  -e "LLAMA_CPP_VERSION=$LLAMA_CPP_VERSION" \
  -e "CUDA_VERSION=$CUDA_VERSION" \
  nvidia/cuda:${CUDA_VERSION}.1-devel-ubi9 \
  bash -c '
    set -euo pipefail

    echo "Installing build dependencies..."
    dnf -y install --setopt=install_weak_deps=False \
      python3 python3-pip python3-devel \
      gcc gcc-c++ cmake ninja-build make pkgconf-pkg-config

    # ccache is optional, not available on UBI9
    dnf -y install ccache 2>/dev/null || echo "ccache not available, continuing without it"

    echo "Upgrading pip..."
    pip3 install --upgrade pip setuptools wheel

    echo "Building llama-cpp-python ${LLAMA_CPP_VERSION} with CUDA ${CUDA_VERSION} backend..."
    CMAKE_ARGS="\
      -DGGML_CUDA=on \
      -DCMAKE_CUDA_ARCHITECTURES=all \
    " FORCE_CMAKE=1 \
      pip3 wheel --no-binary=llama-cpp-python \
        "llama-cpp-python==${LLAMA_CPP_VERSION}" \
        -w /output

    echo "✅ Wheel built successfully"
  '

WHEEL_FILE="$(ls -1 "$OUTPUT_DIR_ABS"/llama_cpp_python-*.whl 2>/dev/null | head -n1)"

if [[ ! -f "$WHEEL_FILE" ]]; then
  echo "❌ Wheel file not found in $OUTPUT_DIR_ABS"
  exit 1
fi

WHEEL_NAME="$(basename "$WHEEL_FILE")"
echo ""
echo "✅ CUDA wheel built successfully!"
echo "   File: $WHEEL_NAME"
echo "   Path: $WHEEL_FILE"
echo "   Backend: CUDA ${CUDA_VERSION} (Linux NVIDIA)"
echo ""
echo "📤 Next step: Upload to GitHub release"
echo "   Run: $SCRIPT_DIR/upload_wheel.sh $LLAMA_CPP_VERSION $WHEEL_FILE cuda"
