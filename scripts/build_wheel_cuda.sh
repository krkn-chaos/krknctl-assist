#!/usr/bin/env bash
set -euo pipefail

# Build llama-cpp-python wheel with CUDA backend (for Linux NVIDIA x86_64)
# CUDA wheels are x86_64 only - ARM64 NVIDIA support is rare (Jetson embedded)
# For ARM64 use Vulkan wheel instead: ./scripts/build_wheel_vulkan.sh

LLAMA_CPP_VERSION="${1:-0.3.19}"
OUTPUT_DIR="${2:-./wheels}"
CUDA_VERSION="${3:-12.6}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-detect platform - native build only, no cross-compilation
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
  docker.io/nvidia/cuda:${CUDA_VERSION}.1-devel-ubi9 \
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
    # Use shared CUDA libraries instead of static linking
    # This makes the wheel much smaller but requires CUDA runtime on target system
    # Supported GPU architectures (realistic for this tool):
    # 70: V100 (DGX-1/DGX-2)
    # 75: Turing consumer (RTX 20xx, GTX 1660 Ti)
    # 80: A100 (DGX A100, cloud)
    # 86: Ampere consumer (RTX 30xx)
    # 89: Ada Lovelace consumer (RTX 40xx)
    # 90: Hopper (H100 cloud GPU sharing - AWS p5, GCP, Azure)
    CMAKE_ARGS="\
      -DGGML_CUDA=on \
      -DCMAKE_CUDA_ARCHITECTURES=70;75;80;86;89;90 \
      -DGGML_STATIC=off \
      -DBUILD_SHARED_LIBS=on \
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
