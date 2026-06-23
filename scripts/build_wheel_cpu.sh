#!/usr/bin/env bash
set -euo pipefail

# Build llama-cpp-python wheel with CPU-only backend (no GPU acceleration)
# Smallest wheel size, works on any platform without GPU

LLAMA_CPP_VERSION="${1:-0.3.19}"
OUTPUT_DIR="${2:-./wheels}"

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

echo "💻 Building llama-cpp-python ${LLAMA_CPP_VERSION} wheel (CPU-only)"
echo "   Backend: CPU (no GPU acceleration)"
echo "   Platform: ${CONTAINER_PLATFORM}"
echo "   Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR_ABS="$(cd "$OUTPUT_DIR" && pwd)"

# Build wheel in container
echo "⏳ Building wheel with CPU backend (5-10 minutes)..."
echo ""

podman run --rm \
  --platform "$CONTAINER_PLATFORM" \
  -v "$OUTPUT_DIR_ABS:/output" \
  -e "LLAMA_CPP_VERSION=$LLAMA_CPP_VERSION" \
  fedora:42 \
  bash -c '
    set -euo pipefail

    echo "Installing build dependencies..."
    dnf -y install --setopt=install_weak_deps=False \
      python3 python3-pip python3-devel \
      gcc gcc-c++ cmake ninja-build make pkgconf-pkg-config \
      ccache >/dev/null 2>&1

    echo "Upgrading pip..."
    pip3 install --upgrade pip setuptools wheel >/dev/null 2>&1

    echo "Building llama-cpp-python ${LLAMA_CPP_VERSION} with CPU backend..."
    export PATH="/usr/lib64/ccache:$PATH"

    # Use all available cores for parallel build
    NCORES=$(nproc)
    echo "Using $NCORES parallel jobs"

    CMAKE_ARGS="\
      -DGGML_CUDA=off \
      -DGGML_VULKAN=off \
      -DGGML_METAL=off \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    " CMAKE_BUILD_PARALLEL_LEVEL=$NCORES \
      MAX_JOBS=$NCORES \
      FORCE_CMAKE=1 \
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
echo "✅ CPU-only wheel built successfully!"
echo "   File: $WHEEL_NAME"
echo "   Size: $WHEEL_SIZE"
echo "   Path: $WHEEL_FILE"
echo "   Backend: CPU (no GPU acceleration)"
echo ""
echo "📤 Next step: Upload to GitHub release"
echo "   Run: $SCRIPT_DIR/upload_wheel.sh $LLAMA_CPP_VERSION $WHEEL_FILE cpu"
