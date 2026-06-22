#!/usr/bin/env bash
set -euo pipefail

# Build llama-cpp-python wheel for Apple Silicon (Vulkan backend, ARM64 only)
# Target: macOS Apple Silicon running in container via Podman/Docker Desktop

LLAMA_CPP_VERSION="${1:-0.3.19}"
OUTPUT_DIR="${2:-./wheels}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🍎 Building llama-cpp-python ${LLAMA_CPP_VERSION} wheel (Apple Silicon)"
echo "   Backend: Vulkan"
echo "   Platform: linux/arm64"
echo "   Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR_ABS="$(cd "$OUTPUT_DIR" && pwd)"

# Build wheel in container
echo "⏳ Building wheel with Vulkan backend (10-15 minutes on ARM64)..."
echo ""

podman run --rm \
  --platform linux/arm64 \
  -v "$OUTPUT_DIR_ABS:/output" \
  -e "LLAMA_CPP_VERSION=$LLAMA_CPP_VERSION" \
  fedora:42 \
  bash -c '
    set -euo pipefail

    echo "Installing build dependencies..."
    dnf -y install --setopt=install_weak_deps=False \
      python3 python3-pip python3-devel \
      gcc gcc-c++ cmake ninja-build make pkgconf-pkg-config \
      mesa-vulkan-drivers vulkan-loader-devel vulkan-headers \
      vulkan-tools vulkan-loader glslc spirv-headers-devel \
      ccache >/dev/null 2>&1

    echo "Upgrading pip..."
    pip3 install --upgrade pip setuptools wheel >/dev/null 2>&1

    echo "Building llama-cpp-python ${LLAMA_CPP_VERSION} with Vulkan backend..."
    export PATH="/usr/lib64/ccache:$PATH"
    CMAKE_ARGS="\
      -DGGML_VULKAN=on \
      -DGGML_VULKAN_COOPMAT=OFF \
      -DGGML_VULKAN_COOPMAT2=OFF \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    " FORCE_CMAKE=1 \
      pip3 wheel --no-binary=llama-cpp-python \
        "llama-cpp-python==${LLAMA_CPP_VERSION}" \
        -w /output

    echo "✅ Wheel built successfully"
  '

WHEEL_FILE="$(ls -1 "$OUTPUT_DIR_ABS"/llama_cpp_python-*-linux_aarch64.whl 2>/dev/null | head -n1)"

if [[ ! -f "$WHEEL_FILE" ]]; then
  echo "❌ Wheel file not found in $OUTPUT_DIR_ABS"
  exit 1
fi

WHEEL_NAME="$(basename "$WHEEL_FILE")"
WHEEL_SIZE="$(du -h "$WHEEL_FILE" | cut -f1)"

echo ""
echo "✅ Apple Silicon wheel built successfully!"
echo "   File: $WHEEL_NAME"
echo "   Size: $WHEEL_SIZE"
echo "   Path: $WHEEL_FILE"
echo "   Backend: Vulkan (Apple Silicon / ARM64)"
echo ""
echo "📤 Next step: Upload to GitHub release"
echo "   Run: $SCRIPT_DIR/upload_wheel.sh $LLAMA_CPP_VERSION $WHEEL_FILE vulkan"
