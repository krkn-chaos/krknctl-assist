#!/usr/bin/env bash
set -euo pipefail

# Build llama-cpp-python wheel locally with Vulkan support
# This is MUCH faster than building in GitHub Actions with QEMU emulation

LLAMA_CPP_VERSION="${1:-0.3.19}"
OUTPUT_DIR="${2:-./wheels}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🔧 Building llama-cpp-python ${LLAMA_CPP_VERSION} wheel locally"
echo "   Output: $OUTPUT_DIR"
echo ""

# Detect architecture
ARCH="$(uname -m)"
case "$ARCH" in
  arm64|aarch64)
    PLATFORM="linux_aarch64"
    CONTAINER_PLATFORM="linux/arm64"
    DOCKERFILE="Dockerfile"
    ;;
  x86_64|amd64)
    PLATFORM="linux_x86_64"
    CONTAINER_PLATFORM="linux/amd64"
    DOCKERFILE="Dockerfile.linux"
    ;;
  *)
    echo "❌ Unsupported architecture: $ARCH"
    exit 1
    ;;
esac

echo "📦 Architecture: $ARCH → $PLATFORM"
echo "🐳 Using Dockerfile: $DOCKERFILE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR_ABS="$(cd "$OUTPUT_DIR" && pwd)"

# Build wheel in container using the same flags as Dockerfile
echo "⏳ Building wheel (this will take 10-20 minutes on native, or ~1.5h on emulated ARM)..."
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

WHEEL_FILE="$(ls -1 "$OUTPUT_DIR_ABS"/llama_cpp_python-*.whl 2>/dev/null | head -n1)"

if [[ ! -f "$WHEEL_FILE" ]]; then
  echo "❌ Wheel file not found in $OUTPUT_DIR_ABS"
  exit 1
fi

WHEEL_NAME="$(basename "$WHEEL_FILE")"
echo ""
echo "✅ Wheel built successfully!"
echo "   File: $WHEEL_NAME"
echo "   Path: $WHEEL_FILE"
echo ""
echo "📤 Next step: Upload to GitHub release"
echo "   Run: $SCRIPT_DIR/upload_wheel.sh $LLAMA_CPP_VERSION $WHEEL_FILE"
