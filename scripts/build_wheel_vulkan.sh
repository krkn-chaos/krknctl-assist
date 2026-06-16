#!/usr/bin/env bash
set -euo pipefail

# Build llama-cpp-python wheel with Vulkan backend (for Apple Silicon / ARM64)
# Use this on macOS with Apple Silicon or Linux ARM64

LLAMA_CPP_VERSION="${1:-0.3.19}"
OUTPUT_DIR="${2:-./wheels}"
TARGET_PLATFORM="${3:-}"  # Optional: linux/arm64, linux/amd64 for cross-compilation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔧 Building llama-cpp-python ${LLAMA_CPP_VERSION} wheel (Vulkan backend)"
echo "   Output: $OUTPUT_DIR"
echo "   Target: Apple Silicon / ARM64"
echo ""

# Detect or use specified platform
if [[ -n "$TARGET_PLATFORM" ]]; then
  # Platform explicitly specified for cross-compilation
  CONTAINER_PLATFORM="$TARGET_PLATFORM"
  case "$TARGET_PLATFORM" in
    linux/arm64|linux/aarch64)
      WHEEL_PLATFORM="linux_aarch64"
      ;;
    linux/amd64|linux/x86_64)
      WHEEL_PLATFORM="linux_x86_64"
      ;;
    *)
      echo "❌ Unsupported platform: $PLATFORM"
      echo "   Use: linux/arm64 or linux/amd64"
      exit 1
      ;;
  esac
  echo "🔀 Cross-compiling for platform: $CONTAINER_PLATFORM"
else
  # Auto-detect from host architecture
  ARCH="$(uname -m)"
  case "$ARCH" in
    arm64|aarch64)
      WHEEL_PLATFORM="linux_aarch64"
      CONTAINER_PLATFORM="linux/arm64"
      ;;
    x86_64|amd64)
      WHEEL_PLATFORM="linux_x86_64"
      CONTAINER_PLATFORM="linux/amd64"
      echo "⚠️  Building Vulkan wheel for x86_64"
      echo "   For NVIDIA GPUs on Linux, consider using build_wheel_cuda.sh instead"
      echo ""
      ;;
    *)
      echo "❌ Unsupported architecture: $ARCH"
      exit 1
      ;;
  esac
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR_ABS="$(cd "$OUTPUT_DIR" && pwd)"

# Build wheel in container
echo "⏳ Building wheel with Vulkan backend (10-15 minutes on native ARM)..."
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
echo "✅ Vulkan wheel built successfully!"
echo "   File: $WHEEL_NAME"
echo "   Path: $WHEEL_FILE"
echo "   Backend: Vulkan (Apple Silicon / ARM64)"
echo ""
echo "📤 Next step: Upload to GitHub release"
echo "   Run: $SCRIPT_DIR/upload_wheel.sh $LLAMA_CPP_VERSION $WHEEL_FILE vulkan"
