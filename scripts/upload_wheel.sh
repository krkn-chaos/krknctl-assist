#!/usr/bin/env bash
set -euo pipefail

LLAMA_CPP_VERSION="${1:?Missing version argument. Usage: $0 <version> [wheel-file]}"
WHEEL_FILE="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -z "$WHEEL_FILE" ]]; then
  echo "Searching for wheel file..."

  if [[ -d "$SCRIPT_DIR/wheels" ]]; then
    WHEEL_FILE="$(ls -1 "$SCRIPT_DIR/wheels"/llama_cpp_python-${LLAMA_CPP_VERSION}-*.whl 2>/dev/null | head -n1 || true)"
  fi

  if [[ -z "$WHEEL_FILE" ]] && [[ -d "$ROOT_DIR/wheels" ]]; then
    WHEEL_FILE="$(ls -1 "$ROOT_DIR/wheels"/llama_cpp_python-${LLAMA_CPP_VERSION}-*.whl 2>/dev/null | head -n1 || true)"
  fi

  if [[ -z "$WHEEL_FILE" ]]; then
    echo "ERROR: Wheel file not found"
    echo "Searched in:"
    [[ -d "$SCRIPT_DIR/wheels" ]] && echo "  - $SCRIPT_DIR/wheels/"
    [[ -d "$ROOT_DIR/wheels" ]] && echo "  - $ROOT_DIR/wheels/"
    echo ""
    echo "Usage: $0 <version> [wheel-file]"
    exit 1
  fi

  echo "Found: $(basename "$WHEEL_FILE")"
fi

if [[ ! -f "$WHEEL_FILE" ]]; then
  echo "ERROR: File not found: $WHEEL_FILE"
  exit 1
fi

WHEEL_NAME="$(basename "$WHEEL_FILE")"
TAG="llama-cpp-${LLAMA_CPP_VERSION}"

echo "Uploading wheel to GitHub release"
echo "  Tag: $TAG"
echo "  File: $WHEEL_NAME"
echo ""

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI (gh) not found"
  echo "Install: brew install gh"
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "ERROR: Not authenticated with GitHub CLI"
  echo "Run: gh auth login"
  exit 1
fi

cd "$ROOT_DIR"

if gh release view "$TAG" >/dev/null 2>&1; then
  echo "Release $TAG exists, uploading to existing release..."

  if gh release view "$TAG" --json assets --jq ".assets[].name" | grep -q "^${WHEEL_NAME}$"; then
    echo "WARNING: File already exists, deleting old version..."
    gh release delete-asset "$TAG" "$WHEEL_NAME" --yes
  fi

  gh release upload "$TAG" "$WHEEL_FILE"
else
  echo "Creating new release $TAG..."

  gh release create "$TAG" \
    --title "llama-cpp-python ${LLAMA_CPP_VERSION} (Vulkan)" \
    --notes "Pre-compiled llama-cpp-python wheel with Vulkan support

- Version: ${LLAMA_CPP_VERSION}
- Backend: Vulkan
- GGML_VULKAN_COOPMAT: OFF
- GGML_VULKAN_COOPMAT2: OFF

## Installation

\`\`\`bash
pip install https://github.com/krkn-chaos/krknctl-assist/releases/download/${TAG}/${WHEEL_NAME}
\`\`\`

## Build from source

\`\`\`bash
./scripts/build_wheel_local.sh ${LLAMA_CPP_VERSION}
./scripts/upload_wheel.sh ${LLAMA_CPP_VERSION}
\`\`\`
" \
    "$WHEEL_FILE"
fi

echo ""
echo "SUCCESS: Wheel uploaded"
echo "URL: https://github.com/krkn-chaos/krknctl-assist/releases/tag/$TAG"
