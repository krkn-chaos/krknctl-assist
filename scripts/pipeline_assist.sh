#!/usr/bin/env bash
set -euo pipefail

# krkn-assist pipeline: retrieves and reranks results
#
# Usage:
#   ./scripts/pipeline_assist.sh [query] [retrieve-k] [rerank-k] [--verbose]
#   (No query starts interactive mode)
#
# Optional environment variables:
#   CONTAINER_ENGINE=podman|docker
#   RETRIEVER_IMAGE=quay.io/krkn-chaos/krknctl-assist:faiss-latest
#   RETRIEVER_DOCKERFILE=/path/to/Dockerfile
#   RETRIEVER_COMPAT_PORT=8080    # host port for /v1/chat/completions
#   RETRIEVER_DEBUG_PORT=18080    # host port for /retrieve and /debug/*
#   RETRIEVER_BACKEND=auto|vulkan
#   RETRIEVER_MODEL=Qwen/Qwen3-Embedding-0.6B
#   RETRIEVER_ACCELERATION=auto|gpu
#   LLAMA_EMBED_MODEL=/abs/path/to/model.gguf
#   LLAMA_RERANKER_MODEL=/abs/path/to/reranker.gguf
#   LLAMA_GPU_LAYERS=-1
#   RETRIEVER_GGUF_REPO=Qwen/Qwen3-Embedding-0.6B-GGUF
#   RETRIEVER_GGUF_FILE=Qwen3-Embedding-0.6B-f16.gguf
#   RETRIEVER_RERANKER_GGUF_REPO=gpustack/bge-reranker-v2-m3-GGUF  (ignored: ONNX reranker used)
#   RETRIEVER_RERANKER_GGUF_FILE=bge-reranker-v2-m3-Q2_K.gguf       (ignored: ONNX reranker used)
#   RETRIEVER_AUTO_DOWNLOAD_MODEL=1
#   INDEX_TTL_DAYS=7
#   GITHUB_REPO=https://github.com/krkn-chaos/website
#   GITHUB_BRANCH=main
#   REPO_PATH=content/en/docs
#   KRKN_HUB_REPO=https://github.com/krkn-chaos/krkn-hub
#   KRKN_HUB_BRANCH=
#   LOCAL_DOCS_PATH=/path/to/website/repo  # repo root or docs root
#   CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
#   RERANK_MAX_LENGTH=192
#   RERANK_CANDIDATE_K=0       # 0 means use RERANK_TOP_FRACTION for the CE window
#   RETRIEVER_FORCE_BUILD=1
#   FORCE_REINDEX=true
#   RETRIEVER_FORCE_REINDEX=1
#   HF_TOKEN / HUGGING_FACE_HUB_TOKEN (optional, for gated HF downloads)
#   HF_CACHE_DIR, TORCH_CACHE_DIR
#
# Output:
#   retrieval container writes ./shared/retrieval_output.json

VERBOSE=0
INTERACTIVE=0
QUERY_MS=0
BUILD_MS=0
INDEX_MS=0
TOTAL_MS=0
KEEP_PIPELINE_LOG=0
POSITIONAL_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --verbose) VERBOSE=1 ;;
    *) POSITIONAL_ARGS+=("$arg") ;;
  esac
done
set -- "${POSITIONAL_ARGS[@]+"${POSITIONAL_ARGS[@]}"}"

QUERY="${1:-}"
RETRIEVE_K="${2:-10}"
RERANK_K="${3:-5}"

if [[ -z "$QUERY" ]]; then
  INTERACTIVE=1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SHARED_DIR="$ROOT_DIR/shared"
INDEX_FILE="$ROOT_DIR/krkn-assist/faiss-index/krkn-scenarios.index"
META_FILE="$ROOT_DIR/krkn-assist/faiss-index/krkn-scenarios.meta"
DOCS_CACHE_FILE="$ROOT_DIR/krkn-assist/faiss-index/krkn-scenarios.docs.json"
HF_CACHE_DIR="${HF_CACHE_DIR:-$ROOT_DIR/.cache/huggingface}"
TORCH_CACHE_DIR="${TORCH_CACHE_DIR:-$ROOT_DIR/.cache/torch}"
ENGINE="${CONTAINER_ENGINE:-podman}"
IMAGE="${RETRIEVER_IMAGE:-quay.io/krkn-chaos/krknctl-assist:faiss-latest}"
DOCKERFILE="${RETRIEVER_DOCKERFILE:-}"
FORCE_BUILD="${RETRIEVER_FORCE_BUILD:-0}"
FORCE_REINDEX_RAW="${FORCE_REINDEX:-${RETRIEVER_FORCE_REINDEX:-0}}"
FORCE_REINDEX="false"
case "$FORCE_REINDEX_RAW" in
  1|true|TRUE|yes|YES) FORCE_REINDEX="true" ;;
  *) FORCE_REINDEX="false" ;;
esac
BACKEND="${RETRIEVER_BACKEND:-vulkan}"
LLAMA_EMBED_MODEL_PATH="${LLAMA_EMBED_MODEL:-}"
LLAMA_RERANKER_MODEL_PATH="${LLAMA_RERANKER_MODEL:-}"
LLAMA_GPU_LAYERS="${LLAMA_GPU_LAYERS:--1}"
GGUF_REPO="${RETRIEVER_GGUF_REPO:-Qwen/Qwen3-Embedding-0.6B-GGUF}"
GGUF_FILE="${RETRIEVER_GGUF_FILE:-Qwen3-Embedding-0.6B-f16.gguf}"
RERANKER_GGUF_REPO="${RETRIEVER_RERANKER_GGUF_REPO:-gpustack/bge-reranker-v2-m3-GGUF}"
RERANKER_GGUF_FILE="${RETRIEVER_RERANKER_GGUF_FILE:-bge-reranker-v2-m3-Q2_K.gguf}"
# NOTE: LLAMA_RERANKER_MODEL / RERANKER_GGUF_* are accepted for backward compat
# but the reranker now uses OnnxCrossEncoder (downloaded from HuggingFace at
# first query).  No GGUF reranker model file is downloaded or mounted.
AUTO_DOWNLOAD_MODEL="${RETRIEVER_AUTO_DOWNLOAD_MODEL:-1}"
ACCELERATION_MODE="${RETRIEVER_ACCELERATION:-auto}"
HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}"
CROSS_ENCODER_MODEL="${CROSS_ENCODER_MODEL:-cross-encoder/ms-marco-MiniLM-L-6-v2}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
RERANK_MAX_LENGTH="${RERANK_MAX_LENGTH:-256}"
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE:-16}"
RERANK_DOC_CHARS="${RERANK_DOC_CHARS:-2400}"
RERANK_THREADS="${RERANK_THREADS:-4}"
RERANK_CANDIDATE_K="${RERANK_CANDIDATE_K:-0}"
RERANK_TOP_FRACTION="${RERANK_TOP_FRACTION:-0.5}"
RETRIEVAL_CANDIDATE_K="${RETRIEVAL_CANDIDATE_K:-34}"
RERANK_SUPPORT_PASSAGES="${RERANK_SUPPORT_PASSAGES:-3}"
RERANK_ONNX_QUANTIZE="${RERANK_ONNX_QUANTIZE:-1}"
MIN_FAISS_SCORE="${MIN_FAISS_SCORE:-0.23}"
MIN_CE_SCORE="${MIN_CE_SCORE:--10.0}"
MIN_MATCH_SCORE="${MIN_MATCH_SCORE:-0.10}"
RELEVANCE_THRESHOLD="${RELEVANCE_THRESHOLD:-$MIN_MATCH_SCORE}"
EXCLUDED_SCENARIO_IDS="${EXCLUDED_SCENARIO_IDS:-dummy-scenario}"
INDEX_TTL_DAYS="${INDEX_TTL_DAYS:-7}"
GITHUB_REPO="${GITHUB_REPO:-https://github.com/krkn-chaos/website}"
GITHUB_BRANCH="${GITHUB_BRANCH:-main}"
REPO_PATH="${REPO_PATH:-content/en/docs}"
KRKN_HUB_REPO="${KRKN_HUB_REPO:-https://github.com/krkn-chaos/krkn-hub}"
KRKN_HUB_BRANCH="${KRKN_HUB_BRANCH:-}"
LOCAL_DOCS_PATH="${LOCAL_DOCS_PATH:-}"

# Support llama.cpp shorthand "repo:QUANT" (e.g. gpustack/bge-reranker-v2-m3-GGUF:Q2_K)
if [[ "$RERANKER_GGUF_REPO" == *:* ]]; then
  RERANKER_QUANT="${RERANKER_GGUF_REPO#*:}"
  RERANKER_GGUF_REPO="${RERANKER_GGUF_REPO%%:*}"
  if [[ -z "${RETRIEVER_RERANKER_GGUF_FILE:-}" ]]; then
    RERANKER_GGUF_FILE="$RERANKER_QUANT"
  fi
fi

# Allow passing just quant name for the gpustack v2-m3 repo.
if [[ "$RERANKER_GGUF_REPO" == "gpustack/bge-reranker-v2-m3-GGUF" && "$RERANKER_GGUF_FILE" != *.gguf ]]; then
  RERANKER_GGUF_FILE="bge-reranker-v2-m3-${RERANKER_GGUF_FILE}.gguf"
fi

# Auto-detect host OS
HOST_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

if [[ -z "$DOCKERFILE" ]]; then
  if [[ "$HOST_OS" == "darwin" ]]; then
    DOCKERFILE="$ROOT_DIR/krkn-assist/Dockerfile"
  else
    DOCKERFILE="$ROOT_DIR/krkn-assist/Dockerfile.linux"
  fi
fi

# On macOS with podman, the libkrun VM provider is required for Vulkan GPU
# passthrough via virtio-gpu/Venus.  Without this, podman silently falls back to
# the applehv provider which has no GPU device and leaves Vulkan on llvmpipe.
# Respect an explicit override; default to libkrun when unset.
if [[ "$HOST_OS" == "darwin" && "$ENGINE" == "podman" && "$BACKEND" == "vulkan" ]]; then
  export CONTAINERS_MACHINE_PROVIDER="${CONTAINERS_MACHINE_PROVIDER:-libkrun}"
fi

case "$ACCELERATION_MODE" in
  auto|gpu) ;;
  cpu) echo "Error: CPU acceleration mode is disabled; Vulkan is always used"; exit 1 ;;
  *) echo "Error: RETRIEVER_ACCELERATION must be one of auto|gpu"; exit 1 ;;
esac

case "$BACKEND" in
  auto|vulkan) ;;
  torch) echo "Error: torch backend is disabled; use RETRIEVER_BACKEND=vulkan"; exit 1 ;;
  *) echo "Error: RETRIEVER_BACKEND must be one of auto|vulkan"; exit 1 ;;
esac

if ! command -v "$ENGINE" >/dev/null 2>&1; then
  echo "Error: container engine '$ENGINE' not found"
  exit 1
fi

ensure_podman_machine() {
  # Linux podman is native: no podman machine management needed.
  [[ "$ENGINE" == "podman" ]] || return 0
  [[ "$HOST_OS" == "darwin" ]] || return 0

  if [[ "$BACKEND" == "vulkan" ]]; then
    export CONTAINERS_MACHINE_PROVIDER="${CONTAINERS_MACHINE_PROVIDER:-libkrun}"
  fi

  # Create machine if missing.
  if ! "$ENGINE" machine list --format "{{.Name}}" 2>/dev/null | grep -q .; then
    echo "No podman machine found. Initializing with libkrun provider..."
    "$ENGINE" machine init --cpus 4 --memory 8192 --disk-size 100
  fi

  local machine
  machine=$("$ENGINE" machine list --format "{{.Name}}" 2>/dev/null | head -n1 | tr -d '*') || true
  if [[ -z "$machine" ]]; then
    echo "Error: could not determine podman machine name"
    return 1
  fi

  # Recreate applehv machine only when Vulkan is explicitly requested.
  if [[ "$BACKEND" == "vulkan" ]] && "$ENGINE" machine inspect "$machine" 2>/dev/null | grep -qi "applehv"; then
    echo "Recreating podman machine with libkrun (required for Vulkan on macOS): $machine"
    "$ENGINE" machine rm -f "$machine"
    "$ENGINE" machine init --cpus 4 --memory 8192 --disk-size 100
    machine=$("$ENGINE" machine list --format "{{.Name}}" 2>/dev/null | head -n1 | tr -d '*') || true
  fi

  # Attempt to start machine; ignore "already running" errors.
  if ! "$ENGINE" machine start "$machine" 2>&1 | grep -qi "already running\|started"; then
    # Attempt failed and didn't say "already running"; may have started anyway, try a final check.
    sleep 1
    if ! "$ENGINE" machine list --format "{{.Name}}" 2>/dev/null | grep -q "$(echo "$machine" | sed 's/\*//')"; then
      echo "Error: podman machine $machine failed to start"
      return 1
    fi
  fi
  
  return 0
}

# Build-time args (auto-detected based on host platform)
BUILD_ARGS=()
GPU_FLAGS=()
DEVICE_ARGS=(--device auto)
LLAMA_MOUNT_ARGS=()
GPU_RUNTIME_KIND="none"
GGML_BACKEND_DESIRED=""
CONTAINER_PLATFORM=""
PLATFORM_BUILD_ARGS=()

MOUNT_LABEL_SUFFIX=""
if [[ "$ENGINE" == "podman" && "$HOST_OS" != "darwin" ]]; then
  MOUNT_LABEL_SUFFIX=":Z"
fi

LOCAL_DOCS_ENV_PATH="$LOCAL_DOCS_PATH"
LOCAL_DOCS_MOUNT_ARGS=()
if [[ -n "$LOCAL_DOCS_PATH" ]]; then
  if [[ -d "$LOCAL_DOCS_PATH" ]]; then
    LOCAL_DOCS_ENV_PATH="/app/local_docs"
    LOCAL_DOCS_MOUNT_ARGS=(-v "$LOCAL_DOCS_PATH:$LOCAL_DOCS_ENV_PATH$MOUNT_LABEL_SUFFIX")
  else
    echo "Warning: LOCAL_DOCS_PATH not found: $LOCAL_DOCS_PATH" >&2
  fi
fi

vlog() {
  if [[ "$VERBOSE" == "1" ]]; then
    echo "$@"
  fi
}

status_note() {
  if [[ "$VERBOSE" != "1" ]]; then
    printf '%s\n' "$*"
  fi
}

status_wait_for_pid() {
  local label="$1"
  local pid="$2"
  local frames=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
  local i=0
  local status

  if [[ "$VERBOSE" == "1" ]]; then
    set +e
    wait "$pid"
    status=$?
    set -e
    return "$status"
  fi

  while kill -0 "$pid" 2>/dev/null; do
    printf '\r%s %s' "${frames[$((i % ${#frames[@]}))]}" "$label"
    i=$((i + 1))
    sleep 0.12
  done

  set +e
  wait "$pid"
  status=$?
  set -e
  return "$status"
}

status_run_quiet() {
  local label="$1"
  shift
  local pid
  local status

  if [[ "$VERBOSE" == "1" ]]; then
    "$@"
    return
  fi

  "$@" >>"$PIPELINE_LOG" 2>&1 &
  pid=$!
  status_wait_for_pid "$label" "$pid"
  status=$?
  printf '\r'
  if [[ "$status" -eq 0 ]]; then
    status_note "✅ $label"
  else
    KEEP_PIPELINE_LOG=1
    status_note "❌ $label failed"
    status_note "📄 Details: $PIPELINE_LOG"
    exit "$status"
  fi
}

status_try_quiet() {
  local label="$1"
  shift
  local pid
  local status

  if [[ "$VERBOSE" == "1" ]]; then
    "$@"
    return
  fi

  "$@" >>"$PIPELINE_LOG" 2>&1 &
  pid=$!
  status_wait_for_pid "$label" "$pid"
  status=$?
  printf '\r'
  if [[ "$status" -eq 0 ]]; then
    status_note "✅ $label"
  else
    KEEP_PIPELINE_LOG=1
    status_note "❌ $label failed"
    status_note "📄 Details: $PIPELINE_LOG"
  fi
  return "$status"
}

stop_stale_krknctl_processes() {
  command -v pgrep >/dev/null 2>&1 || return 0

  local pids
  pids="$(pgrep -f 'krknctl[[:space:]]+assist[[:space:]]+run' 2>/dev/null \
    | awk -v self="$$" '$1 != self {print $1}')" || true
  [[ -n "$pids" ]] || return 0

  vlog "      Stopping stale krknctl assist process(es): $pids"
  # shellcheck disable=SC2086
  kill $pids >/dev/null 2>&1 || true
  sleep 2

  local remaining
  remaining="$(pgrep -f 'krknctl[[:space:]]+assist[[:space:]]+run' 2>/dev/null \
    | awk -v self="$$" '$1 != self {print $1}')" || true
  if [[ -n "$remaining" ]]; then
    vlog "      Force-stopping stale krknctl assist process(es): $remaining"
    # shellcheck disable=SC2086
    kill -9 $remaining >/dev/null 2>&1 || true
  fi
}

DOCS_MOUNT_ARGS=()
if [[ -d "$ROOT_DIR/docs" ]]; then
  DOCS_MOUNT_ARGS=(-v "$ROOT_DIR/docs:/app/docs$MOUNT_LABEL_SUFFIX")
else
  vlog "      Local docs directory missing; retriever will fetch docs from GitHub"
fi

now_ms() {
  # macOS-compatible millisecond timestamp (date +%s%3N not available on macOS)
  if [[ "$HOST_OS" == "darwin" ]]; then
    python3 -c "import time; print(int(time.time() * 1000))"
  else
    date +%s%3N
  fi
}

PIPELINE_LOG="$(mktemp)"
QUERY_LOG=""
API_CONTAINER_ID=""
API_CONTAINER_NAME="krkn-assist-pipeline-$$"
# Debug API (internal) default: 18080
# Compatibility API (public/krknctl) default: 8080
DEBUG_PORT="${RETRIEVER_DEBUG_PORT:-${RETRIEVER_API_PORT:-18080}}"
COMPAT_PORT="${RETRIEVER_COMPAT_PORT:-8080}"
API_BASE_URL="http://127.0.0.1:${DEBUG_PORT}"
COMPAT_BASE_URL="http://127.0.0.1:${COMPAT_PORT}"
cleanup_logs() {
  if [[ "$KEEP_PIPELINE_LOG" != "1" ]]; then
    rm -f "$PIPELINE_LOG"
  fi
  if [[ -n "$QUERY_LOG" ]]; then
    rm -f "$QUERY_LOG"
  fi
  if [[ -n "$API_CONTAINER_ID" ]]; then
    "$ENGINE" rm -f "$API_CONTAINER_ID" >/dev/null 2>&1 || true
  fi
  "$ENGINE" rm -f "$API_CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup_logs EXIT

cleanup_assist_containers() {
  stop_stale_krknctl_processes

  local ids
  ids="$("$ENGINE" ps --format '{{.ID}} {{.Names}} {{.Image}} {{.Ports}}' 2>/dev/null \
    | awk -v compat=":${COMPAT_PORT}->" -v debug=":${DEBUG_PORT}->" \
      '($0 ~ compat || $0 ~ debug || /krknctl-assist-rag|krknctl-assist-smoke/) {print $1}')"
  if [[ -n "$ids" ]]; then
    vlog "      Removing running assist containers on ${COMPAT_PORT}/${DEBUG_PORT}"
    # shellcheck disable=SC2086
    "$ENGINE" rm -f $ids >/dev/null 2>&1 || true
  fi

  ids="$("$ENGINE" ps -a --format '{{.ID}} {{.Names}} {{.Image}}' 2>/dev/null \
    | awk '/krknctl-assist-rag|krknctl-assist-smoke|krkn-assist-pipeline/ {print $1}')"
  if [[ -n "$ids" ]]; then
    vlog "      Removing stale assist containers"
    # shellcheck disable=SC2086
    "$ENGINE" rm -f $ids >/dev/null 2>&1 || true
  fi
}

run_cmd() {
  if [[ "$VERBOSE" == "1" ]]; then
    "$@"
    return
  fi

  if ! "$@" >>"$PIPELINE_LOG" 2>&1; then
    KEEP_PIPELINE_LOG=1
    echo "❌ Retrieval pipeline failed."
    echo "📄 Details: $PIPELINE_LOG"
    exit 1
  fi
}

index_doc_count() {
  if [[ ! -f "$META_FILE" ]]; then
    echo "0"
    return
  fi
  python3 - "$META_FILE" <<'PY'
import pickle
import sys
from pathlib import Path

meta = Path(sys.argv[1])
if not meta.exists():
    print("0")
    raise SystemExit(0)
with meta.open("rb") as f:
    ids = pickle.load(f)
print(len(ids))
PY
}

index_is_stale() {
  python3 - "$INDEX_FILE" "$META_FILE" "$DOCS_CACHE_FILE" "$INDEX_TTL_DAYS" <<'PY'
import os
import sys
import time

index_path, meta_path, docs_path, ttl_days = sys.argv[1:5]
try:
  ttl = float(ttl_days)
except ValueError:
  ttl = 0.0

if ttl <= 0:
  print("0")
  raise SystemExit(0)

if not (os.path.exists(index_path) and os.path.exists(meta_path)):
  print("1")
  raise SystemExit(0)

if not os.path.exists(docs_path):
  print("1")
  raise SystemExit(0)

mtimes = []
for path in (index_path, meta_path, docs_path):
  if os.path.exists(path):
    mtimes.append(os.path.getmtime(path))

if not mtimes:
  print("1")
  raise SystemExit(0)

age = time.time() - max(mtimes)
print("1" if age >= (ttl * 86400.0) else "0")
PY
}

append_linux_vulkan_devices() {
  local added_device=0

  if [[ -e /dev/dri ]]; then
    GPU_FLAGS+=(--device /dev/dri)
    GPU_RUNTIME_KIND="vulkan-dri"
    added_device=1
  fi

  if [[ -e /dev/kfd ]]; then
    GPU_FLAGS+=(--device /dev/kfd)
    GPU_RUNTIME_KIND="${GPU_RUNTIME_KIND}+kfd"
    added_device=1
  fi

  if [[ -e /dev/dxg ]]; then
    GPU_FLAGS+=(--device /dev/dxg)
    GPU_RUNTIME_KIND="vulkan-wsl-dxg"
    added_device=1
    if [[ -d /usr/lib/wsl ]]; then
      GPU_FLAGS+=(
        -v /usr/lib/wsl:/usr/lib/wsl:ro
        -e "LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
      )
    fi
  fi

  local nvidia_dev
  for nvidia_dev in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset /dev/nvidia[0-9]*; do
    if [[ -e "$nvidia_dev" ]]; then
      GPU_FLAGS+=(--device "$nvidia_dev")
      GPU_RUNTIME_KIND="${GPU_RUNTIME_KIND}+nvidia"
      added_device=1
    fi
  done

  if [[ "$added_device" == "1" && "$ENGINE" == "podman" ]]; then
    GPU_FLAGS+=(
      --group-add keep-groups
      --security-opt=label=disable
      -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
    )
  fi
}

image_is_compatible() {
  "$ENGINE" run --rm --entrypoint python3 "$IMAGE" -c "import faiss, llama_cpp" >/dev/null 2>&1 || return 1

  # Verify image was built with the right GGML backend
  if [[ -n "$GGML_BACKEND_DESIRED" ]]; then
    local image_backend
    image_backend=$("$ENGINE" run --rm --entrypoint sh "$IMAGE" -c 'echo $GGML_BACKEND_BUILT' 2>/dev/null)
    if [[ "$image_backend" != "$GGML_BACKEND_DESIRED" ]]; then
      vlog "      Image backend mismatch: built=$image_backend want=$GGML_BACKEND_DESIRED"
      return 1
    fi
  fi
  "$ENGINE" run --rm --entrypoint sh "$IMAGE" -c 'test "${RETRIEVER_BACKEND:-}" = "vulkan" && test -n "${LLAMA_EMBED_MODEL:-}" && test -f "$LLAMA_EMBED_MODEL"' >/dev/null 2>&1 || return 1
}

download_gguf_model() {
  local out_path="$1"
  local url="https://huggingface.co/${GGUF_REPO}/resolve/main/${GGUF_FILE}"
  vlog "      Downloading GGUF model: ${GGUF_REPO}/${GGUF_FILE}"
  mkdir -p "$(dirname "$out_path")"
  local curl_args=(-L --fail --retry 3 --continue-at - -o "$out_path")
  if [[ -n "$HF_TOKEN" ]]; then
    curl_args+=(-H "Authorization: Bearer $HF_TOKEN")
  fi
  curl "${curl_args[@]}" "$url"
}

download_reranker_gguf_model() {
  local out_path="$1"
  local url="https://huggingface.co/${RERANKER_GGUF_REPO}/resolve/main/${RERANKER_GGUF_FILE}"
  vlog "      Downloading GGUF reranker: ${RERANKER_GGUF_REPO}/${RERANKER_GGUF_FILE}"
  mkdir -p "$(dirname "$out_path")"
  local curl_args=(-L --fail --retry 3 --continue-at - -o "$out_path")
  if [[ -n "$HF_TOKEN" ]]; then
    curl_args+=(-H "Authorization: Bearer $HF_TOKEN")
  fi
  curl "${curl_args[@]}" "$url"
}

configure_acceleration() {
  GPU_FLAGS=()
  DEVICE_ARGS=(--device auto)
  GPU_RUNTIME_KIND="vulkan"
  GGML_BACKEND_DESIRED="vulkan"
  BUILD_ARGS+=(--build-arg GGML_BACKEND=vulkan)
  if [[ -n "$LLAMA_EMBED_MODEL_PATH" ]]; then
    BUILD_ARGS+=(--build-arg LLAMA_EMBED_MODEL="$LLAMA_EMBED_MODEL_PATH")
  fi

  # macOS: Vulkan acceleration via libkrun + virtio-gpu + Mesa Venus.
  # The libkrun VM exposes /dev/dri inside the VM; podman does NOT forward it to
  # containers automatically.  --device /dev/dri is the one flag that bridges the
  # VM's DRM nodes into the container so Mesa Venus picks up real GPU hardware
  # instead of llvmpipe.
  if [[ "$HOST_OS" == "darwin" || "$HOST_OS" == "macos" ]]; then
    GPU_RUNTIME_KIND="vulkan-venus"
    GPU_FLAGS=(--device /dev/dri)
    return
  fi

  if [[ "$HOST_OS" == "linux" ]]; then
    append_linux_vulkan_devices
    if [[ ${#GPU_FLAGS[@]} -eq 0 ]]; then
      GPU_RUNTIME_KIND="vulkan-container"
      vlog "      No Linux GPU device nodes detected; using container-visible Vulkan devices"
    fi
    return
  fi

  GPU_RUNTIME_KIND="vulkan-container"
}

build_image() {
  "$ENGINE" build \
    -t "$IMAGE" \
    -f "$DOCKERFILE" \
    ${PLATFORM_BUILD_ARGS[@]+"${PLATFORM_BUILD_ARGS[@]}"} \
    ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} \
    "$ROOT_DIR"
}

run_engine() {
  if [[ -n "${GPU_FLAGS+x}" ]] && [[ ${#GPU_FLAGS[@]} -gt 0 ]]; then
    "$ENGINE" run --rm "${GPU_FLAGS[@]}" "$@"
  else
    "$ENGINE" run --rm "$@"
  fi
}

run_engine_with_stdin() {
  if [[ -n "${GPU_FLAGS+x}" ]] && [[ ${#GPU_FLAGS[@]} -gt 0 ]]; then
    "$ENGINE" run --rm -i "${GPU_FLAGS[@]}" "$@"
  else
    "$ENGINE" run --rm -i "$@"
  fi
}

run_retriever_python() {
  local run_args=(--entrypoint python3)
  if [[ -n "${DOCS_MOUNT_ARGS+x}" ]] && [[ ${#DOCS_MOUNT_ARGS[@]} -gt 0 ]]; then
    run_args+=("${DOCS_MOUNT_ARGS[@]}")
  fi
  if [[ -n "${LLAMA_MOUNT_ARGS+x}" ]] && [[ ${#LLAMA_MOUNT_ARGS[@]} -gt 0 ]]; then
    run_args+=("${LLAMA_MOUNT_ARGS[@]}")
  fi
  if [[ -n "${LOCAL_DOCS_MOUNT_ARGS+x}" ]] && [[ ${#LOCAL_DOCS_MOUNT_ARGS[@]} -gt 0 ]]; then
    run_args+=("${LOCAL_DOCS_MOUNT_ARGS[@]}")
  fi
  run_engine "${run_args[@]}" "$@"
}

start_api_standby() {
  local run_args=()
  if [[ -n "${DOCS_MOUNT_ARGS+x}" ]] && [[ ${#DOCS_MOUNT_ARGS[@]} -gt 0 ]]; then
    run_args+=("${DOCS_MOUNT_ARGS[@]}")
  fi
  if [[ -n "${LLAMA_MOUNT_ARGS+x}" ]] && [[ ${#LLAMA_MOUNT_ARGS[@]} -gt 0 ]]; then
    run_args+=("${LLAMA_MOUNT_ARGS[@]}")
  fi
  if [[ -n "${LOCAL_DOCS_MOUNT_ARGS+x}" ]] && [[ ${#LOCAL_DOCS_MOUNT_ARGS[@]} -gt 0 ]]; then
    run_args+=("${LOCAL_DOCS_MOUNT_ARGS[@]}")
  fi

  local container_id
  if [[ -n "${GPU_FLAGS+x}" ]] && [[ ${#GPU_FLAGS[@]} -gt 0 ]]; then
    container_id=$(
      "$ENGINE" run -d \
        --name "$API_CONTAINER_NAME" \
        "${GPU_FLAGS[@]}" \
        ${run_args[@]+"${run_args[@]}"} \
        -p "127.0.0.1:${COMPAT_PORT}:8080" \
        -p "127.0.0.1:${DEBUG_PORT}:18080" \
        -v "$ROOT_DIR/krkn-assist:/app$MOUNT_LABEL_SUFFIX" \
        -v "$HF_CACHE_DIR:/root/.cache/huggingface$MOUNT_LABEL_SUFFIX" \
        -v "$TORCH_CACHE_DIR:/root/.cache/torch$MOUNT_LABEL_SUFFIX" \
        -e DOCS_DIR=/app/docs \
        -e INDEX_TTL_DAYS="$INDEX_TTL_DAYS" \
        -e FORCE_REINDEX="$FORCE_REINDEX" \
        -e GITHUB_REPO="$GITHUB_REPO" \
        -e GITHUB_BRANCH="$GITHUB_BRANCH" \
        -e REPO_PATH="$REPO_PATH" \
        -e KRKN_HUB_REPO="$KRKN_HUB_REPO" \
        -e KRKN_HUB_BRANCH="$KRKN_HUB_BRANCH" \
        -e LOCAL_DOCS_PATH="$LOCAL_DOCS_ENV_PATH" \
        -e RETRIEVER_BACKEND="$BACKEND" \
        -e RETRIEVER_MODEL="$RETRIEVER_MODEL" \
        -e CROSS_ENCODER_MODEL="$CROSS_ENCODER_MODEL" \
        -e RERANK_MAX_LENGTH="$RERANK_MAX_LENGTH" \
        -e RERANK_BATCH_SIZE="$RERANK_BATCH_SIZE" \
        -e RERANK_DOC_CHARS="$RERANK_DOC_CHARS" \
        -e RERANK_THREADS="$RERANK_THREADS" \
        -e RERANK_CANDIDATE_K="$RERANK_CANDIDATE_K" \
        -e RERANK_TOP_FRACTION="$RERANK_TOP_FRACTION" \
        -e RETRIEVAL_CANDIDATE_K="$RETRIEVAL_CANDIDATE_K" \
        -e RERANK_SUPPORT_PASSAGES="$RERANK_SUPPORT_PASSAGES" \
        -e RERANK_ONNX_QUANTIZE="$RERANK_ONNX_QUANTIZE" \
        -e MIN_FAISS_SCORE="$MIN_FAISS_SCORE" \
        -e MIN_CE_SCORE="$MIN_CE_SCORE" \
        -e MIN_MATCH_SCORE="$MIN_MATCH_SCORE" \
        -e RELEVANCE_THRESHOLD="$RELEVANCE_THRESHOLD" \
        -e EXCLUDED_SCENARIO_IDS="$EXCLUDED_SCENARIO_IDS" \
        -e LLAMA_EMBED_MODEL="$LLAMA_EMBED_MODEL_PATH" \
        -e LLAMA_RERANKER_MODEL="$LLAMA_RERANKER_MODEL_PATH" \
        -e LLAMA_GPU_LAYERS="$LLAMA_GPU_LAYERS" \
        -e HF_HOME=/root/.cache/huggingface \
        -e SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface \
        -e TORCH_HOME=/root/.cache/torch \
        -e RETRIEVE_K="$RETRIEVE_K" \
        -e RERANK_K="$RERANK_K" \
        -w /app \
        "$IMAGE"
    )
  else
    container_id=$(
      "$ENGINE" run -d \
        --name "$API_CONTAINER_NAME" \
        ${run_args[@]+"${run_args[@]}"} \
        -p "127.0.0.1:${COMPAT_PORT}:8080" \
        -p "127.0.0.1:${DEBUG_PORT}:18080" \
        -v "$ROOT_DIR/krkn-assist:/app$MOUNT_LABEL_SUFFIX" \
        -v "$HF_CACHE_DIR:/root/.cache/huggingface$MOUNT_LABEL_SUFFIX" \
        -v "$TORCH_CACHE_DIR:/root/.cache/torch$MOUNT_LABEL_SUFFIX" \
        -e DOCS_DIR=/app/docs \
        -e INDEX_TTL_DAYS="$INDEX_TTL_DAYS" \
        -e FORCE_REINDEX="$FORCE_REINDEX" \
        -e GITHUB_REPO="$GITHUB_REPO" \
        -e GITHUB_BRANCH="$GITHUB_BRANCH" \
        -e REPO_PATH="$REPO_PATH" \
        -e KRKN_HUB_REPO="$KRKN_HUB_REPO" \
        -e KRKN_HUB_BRANCH="$KRKN_HUB_BRANCH" \
        -e LOCAL_DOCS_PATH="$LOCAL_DOCS_ENV_PATH" \
        -e RETRIEVER_BACKEND="$BACKEND" \
        -e RETRIEVER_MODEL="$RETRIEVER_MODEL" \
        -e CROSS_ENCODER_MODEL="$CROSS_ENCODER_MODEL" \
        -e RERANK_MAX_LENGTH="$RERANK_MAX_LENGTH" \
        -e RERANK_BATCH_SIZE="$RERANK_BATCH_SIZE" \
        -e RERANK_DOC_CHARS="$RERANK_DOC_CHARS" \
        -e RERANK_THREADS="$RERANK_THREADS" \
        -e RERANK_CANDIDATE_K="$RERANK_CANDIDATE_K" \
        -e RERANK_TOP_FRACTION="$RERANK_TOP_FRACTION" \
        -e RETRIEVAL_CANDIDATE_K="$RETRIEVAL_CANDIDATE_K" \
        -e RERANK_SUPPORT_PASSAGES="$RERANK_SUPPORT_PASSAGES" \
        -e RERANK_ONNX_QUANTIZE="$RERANK_ONNX_QUANTIZE" \
        -e MIN_FAISS_SCORE="$MIN_FAISS_SCORE" \
        -e MIN_CE_SCORE="$MIN_CE_SCORE" \
        -e MIN_MATCH_SCORE="$MIN_MATCH_SCORE" \
        -e RELEVANCE_THRESHOLD="$RELEVANCE_THRESHOLD" \
        -e EXCLUDED_SCENARIO_IDS="$EXCLUDED_SCENARIO_IDS" \
        -e LLAMA_EMBED_MODEL="$LLAMA_EMBED_MODEL_PATH" \
        -e LLAMA_RERANKER_MODEL="$LLAMA_RERANKER_MODEL_PATH" \
        -e LLAMA_GPU_LAYERS="$LLAMA_GPU_LAYERS" \
        -e HF_HOME=/root/.cache/huggingface \
        -e SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface \
        -e TORCH_HOME=/root/.cache/torch \
        -e RETRIEVE_K="$RETRIEVE_K" \
        -e RERANK_K="$RERANK_K" \
        -w /app \
        "$IMAGE"
    )
  fi

  API_CONTAINER_ID="${container_id//$'\n'/}"
}

wait_for_api_ready() {
  local attempts=120
  local i
  for ((i=1; i<=attempts; i++)); do
    if curl -fsS "$API_BASE_URL/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

run_query_via_api() {
  local query="$1"
  local request_json
  request_json="$(python3 - "$query" "$RETRIEVE_K" "$RERANK_K" <<'PY'
import json
import sys

payload = {
    "query": sys.argv[1],
    "k": int(sys.argv[2]),
    "rerank_k": int(sys.argv[3]),
}
print(json.dumps(payload))
PY
)"

  if ! curl -fsS \
    -X POST "$API_BASE_URL/retrieve" \
    -H "Content-Type: application/json" \
    -d "$request_json" \
    >"$SHARED_DIR/retrieval_output.json" 2>>"$QUERY_LOG"; then
    return 1
  fi
}

run_query_once() {
  if ! run_query_via_api "$QUERY"; then
    echo "❌ Retrieval query failed."
    if [[ "$VERBOSE" == "1" ]]; then
      cat "$QUERY_LOG"
    else
      echo "📄 Query log: $QUERY_LOG"
    fi
    exit 1
  fi
}

render_query_output() {
  local query_ms="$1"
  local doc_count output_path seconds_display
  doc_count="$(index_doc_count)"
  output_path="$SHARED_DIR/retrieval_output.json"
  seconds_display="$(python3 - "$query_ms" <<'PY'
import sys
ms = int(sys.argv[1])
print(f"{ms/1000.0:.1f}")
PY
)"

  echo ""
  echo "  Searching ${doc_count} scenarios...  done in ${seconds_display}s"
  echo ""

  python3 - "$output_path" "$VERBOSE" <<'PY'
import json, math, sys
from pathlib import Path

FAISS_GAP = 0.07
CE_GAP = 1.0

path = Path(sys.argv[1])
verbose = sys.argv[2] == "1"
if not path.exists():
    print("  No matching scenario")
    raise SystemExit(0)

payload = json.loads(path.read_text(encoding="utf-8"))
results = payload.get("results", [])

if not results:
    print("  No matching scenario")
    raise SystemExit(0)

def final_score(row):
    if "final_score" in row:
        return max(0.0, min(1.0, float(row.get("final_score", 0.0))))
    # Back-compat with older payloads
    ce = float(row.get("score", 0.0))
    faiss = float(row.get("retrieval_score", 0.0))
    ce_sigmoid = 1.0 / (1.0 + math.exp(-ce))
    return max(0.0, min(1.0, (0.8 * ce_sigmoid) + (0.2 * max(0.0, min(1.0, faiss)))))

def ce_score(row):
    return float(row.get("rerank_score", row.get("score", 0.0)))

clear_count = 1
if len(results) >= 2:
    faiss_gap = abs(float(results[0].get("retrieval_score", 0.0)) - float(results[1].get("retrieval_score", 0.0)))
    ce_gap = abs(ce_score(results[0]) - ce_score(results[1]))
    if faiss_gap < FAISS_GAP or ce_gap < CE_GAP:
        clear_count = 2

best = results[:clear_count]
all_scores = [final_score(r) for r in results]

def render_bar(score, width=10):
    filled = max(0, min(width, int(round(score * width))))
    return ("█" * filled) + ("░" * (width - filled))

use_color = sys.stdout.isatty()
RESET = "\033[0m" if use_color else ""
CYAN  = "\033[36m" if use_color else ""
GREEN = "\033[32m" if use_color else ""
YELLOW= "\033[33m" if use_color else ""
BOLD  = "\033[1m"  if use_color else ""

# STRICT COLUMN WIDTHS
COL_OPT = 6
COL_ACT = 32
COL_FIT = 14
COL_MAT = 8
INNER_WIDTH = COL_OPT + COL_ACT + COL_FIT + COL_MAT

top_border = f"  {CYAN}┌─ Suggested Chaos Experiments {'─' * (INNER_WIDTH - 28)}┐{RESET}"
bottom_border = f"  {CYAN}└{'─' * INNER_WIDTH}┘{RESET}"

print(top_border)

# Header
h_opt = f"{BOLD}{'OPT':<{COL_OPT}}{RESET}"
h_act = f"{BOLD}{'ACTION':<{COL_ACT}}{RESET}"
h_fit = f"{BOLD}{'FITMENT':<{COL_FIT}}{RESET}"
h_mat = f"{BOLD}{'MATCH':<{COL_MAT}}{RESET}"
print(f"  {CYAN}│{RESET}{h_opt}{h_act}{h_fit}{h_mat}{CYAN}│{RESET}")

# Rows
for idx, row in enumerate(best, 1):
    raw_name = str(row.get("id", "unknown"))[:COL_ACT-2]
    
    # 1. Pad raw text strings first
    c_opt = f"[{idx}]".ljust(COL_OPT)
    c_act = raw_name.ljust(COL_ACT)
    
    # 2. Build the bar and pad it explicitly
    score = all_scores[idx - 1]
    bar_str = render_bar(score, width=10)
    c_fit = f"{GREEN}{bar_str}{RESET}" + (" " * (COL_FIT - 10))
    
    # 3. Format the score, pad it, then wrap in color
    c_mat_plain = f"{score:0.3f}".ljust(COL_MAT)
    c_mat = f"{YELLOW}{c_mat_plain}{RESET}"
    
    # 4. Print the line
    print(f"  {CYAN}│{RESET}{c_opt}{c_act}{c_fit}{c_mat}{CYAN}│{RESET}")

print(bottom_border)

if verbose:
    timing = results[0].get("timing_ms", {}) if results else {}
    if timing:
        print(
            "\n  Model ms: "
            f"retrieve={timing.get('retrieve', '?')}  "
            f"rerank={timing.get('rerank', '?')}  "
            f"total={timing.get('total', '?')}  "
            f"reranked={timing.get('reranked', '?')}"
        )
PY

}

# ── Setup ──

mkdir -p "$SHARED_DIR" "$HF_CACHE_DIR" "$TORCH_CACHE_DIR"

# Platform-specific backend defaults
if [[ "$BACKEND" == "auto" ]]; then
  BACKEND="vulkan"
  vlog "$HOST_OS detected; using llama.cpp Vulkan backend"
fi

# Resolve GGUF models for vulkan backend
if [[ "$BACKEND" == "vulkan" ]]; then
  if [[ -z "$LLAMA_EMBED_MODEL_PATH" ]]; then
    LLAMA_EMBED_MODEL_PATH="$ROOT_DIR/models/$GGUF_FILE"
  fi

  if [[ ! -f "$LLAMA_EMBED_MODEL_PATH" && "$AUTO_DOWNLOAD_MODEL" == "1" ]]; then
    if ! command -v curl >/dev/null 2>&1; then
      echo "Error: curl is required to auto-download the GGUF embedding model"; exit 1
    fi
  fi

  if [[ ! -f "$LLAMA_EMBED_MODEL_PATH" && "$AUTO_DOWNLOAD_MODEL" == "1" ]]; then
    status_run_quiet "📥 Downloading embedding model" download_gguf_model "$LLAMA_EMBED_MODEL_PATH"
  fi

  if [[ ! -f "$LLAMA_EMBED_MODEL_PATH" ]]; then
    echo "Error: Vulkan backend needs a GGUF embedding model file"
    echo "       Expected: $LLAMA_EMBED_MODEL_PATH"
    echo "       Set LLAMA_EMBED_MODEL or keep RETRIEVER_AUTO_DOWNLOAD_MODEL=1"
    exit 1
  fi
  LLAMA_RERANKER_MODEL_PATH=""

  LLAMA_MODEL_ABS="$(cd "$(dirname "$LLAMA_EMBED_MODEL_PATH")" && pwd)/$(basename "$LLAMA_EMBED_MODEL_PATH")"
  LLAMA_MODEL_BASENAME="$(basename "$LLAMA_MODEL_ABS")"
  if [[ "$(dirname "$LLAMA_MODEL_ABS")" != "$ROOT_DIR/models" ]]; then
    mkdir -p "$ROOT_DIR/models"
    cp -f "$LLAMA_MODEL_ABS" "$ROOT_DIR/models/$LLAMA_MODEL_BASENAME"
    LLAMA_MODEL_ABS="$ROOT_DIR/models/$LLAMA_MODEL_BASENAME"
  fi
  LLAMA_EMBED_MODEL_PATH="/models/$LLAMA_MODEL_BASENAME"
  LLAMA_MOUNT_ARGS=(-v "$(dirname "$LLAMA_MODEL_ABS"):/models$MOUNT_LABEL_SUFFIX")
fi

# ── Configure acceleration (sets BUILD_ARGS, GPU_FLAGS, DEVICE_ARGS) ──

if [[ "$ENGINE" == "podman" ]]; then
  if [[ "$DOCKERFILE" == "$ROOT_DIR/krkn-assist/Dockerfile" ]]; then
    CONTAINER_PLATFORM="linux/arm64"
  else
    podman_platform="$("$ENGINE" info --format '{{.Host.OS}}/{{.Host.Arch}}' 2>/dev/null || true)"
    case "$podman_platform" in
      linux/amd64|linux/arm64)
        CONTAINER_PLATFORM="$podman_platform"
        ;;
      */x86_64)
        CONTAINER_PLATFORM="linux/amd64"
        ;;
      */aarch64|*/arm64)
        CONTAINER_PLATFORM="linux/arm64"
        ;;
    esac
  fi
fi
if [[ -n "$CONTAINER_PLATFORM" ]]; then
  PLATFORM_BUILD_ARGS=(--platform "$CONTAINER_PLATFORM")
fi

configure_acceleration

if [[ "$VERBOSE" == "1" ]]; then
  echo "========================================"
  echo "Krkn Assist Pipeline"
  echo "========================================"
  echo "Query: ${QUERY:-<interactive>}"
  echo "Retrieve-K: $RETRIEVE_K  |  Rerank-K: $RERANK_K"
  echo "Engine: $ENGINE  |  Image: $IMAGE"
  echo "Container platform: ${CONTAINER_PLATFORM:-auto}"
  echo "Host OS: $HOST_OS  |  Backend: $BACKEND"
  echo "========================================"
  echo ""
fi

if [[ "$VERBOSE" != "1" ]]; then
  status_note "🚀 krknctl-assist is getting ready"
  status_note "🧭 Backend: $BACKEND  |  Engine: $ENGINE"
fi

TOTAL_START_MS="$(now_ms)"

# ── [1/3] Build image ──
status_run_quiet "🛠️  Preparing container runtime" ensure_podman_machine

vlog "[1/3] Building retriever container image"
STEP_START_MS="$(now_ms)"
if [[ "$FORCE_BUILD" == "1" ]]; then
  vlog "      RETRIEVER_FORCE_BUILD=1 — rebuilding image"
  status_run_quiet "🏗️  Building retrieval image" build_image
elif "$ENGINE" image exists "$IMAGE"; then
  if image_is_compatible; then
    vlog "      Image $IMAGE already present and compatible, skipping build"
    status_note "✅ Retrieval image already prepared"
  else
    vlog "      Existing image incompatible, rebuilding..."
    status_run_quiet "🏗️  Rebuilding retrieval image" build_image
  fi
else
  vlog "      Building image..."
  status_run_quiet "🏗️  Building retrieval image" build_image
fi
STEP_END_MS="$(now_ms)"
BUILD_MS="$((STEP_END_MS - STEP_START_MS))"
vlog "      Step time: ${BUILD_MS}ms"
if [[ "$VERBOSE" == "1" ]]; then
  echo "      Image ready"
fi

# ── [2/3] Ensure FAISS index ──

vlog ""
vlog "[2/3] Ensuring FAISS index exists"
STEP_START_MS="$(now_ms)"
should_reindex=0
if [[ -f "$INDEX_FILE" && -f "$META_FILE" && -f "$DOCS_CACHE_FILE" ]]; then
  if [[ "$FORCE_REINDEX" == "true" ]]; then
    vlog "      FORCE_REINDEX enabled, rebuilding"
    should_reindex=1
  elif [[ "$(index_is_stale)" == "1" ]]; then
    vlog "      FAISS index older than TTL (${INDEX_TTL_DAYS}d), rebuilding"
    should_reindex=1
  else
    vlog "      FAISS index already present, skipping indexing"
  fi
else
  vlog "      FAISS index assets missing, building now..."
  should_reindex=1
fi

if [[ "$should_reindex" == "1" ]]; then
  status_run_quiet "🧠 Building search index" run_retriever_python \
    -v "$ROOT_DIR/krkn-assist:/app$MOUNT_LABEL_SUFFIX" \
    -v "$HF_CACHE_DIR:/root/.cache/huggingface$MOUNT_LABEL_SUFFIX" \
    -v "$TORCH_CACHE_DIR:/root/.cache/torch$MOUNT_LABEL_SUFFIX" \
    -e DOCS_DIR=/app/docs \
    -e INDEX_TTL_DAYS="$INDEX_TTL_DAYS" \
    -e FORCE_REINDEX="$FORCE_REINDEX" \
    -e GITHUB_REPO="$GITHUB_REPO" \
    -e GITHUB_BRANCH="$GITHUB_BRANCH" \
    -e REPO_PATH="$REPO_PATH" \
    -e KRKN_HUB_REPO="$KRKN_HUB_REPO" \
    -e KRKN_HUB_BRANCH="$KRKN_HUB_BRANCH" \
    -e LOCAL_DOCS_PATH="$LOCAL_DOCS_ENV_PATH" \
    -e RETRIEVER_BACKEND="$BACKEND" \
    -e RETRIEVER_MODEL="$RETRIEVER_MODEL" \
    -e CROSS_ENCODER_MODEL="$CROSS_ENCODER_MODEL" \
    -e RERANK_MAX_LENGTH="$RERANK_MAX_LENGTH" \
    -e RERANK_BATCH_SIZE="$RERANK_BATCH_SIZE" \
    -e RERANK_DOC_CHARS="$RERANK_DOC_CHARS" \
    -e RERANK_THREADS="$RERANK_THREADS" \
    -e RERANK_CANDIDATE_K="$RERANK_CANDIDATE_K" \
    -e RERANK_TOP_FRACTION="$RERANK_TOP_FRACTION" \
    -e RETRIEVAL_CANDIDATE_K="$RETRIEVAL_CANDIDATE_K" \
    -e RERANK_SUPPORT_PASSAGES="$RERANK_SUPPORT_PASSAGES" \
    -e RERANK_ONNX_QUANTIZE="$RERANK_ONNX_QUANTIZE" \
    -e MIN_FAISS_SCORE="$MIN_FAISS_SCORE" \
    -e MIN_CE_SCORE="$MIN_CE_SCORE" \
    -e MIN_MATCH_SCORE="$MIN_MATCH_SCORE" \
    -e RELEVANCE_THRESHOLD="$RELEVANCE_THRESHOLD" \
    -e EXCLUDED_SCENARIO_IDS="$EXCLUDED_SCENARIO_IDS" \
    -e LLAMA_EMBED_MODEL="$LLAMA_EMBED_MODEL_PATH" \
    -e LLAMA_RERANKER_MODEL="$LLAMA_RERANKER_MODEL_PATH" \
    -e LLAMA_GPU_LAYERS="$LLAMA_GPU_LAYERS" \
    -e HF_HOME=/root/.cache/huggingface \
    -e SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface \
    -e TORCH_HOME=/root/.cache/torch \
    -w /app \
    "$IMAGE" \
    retriever.py "${DEVICE_ARGS[@]}" index
else
  status_note "✅ Search index already ready"
fi
STEP_END_MS="$(now_ms)"
INDEX_MS="$((STEP_END_MS - STEP_START_MS))"
vlog "      Step time: ${INDEX_MS}ms"

# ── [3/3] Retrieval + reranking ──

vlog ""
vlog "[3/3] Running retrieval and reranking query"
STEP_START_MS="$(now_ms)"
QUERY_LOG="$(mktemp)"
cleanup_assist_containers
vlog "      Starting debug API service on $API_BASE_URL"
vlog "      Starting compat API service on $COMPAT_BASE_URL"
status_run_quiet "🔌 Starting assist API" start_api_standby
status_run_quiet "⏳ Waiting for assist API readiness" wait_for_api_ready
status_note "✅ Assist API is ready"

if [[ "$INTERACTIVE" == "1" ]]; then
  echo ""
  printf '\033[36m[KRKN-AI]\033[0m Ready to profile your cluster. What should we test?\n'
  echo ""
  echo "(type 'exit' to quit)"
  while true; do
    printf ">> "
    if ! IFS= read -r QUERY; then
      echo ""
      break
    fi
    if [[ "$QUERY" == "exit" || "$QUERY" == "quit" ]]; then
      echo ""
      break
    fi
    if [[ -z "${QUERY// }" ]]; then
      echo "Empty query. Please try again."
      continue
    fi
    STEP_START_MS="$(now_ms)"
    if [[ "$VERBOSE" == "1" ]]; then
      if ! run_query_via_api "$QUERY"; then
        echo "Error: retrieval query failed."
        tail -n 40 "$QUERY_LOG"
        continue
      fi
    else
      if ! status_try_quiet "🔎 Searching and reranking" run_query_via_api "$QUERY"; then
        continue
      fi
    fi
    STEP_END_MS="$(now_ms)"
    QUERY_MS="$((STEP_END_MS - STEP_START_MS))"
    vlog "      Step time: ${QUERY_MS}ms"
    render_query_output "$QUERY_MS"
  done
else
  if [[ "$VERBOSE" == "1" ]]; then
    run_query_once
  else
    status_run_quiet "🔎 Searching and reranking" run_query_once
  fi
  STEP_END_MS="$(now_ms)"
  QUERY_MS="$((STEP_END_MS - STEP_START_MS))"
  vlog "      Step time: ${QUERY_MS}ms"
  render_query_output "$QUERY_MS"
fi

TOTAL_END_MS="$(now_ms)"
TOTAL_MS="$((TOTAL_END_MS - TOTAL_START_MS))"

if [[ "$VERBOSE" == "1" ]]; then
  echo ""
  echo "[verbose] Step timings"
  echo "[1/3] Building retriever container image   (${BUILD_MS}ms)"
  echo "[2/3] Ensuring FAISS index exists          (${INDEX_MS}ms)"
  if [[ "$QUERY_MS" -gt 0 ]]; then
    echo "[3/3] Running retrieval and reranking      (${QUERY_MS}ms)"
  else
    echo "[3/3] Running retrieval and reranking      (skipped)"
  fi
  echo "Total elapsed: ${TOTAL_MS}ms"
else
  status_note "🎉 Assist query flow complete"
fi
