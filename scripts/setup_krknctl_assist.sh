#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${KRKNCTL_ASSIST_IMAGE:-quay.io/krkn-chaos/krknctl-assist:faiss-latest}"
KRKNCTL_REPO="${KRKNCTL_REPO:-https://github.com/krkn-chaos/krknctl.git}"
KRKNCTL_BRANCH="${KRKNCTL_BRANCH:-gpu_check}"
KRKNCTL_DIR="${KRKNCTL_DIR:-$HOME/krknctl}"
QUERY="${KRKNCTL_ASSIST_QUERY:-how do i run a pod deletion scenario}"
EXPECTED_SCENARIO="${KRKNCTL_ASSIST_EXPECTED:-pod-scenarios}"
GITHUB_REPO="${GITHUB_REPO:-https://github.com/krkn-chaos/website}"
GITHUB_BRANCH="${GITHUB_BRANCH:-main}"
REPO_PATH="${REPO_PATH:-content/en/docs}"
KRKN_HUB_REPO="${KRKN_HUB_REPO:-https://github.com/krkn-chaos/krkn-hub}"
KRKN_HUB_BRANCH="${KRKN_HUB_BRANCH:-}"
LOCAL_DOCS_PATH="${LOCAL_DOCS_PATH:-}"
MODE="verify"
FORCE_BUILD=0
SKIP_BREW=0
SKIP_KRKNCTL=0
HOST_OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
DOCKERFILE_PATH="${KRKNCTL_ASSIST_DOCKERFILE:-}"
IMAGE_BUILD_BACKEND=""
CONTAINER_PLATFORM=""
BUILD_ARGS=()
LOG_DIR="${KRKNCTL_ASSIST_LOG_DIR:-$ROOT_DIR/setup-logs}"
LOG_FILE="$LOG_DIR/krknctl-assist-setup-$(date -u +%Y%m%dT%H%M%SZ).log"
HEALTH_TIMEOUT_SECONDS="${KRKNCTL_ASSIST_HEALTH_TIMEOUT_SECONDS:-600}"
HF_CACHE_DIR="${KRKNCTL_ASSIST_HF_CACHE_DIR:-$ROOT_DIR/.cache/huggingface}"
TORCH_CACHE_DIR="${KRKNCTL_ASSIST_TORCH_CACHE_DIR:-$ROOT_DIR/.cache/torch}"
GGUF_REPO="${RETRIEVER_GGUF_REPO:-Qwen/Qwen3-Embedding-0.6B-GGUF}"
GGUF_FILE="${RETRIEVER_GGUF_FILE:-Qwen3-Embedding-0.6B-f16.gguf}"
AUTO_DOWNLOAD_MODEL="${RETRIEVER_AUTO_DOWNLOAD_MODEL:-1}"
HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}"
SMOKE_CONTAINER=""

usage() {
  cat <<EOF
Usage: $0 [options]

Build, smoke-test, and verify krknctl assist with the local krkn-assist image.

Options:
  --setup-only              Setup/build/smoke-test only; do not run krknctl verification
  --verify                  Run full non-interactive krknctl verification (default)
  --launch                  Setup everything, then launch krknctl assist interactively
  --cleanup                 Remove krknctl assist containers and exit
  --query TEXT              Verification query (default: "$QUERY")
  --expect SCENARIO         Expected scenario in verification output
  --image TAG               Image tag krknctl should use
  --krknctl-dir PATH        krknctl checkout/install directory
  --krknctl-branch BRANCH   krknctl branch to build
  --force-build             Rebuild image even if it already exists
  --skip-brew               Do not install Homebrew packages
  --skip-krknctl            Do not clone/build/run krknctl
  Env: KRKNCTL_ASSIST_HEALTH_TIMEOUT_SECONDS
                           Override smoke-test readiness wait in seconds
  -h, --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup-only) MODE="setup-only"; shift ;;
    --verify) MODE="verify"; shift ;;
    --launch) MODE="launch"; shift ;;
    --cleanup) MODE="cleanup"; shift ;;
    --query) QUERY="${2:?missing value for --query}"; shift 2 ;;
    --expect) EXPECTED_SCENARIO="${2:?missing value for --expect}"; shift 2 ;;
    --image) IMAGE="${2:?missing value for --image}"; shift 2 ;;
    --krknctl-dir) KRKNCTL_DIR="${2:?missing value for --krknctl-dir}"; shift 2 ;;
    --krknctl-branch) KRKNCTL_BRANCH="${2:?missing value for --krknctl-branch}"; shift 2 ;;
    --force-build) FORCE_BUILD=1; shift ;;
    --skip-brew) SKIP_BREW=1; shift ;;
    --skip-krknctl) SKIP_KRKNCTL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

log() {
  printf '%s\n' "$*" | tee -a "$LOG_FILE"
}

warn() {
  log "⚠️  $*"
}

fail() {
  log "❌ $*"
  log "Log: $LOG_FILE"
  exit 1
}

run() {
  log "+ $*"
  set +e
  "$@" 2>&1 | tee -a "$LOG_FILE"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 ]]; then
    fail "Command failed: $*"
  fi
}

run_quiet() {
  log "+ $*"
  if ! "$@" >>"$LOG_FILE" 2>&1; then
    tail -n 80 "$LOG_FILE"
    fail "Command failed: $*"
  fi
}

have() {
  command -v "$1" >/dev/null 2>&1
}

download_gguf_model() {
  local out_path="$1"
  local url="https://huggingface.co/${GGUF_REPO}/resolve/main/${GGUF_FILE}"
  mkdir -p "$(dirname "$out_path")"
  local curl_args=(-L --fail --retry 3 --continue-at - -o "$out_path")
  if [[ -n "$HF_TOKEN_VALUE" ]]; then
    curl_args+=(-H "Authorization: Bearer $HF_TOKEN_VALUE")
  fi
  run curl "${curl_args[@]}" "$url"
}

stop_stale_krknctl_processes() {
  have pgrep || return 0

  local pids
  pids="$(pgrep -f 'krknctl[[:space:]]+assist[[:space:]]+run' 2>/dev/null \
    | awk -v self="$$" '$1 != self {print $1}')" || true
  [[ -n "$pids" ]] || return 0

  log "Stopping stale krknctl assist process(es): $pids"
  # shellcheck disable=SC2086
  kill $pids >>"$LOG_FILE" 2>&1 || true
  sleep 2

  local remaining
  remaining="$(pgrep -f 'krknctl[[:space:]]+assist[[:space:]]+run' 2>/dev/null \
    | awk -v self="$$" '$1 != self {print $1}')" || true
  if [[ -n "$remaining" ]]; then
    log "Force-stopping stale krknctl assist process(es): $remaining"
    # shellcheck disable=SC2086
    kill -9 $remaining >>"$LOG_FILE" 2>&1 || true
  fi
}

add_homebrew_path() {
  if [[ -d /opt/homebrew/bin ]]; then
    export PATH="/opt/homebrew/bin:$PATH"
  fi
  if [[ -d /usr/local/bin ]]; then
    export PATH="/usr/local/bin:$PATH"
  fi
}

ensure_brew_packages() {
  [[ "$HOST_OS" == "darwin" ]] || return 0
  add_homebrew_path

  if ! have brew; then
    fail "Homebrew is required on macOS. Install it first, then rerun this script."
  fi

  [[ "$SKIP_BREW" == "0" ]] || return 0

  local packages=(git curl podman go pkgconf gpgme)
  local missing=()
  local pkg
  for pkg in "${packages[@]}"; do
    if ! brew list --versions "$pkg" >/dev/null 2>&1; then
      missing+=("$pkg")
    fi
  done

  if [[ ${#missing[@]} -gt 0 ]]; then
    run brew install "${missing[@]}"
  else
    log "✅ Homebrew prerequisites already installed"
  fi
}

ensure_basic_tools() {
  add_homebrew_path
  local cmd
  for cmd in git curl podman python3; do
    have "$cmd" || fail "Missing required command: $cmd"
  done
  if [[ "$SKIP_KRKNCTL" == "0" ]]; then
    have go || fail "Missing required command: go"
  fi
}

ensure_podman_ready() {
  if [[ "$HOST_OS" == "darwin" ]]; then
    export CONTAINERS_MACHINE_PROVIDER="${CONTAINERS_MACHINE_PROVIDER:-libkrun}"

    if ! podman machine list --format '{{.Name}}' 2>/dev/null | grep -q .; then
      run podman machine init --cpus 4 --memory 8192 --disk-size 100
    fi

    local machine
    machine="$(podman machine list --format '{{.Name}}' 2>/dev/null | head -n1 | sed 's/[*[:space:]]//g')"
    [[ -n "$machine" ]] || fail "Unable to determine Podman machine name"

    if ! podman machine start "$machine" >>"$LOG_FILE" 2>&1; then
      if ! grep -qi "already running" "$LOG_FILE"; then
        fail "Podman machine failed to start"
      fi
      log "✅ Podman machine already running"
    else
      log "✅ Podman machine running: $machine"
    fi

    local socket_path
    socket_path="$(podman machine inspect --format '{{.ConnectionInfo.PodmanSocket.Path}}' "$machine" 2>/dev/null || true)"
    if [[ -n "$socket_path" ]]; then
      mkdir -p "$HOME/.local/share/containers/podman/machine"
      ln -sf "$socket_path" "$HOME/.local/share/containers/podman/machine/podman.sock"
      log "✅ Refreshed Podman socket symlink for krknctl"
    else
      warn "Could not discover Podman socket path; krknctl runtime detection may fail"
    fi
  fi

  run_quiet podman info
  log "✅ Podman is reachable"
}

stop_assist_containers_on_8080() {
  stop_stale_krknctl_processes

  local ids
  ids="$(podman ps --format '{{.ID}} {{.Image}} {{.Names}} {{.Ports}}' 2>/dev/null \
    | awk '/8080/ && /krknctl-assist|krkn-assist/ {print $1}')"
  if [[ -n "$ids" ]]; then
    log "Removing old krkn assist containers bound to port 8080"
    # shellcheck disable=SC2086
    run podman rm -f $ids
  fi

  ids="$(podman ps -a --format '{{.ID}} {{.Names}} {{.Image}}' 2>/dev/null \
    | awk '/krknctl-assist-rag|krknctl-assist-smoke/ {print $1}')"
  if [[ -n "$ids" ]]; then
    log "Removing stale krkn assist containers"
    # shellcheck disable=SC2086
    run podman rm -f $ids
  fi

  if have lsof && lsof -nP -iTCP:8080 -sTCP:LISTEN >/tmp/krknctl-assist-port-8080.$$ 2>/dev/null; then
    if grep -q . /tmp/krknctl-assist-port-8080.$$; then
      cat /tmp/krknctl-assist-port-8080.$$ | tee -a "$LOG_FILE"
      rm -f /tmp/krknctl-assist-port-8080.$$
      fail "Port 8080 is still in use by a non-assist process. Free it and rerun."
    fi
  fi
  rm -f /tmp/krknctl-assist-port-8080.$$
}

hash_repo_files() {
  (
    cd "$ROOT_DIR/krkn-assist"
    if have shasum; then
      shasum -a 256 "$@"
    else
      sha256sum "$@"
    fi
  )
}

repo_files_exist() {
  (
    cd "$ROOT_DIR/krkn-assist"
    local relpath
    for relpath in "$@"; do
      [[ -e "$relpath" ]] || return 1
    done
  )
}

image_matches_checkout() {
  podman image exists "$IMAGE" >/dev/null 2>&1 || return 1

  local files=(
    "api_server.py"
    "assist/api.py"
    "assist/ingestion.py"
    "assist/policy.py"
    "assist/service.py"
    "assist/settings.py"
    "assist/ranking.py"
    "faiss-index/krkn-scenarios.index"
    "faiss-index/krkn-scenarios.meta"
    "faiss-index/krkn-scenarios.docs.json"
    "faiss-index/krkn-scenarios.list.txt"
    "data.csv"
  )

  repo_files_exist "${files[@]}" || return 1

  local host_file image_file
  host_file="$(mktemp)"
  image_file="$(mktemp)"

  if ! hash_repo_files "${files[@]}" | awk '{print $1 " " $2}' >"$host_file"; then
    rm -f "$host_file" "$image_file"
    return 1
  fi

  if ! podman run --rm --entrypoint sha256sum "$IMAGE" \
    /app/api_server.py \
    /app/assist/api.py \
    /app/assist/ingestion.py \
    /app/assist/policy.py \
    /app/assist/service.py \
    /app/assist/settings.py \
    /app/assist/ranking.py \
    /app/faiss-index/krkn-scenarios.index \
    /app/faiss-index/krkn-scenarios.meta \
    /app/faiss-index/krkn-scenarios.docs.json \
    /app/faiss-index/krkn-scenarios.list.txt \
    /app/data.csv \
    | sed 's#/app/##' | awk '{print $1 " " $2}' >"$image_file"; then
    rm -f "$host_file" "$image_file"
    return 1
  fi

  set +e
  diff -q "$host_file" "$image_file" >/dev/null 2>&1
  local status=$?
  set -e
  rm -f "$host_file" "$image_file"
  return "$status"
}

configure_image_build() {
  local podman_platform
  local requested_backend="${RETRIEVER_BACKEND:-auto}"
  podman_platform="$(podman info --format '{{.Host.OS}}/{{.Host.Arch}}' 2>/dev/null || true)"

  if [[ -z "$DOCKERFILE_PATH" ]]; then
    if [[ "$HOST_OS" == "darwin" ]]; then
      DOCKERFILE_PATH="$ROOT_DIR/krkn-assist/Dockerfile"
    else
      DOCKERFILE_PATH="$ROOT_DIR/krkn-assist/Dockerfile.linux"
    fi
  fi

  if [[ "$DOCKERFILE_PATH" == "$ROOT_DIR/krkn-assist/Dockerfile" ]]; then
    CONTAINER_PLATFORM="linux/arm64"
  else
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
      *)
        if [[ "$(uname -m)" == "x86_64" ]]; then
          CONTAINER_PLATFORM="linux/amd64"
        else
          CONTAINER_PLATFORM="linux/arm64"
        fi
        ;;
    esac
  fi

  case "$requested_backend" in
    vulkan)
      IMAGE_BUILD_BACKEND="vulkan"
      ;;
    auto)
      if [[ "$HOST_OS" == "darwin" ]]; then
        IMAGE_BUILD_BACKEND="vulkan"
      else
        IMAGE_BUILD_BACKEND="cpu"
      fi
      ;;
    *)
      IMAGE_BUILD_BACKEND="cpu"
      ;;
  esac

  BUILD_ARGS=(
    --platform "$CONTAINER_PLATFORM"
    --build-arg "GGML_BACKEND=$IMAGE_BUILD_BACKEND"
  )
}

build_image() {
  run podman build -f "$DOCKERFILE_PATH" "${BUILD_ARGS[@]}" -t "$IMAGE" "$ROOT_DIR"
}

warm_index_assets() {
  local mount_suffix=""
  local warm_index_dir="faiss-index.warm"
  local warm_index_path="$ROOT_DIR/krkn-assist/$warm_index_dir"
  local final_index_path="$ROOT_DIR/krkn-assist/faiss-index"
  local warm_backend="${RETRIEVER_BACKEND:-}"
  local warm_model_host_path="${LLAMA_EMBED_MODEL:-}"
  local warm_model_container_path=""
  local warm_mount_args=()
  local warm_run_args=()
  if [[ "$HOST_OS" != "darwin" ]]; then
    mount_suffix=":Z"
  fi

  if [[ -z "$warm_backend" ]]; then
    if [[ "$HOST_OS" == "darwin" ]]; then
      warm_backend="vulkan"
    else
      warm_backend="torch"
    fi
  fi

  if [[ "$warm_backend" == "vulkan" ]]; then
    if [[ -z "$warm_model_host_path" ]]; then
      warm_model_host_path="$ROOT_DIR/models/$GGUF_FILE"
    fi

    if [[ ! -f "$warm_model_host_path" ]]; then
      if [[ "$AUTO_DOWNLOAD_MODEL" != "1" ]]; then
        fail "Warm index requires GGUF model at $warm_model_host_path or RETRIEVER_AUTO_DOWNLOAD_MODEL=1"
      fi
      log "Downloading GGUF embedding model for warm index: ${GGUF_REPO}/${GGUF_FILE}"
      download_gguf_model "$warm_model_host_path"
    fi

    local warm_model_abs
    warm_model_abs="$(cd "$(dirname "$warm_model_host_path")" && pwd)/$(basename "$warm_model_host_path")"
    local warm_model_basename
    warm_model_basename="$(basename "$warm_model_abs")"
    warm_model_container_path="/models/$warm_model_basename"
    warm_mount_args=(-v "$(dirname "$warm_model_abs"):/models$mount_suffix")

    if [[ "$HOST_OS" == "darwin" ]]; then
      warm_run_args=(--device /dev/dri)
    fi
  fi

  mkdir -p "$HF_CACHE_DIR" "$TORCH_CACHE_DIR"
  rm -rf "$warm_index_path"
  log "Precomputing FAISS assets before final image build (backend=$warm_backend)"

  local env_args=(
    -e "GITHUB_REPO=$GITHUB_REPO"
    -e "GITHUB_BRANCH=$GITHUB_BRANCH"
    -e "REPO_PATH=$REPO_PATH"
    -e "KRKN_HUB_REPO=$KRKN_HUB_REPO"
    -e "INDEX_DIR=$warm_index_dir"
    -e "HF_HOME=/root/.cache/huggingface"
    -e "SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface"
    -e "TORCH_HOME=/root/.cache/torch"
    -e "RETRIEVER_BACKEND=$warm_backend"
    -e "LOG_LEVEL=WARNING"
    -e "TRANSFORMERS_VERBOSITY=error"
    -e "HF_HUB_DISABLE_PROGRESS_BARS=1"
    -e "TOKENIZERS_PARALLELISM=false"
  )
  local passthrough_vars=(
    RETRIEVER_BACKEND
    RETRIEVER_MODEL
    RETRIEVER_DEVICE
    RETRIEVER_CPU_ONLY
    CROSS_ENCODER_MODEL
    LLAMA_EMBED_MODEL
    LLAMA_GPU_LAYERS
    RELEVANCE_THRESHOLD
    MIN_MATCH_SCORE
    MIN_FAISS_SCORE
    MIN_CE_SCORE
    EXCLUDED_SCENARIO_IDS
    MIN_MULTI_SCORE
    MULTI_MATCH_SCORE_GAP
    MAX_MULTI_SCENARIOS
  )
  local env_name
  for env_name in "${passthrough_vars[@]}"; do
    if [[ -n "${!env_name:-}" ]]; then
      env_args+=(-e "$env_name=${!env_name}")
    fi
  done
  if [[ -n "$warm_model_container_path" ]]; then
    env_args+=(-e "LLAMA_EMBED_MODEL=$warm_model_container_path")
  fi
  if [[ -n "$KRKN_HUB_BRANCH" ]]; then
    env_args+=(-e "KRKN_HUB_BRANCH=$KRKN_HUB_BRANCH")
  fi
  if [[ -n "$LOCAL_DOCS_PATH" ]]; then
    env_args+=(-e "LOCAL_DOCS_PATH=$LOCAL_DOCS_PATH")
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    env_args+=(-e "HF_TOKEN=$HF_TOKEN")
  fi
  if [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    env_args+=(-e "HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN")
  fi

  log "Running warm index container; detailed indexing output is in $LOG_FILE"
  if ! podman run --rm \
    ${warm_run_args[@]+"${warm_run_args[@]}"} \
    -v "$ROOT_DIR/krkn-assist:/app$mount_suffix" \
    -v "$HF_CACHE_DIR:/root/.cache/huggingface$mount_suffix" \
    -v "$TORCH_CACHE_DIR:/root/.cache/torch$mount_suffix" \
    ${warm_mount_args[@]+"${warm_mount_args[@]}"} \
    ${env_args[@]+"${env_args[@]}"} \
    -w /app \
    "$IMAGE" \
    python3 retriever.py index >>"$LOG_FILE" 2>&1; then
    tail -n 120 "$LOG_FILE"
    fail "FAISS warm-up failed"
  fi
  log "✅ FAISS assets precomputed"

  test -s "$warm_index_path/krkn-scenarios.index" \
    || fail "FAISS warm-up did not produce krkn-scenarios.index"
  test -s "$warm_index_path/krkn-scenarios.meta" \
    || fail "FAISS warm-up did not produce krkn-scenarios.meta"
  test -s "$warm_index_path/krkn-scenarios.docs.json" \
    || fail "FAISS warm-up did not produce krkn-scenarios.docs.json"

  mkdir -p "$final_index_path"
  cp -f "$warm_index_path"/krkn-scenarios.* "$final_index_path"/
  rm -rf "$final_index_path/merged-scenarios"
  if [[ -d "$warm_index_path/merged-scenarios" ]]; then
    cp -R "$warm_index_path/merged-scenarios" "$final_index_path/merged-scenarios"
  fi
  rm -rf "$warm_index_path"
}

build_assist_image() {
  if [[ "$FORCE_BUILD" == "0" ]] && image_matches_checkout; then
    log "✅ Image already present and matches this checkout: $IMAGE"
    return 0
  elif [[ "$FORCE_BUILD" == "0" ]] && podman image exists "$IMAGE" >/dev/null 2>&1; then
    warn "Existing $IMAGE does not match this checkout; rebuilding"
  fi

  log "Building bootstrap image from repo root so Dockerfile COPY paths resolve"
  build_image
  warm_index_assets
  log "Rebuilding final image with warmed FAISS assets"
  build_image
}

wait_for_health() {
  local base_url="$1"
  local attempts="${2:-$HEALTH_TIMEOUT_SECONDS}"
  local i
  for ((i=1; i<=attempts; i++)); do
    if curl -fsS "$base_url/health" >>"$LOG_FILE" 2>&1; then
      log "✅ Assist service healthy at $base_url"
      return 0
    fi
    if [[ "$i" == "1" || "$((i % 10))" == "0" ]]; then
      log "Waiting for assist service... ($i/$attempts)"
    fi
    sleep 1
  done
  return 1
}

smoke_test_image() {
  stop_assist_containers_on_8080
  SMOKE_CONTAINER="krknctl-assist-smoke-$$"
  podman rm -f "$SMOKE_CONTAINER" >>"$LOG_FILE" 2>&1 || true

  local env_args=()
  local passthrough_vars=(
    RETRIEVER_BACKEND
    RETRIEVER_MODEL
    RETRIEVER_DEVICE
    RETRIEVER_CPU_ONLY
    CROSS_ENCODER_MODEL
    LLAMA_EMBED_MODEL
    LLAMA_GPU_LAYERS
    RELEVANCE_THRESHOLD
    MIN_MATCH_SCORE
    MIN_FAISS_SCORE
    MIN_CE_SCORE
    EXCLUDED_SCENARIO_IDS
    MIN_MULTI_SCORE
    MULTI_MATCH_SCORE_GAP
    MAX_MULTI_SCENARIOS
  )
  local env_name
  for env_name in "${passthrough_vars[@]}"; do
    if [[ -n "${!env_name:-}" ]]; then
      env_args+=(-e "$env_name=${!env_name}")
    fi
  done

  local container_id
  container_id="$(podman run -d --name "$SMOKE_CONTAINER" \
    ${env_args[@]+"${env_args[@]}"} \
    -p 127.0.0.1:8080:8080 "$IMAGE")" \
    || fail "Unable to start smoke-test container"
  log "Started smoke-test container: $container_id"

  if ! wait_for_health "http://127.0.0.1:8080"; then
    podman logs "$SMOKE_CONTAINER" 2>&1 | tail -n 120 | tee -a "$LOG_FILE" || true
    podman rm -f "$SMOKE_CONTAINER" >>"$LOG_FILE" 2>&1 || true
    SMOKE_CONTAINER=""
    fail "Assist image did not become healthy within ${HEALTH_TIMEOUT_SECONDS}s"
  fi

  local response
  response="$(curl -fsS -X POST http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "$(python3 - "$QUERY" <<'PY'
import json
import sys
print(json.dumps({"messages": [{"role": "user", "content": sys.argv[1]}]}))
PY
)")" || fail "Compat API smoke query failed"

  printf '%s\n' "$response" >>"$LOG_FILE"
  set +e
  python3 - "$response" "$EXPECTED_SCENARIO" <<'PY' | tee -a "$LOG_FILE"
import json
import sys

payload = json.loads(sys.argv[1])
expected = sys.argv[2]
content = payload.get("choices", [{}])[0].get("message", {}).get("content", "")
scenario = payload.get("scenario_name")
print(f"Smoke API response: {content}")
if expected and scenario != expected and expected not in content:
    print(f"Expected scenario {expected!r}, got scenario_name={scenario!r}, content={content!r}")
    raise SystemExit(1)
PY
  local status=${PIPESTATUS[0]}
  set -e
  [[ "$status" -eq 0 ]] || fail "Smoke query did not return expected scenario"

  podman rm -f "$SMOKE_CONTAINER" >>"$LOG_FILE" 2>&1 || true
  SMOKE_CONTAINER=""
  log "✅ Direct image smoke test passed"
}

verify_image_provenance() {
  local host_file image_file
  local files=(
    "api_server.py"
    "assist/api.py"
    "assist/ingestion.py"
    "assist/policy.py"
    "assist/service.py"
    "assist/settings.py"
    "assist/ranking.py"
    "faiss-index/krkn-scenarios.index"
    "faiss-index/krkn-scenarios.meta"
    "faiss-index/krkn-scenarios.docs.json"
    "faiss-index/krkn-scenarios.list.txt"
    "data.csv"
  )

  host_file="$(mktemp)"
  image_file="$(mktemp)"

  if ! hash_repo_files "${files[@]}" | awk '{print $1 " " $2}' >"$host_file"; then
    rm -f "$host_file" "$image_file"
    fail "Unable to hash host files for provenance check"
  fi

  if ! podman run --rm --entrypoint sha256sum "$IMAGE" \
    /app/api_server.py \
    /app/assist/api.py \
    /app/assist/ingestion.py \
    /app/assist/policy.py \
    /app/assist/service.py \
    /app/assist/settings.py \
    /app/assist/ranking.py \
    /app/faiss-index/krkn-scenarios.index \
    /app/faiss-index/krkn-scenarios.meta \
    /app/faiss-index/krkn-scenarios.docs.json \
    /app/faiss-index/krkn-scenarios.list.txt \
    /app/data.csv \
    | sed 's#/app/##' | awk '{print $1 " " $2}' >"$image_file"; then
    rm -f "$host_file" "$image_file"
    fail "Unable to hash image files for provenance check"
  fi

  if ! diff -u "$host_file" "$image_file" >>"$LOG_FILE" 2>&1; then
    rm -f "$host_file" "$image_file"
    fail "Image provenance check failed; container contents do not match this checkout"
  fi

  rm -f "$host_file" "$image_file"
  log "✅ Image provenance matches this checkout"
}

ensure_krknctl_checkout() {
  [[ "$SKIP_KRKNCTL" == "0" ]] || return 0

  local fetch_ref="$KRKNCTL_BRANCH"
  local checkout_ref="$KRKNCTL_BRANCH"
  local is_pr_ref=0

  if [[ "$KRKNCTL_BRANCH" =~ ^pr-([0-9]+)$ ]]; then
    fetch_ref="pull/${BASH_REMATCH[1]}/head"
    is_pr_ref=1
  elif [[ "$KRKNCTL_BRANCH" =~ ^(refs/)?pull/([0-9]+)/head$ ]]; then
    checkout_ref="pr-${BASH_REMATCH[2]}"
    fetch_ref="pull/${BASH_REMATCH[2]}/head"
    is_pr_ref=1
  fi

  if [[ ! -d "$KRKNCTL_DIR/.git" ]]; then
    mkdir -p "$(dirname "$KRKNCTL_DIR")"
    run git clone "$KRKNCTL_REPO" "$KRKNCTL_DIR"
  fi

  run git -C "$KRKNCTL_DIR" fetch origin "$fetch_ref"

  if [[ "$is_pr_ref" == "1" ]]; then
    local current_branch
    current_branch="$(git -C "$KRKNCTL_DIR" branch --show-current)"

    if [[ "$current_branch" == "$checkout_ref" ]]; then
      if git -C "$KRKNCTL_DIR" diff --quiet && git -C "$KRKNCTL_DIR" diff --cached --quiet; then
        run git -C "$KRKNCTL_DIR" checkout -B "$checkout_ref" FETCH_HEAD
      else
        warn "krknctl checkout has local changes; staying on $current_branch"
      fi
    elif git -C "$KRKNCTL_DIR" show-ref --verify --quiet "refs/heads/$checkout_ref"; then
      if ! git -C "$KRKNCTL_DIR" checkout "$checkout_ref" >>"$LOG_FILE" 2>&1; then
        fail "Could not checkout $checkout_ref in $KRKNCTL_DIR. Resolve local changes and rerun."
      fi
      if git -C "$KRKNCTL_DIR" diff --quiet && git -C "$KRKNCTL_DIR" diff --cached --quiet; then
        run git -C "$KRKNCTL_DIR" checkout -B "$checkout_ref" FETCH_HEAD
      else
        warn "krknctl checkout has local changes; staying on $checkout_ref"
      fi
    else
      run git -C "$KRKNCTL_DIR" checkout -B "$checkout_ref" FETCH_HEAD
    fi

    log "✅ krknctl branch ready: $checkout_ref"
    run go -C "$KRKNCTL_DIR" build -o krknctl .
    log "✅ krknctl binary built: $KRKNCTL_DIR/krknctl"
    return 0
  fi

  if ! git -C "$KRKNCTL_DIR" checkout "$checkout_ref" >>"$LOG_FILE" 2>&1; then
    local current_branch
    current_branch="$(git -C "$KRKNCTL_DIR" branch --show-current)"
    if [[ "$current_branch" != "$checkout_ref" ]]; then
      fail "Could not checkout $checkout_ref in $KRKNCTL_DIR. Resolve local changes and rerun."
    fi
    warn "krknctl checkout has local changes; staying on $current_branch"
  else
    log "✅ krknctl branch ready: $checkout_ref"
  fi

  run go -C "$KRKNCTL_DIR" build -o krknctl .
  log "✅ krknctl binary built: $KRKNCTL_DIR/krknctl"
}

verify_krknctl_noninteractive() {
  [[ "$SKIP_KRKNCTL" == "0" ]] || return 0
  stop_assist_containers_on_8080

  log "Running non-interactive krknctl assist verification"
  set +e
  python3 - "$KRKNCTL_DIR/krknctl" "$QUERY" "$EXPECTED_SCENARIO" "$LOG_FILE" <<'PY'
import os
import signal
import subprocess
import sys

binary, query, expected, log_file = sys.argv[1:5]
env = os.environ.copy()
env["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + env.get("PATH", "")

proc = subprocess.Popen(
    [binary, "assist", "run"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env=env,
)

try:
    output, _ = proc.communicate(f"{query}\n\nexit\n", timeout=420)
except subprocess.TimeoutExpired:
    proc.send_signal(signal.SIGINT)
    try:
        output, _ = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate()
    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write(output)
    print("krknctl verification timed out")
    raise SystemExit(1)

with open(log_file, "a", encoding="utf-8") as handle:
    handle.write(output)

print(output)
if proc.returncode not in (0, None):
    print(f"krknctl exited with status {proc.returncode}")
    raise SystemExit(proc.returncode)
if "service healthy" not in output and "assist service is ready" not in output:
    print("krknctl did not report a healthy assist service")
    raise SystemExit(1)
if expected and expected not in output:
    print(f"Expected scenario {expected!r} not found in krknctl output")
    raise SystemExit(1)
PY
  local status=$?
  set -e
  stop_assist_containers_on_8080
  [[ "$status" -eq 0 ]] || fail "krknctl assist verification failed"
  log "✅ krknctl assist verification passed"
}

launch_krknctl() {
  [[ "$SKIP_KRKNCTL" == "0" ]] || return 0
  stop_assist_containers_on_8080
  log "Launching krknctl assist interactively"
  local old_stty=""
  if [[ -t 0 ]] && have stty; then
    old_stty="$(stty -g 2>/dev/null || true)"
    stty susp undef 2>/dev/null || true
  fi
  set +e
  PATH="/opt/homebrew/bin:/usr/local/bin:$PATH" "$KRKNCTL_DIR/krknctl" assist run
  local status=$?
  set -e
  if [[ -n "$old_stty" ]]; then
    stty "$old_stty" 2>/dev/null || true
  fi
  stop_assist_containers_on_8080
  return "$status"
}

cleanup() {
  if [[ -n "$SMOKE_CONTAINER" ]]; then
    podman rm -f "$SMOKE_CONTAINER" >>"$LOG_FILE" 2>&1 || true
  fi
}
trap cleanup EXIT

log "🚀 krknctl assist setup started"
log "Repo: $ROOT_DIR"
log "Image: $IMAGE"
log "Mode: $MODE"
log "Log: $LOG_FILE"

ensure_brew_packages
ensure_basic_tools
ensure_podman_ready
configure_image_build
log "Dockerfile: $DOCKERFILE_PATH"
log "Container platform: $CONTAINER_PLATFORM"
log "Image build backend: $IMAGE_BUILD_BACKEND"

if [[ "$MODE" == "cleanup" ]]; then
  stop_assist_containers_on_8080
  log "✅ krknctl assist cleanup finished"
  log "Log: $LOG_FILE"
  exit 0
fi

build_assist_image
smoke_test_image
verify_image_provenance
ensure_krknctl_checkout

case "$MODE" in
  setup-only) ;;
  verify) verify_krknctl_noninteractive ;;
  launch) launch_krknctl ;;
esac

log "🎉 krknctl assist setup finished successfully"
log "Log: $LOG_FILE"
