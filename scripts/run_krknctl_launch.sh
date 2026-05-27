#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_KRKNCTL_REPO="https://github.com/krkn-chaos/krknctl.git"
DEFAULT_KRKNCTL_BRANCH="gpu_check"
DEFAULT_KRKNCTL_DIR="$HOME/.cache/krknctl-assist/krknctl"
DEFAULT_RETRIEVER_BACKEND="vulkan"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage:
  $0 [additional setup_krknctl_assist.sh options]

Launch krknctl assist interactively on the current host using the configured krknctl branch.

Defaults:
  KRKNCTL_REPO=$DEFAULT_KRKNCTL_REPO
  KRKNCTL_BRANCH=$DEFAULT_KRKNCTL_BRANCH
  KRKNCTL_DIR=$DEFAULT_KRKNCTL_DIR
  RETRIEVER_BACKEND=$DEFAULT_RETRIEVER_BACKEND

Examples:
  $0
  KRKNCTL_BRANCH=main $0
  KRKNCTL_DIR=\$HOME/tmp/krknctl $0 --force-build
EOF
  exit 0
fi

KRKNCTL_REPO="${KRKNCTL_REPO:-$DEFAULT_KRKNCTL_REPO}"
KRKNCTL_BRANCH="${KRKNCTL_BRANCH:-$DEFAULT_KRKNCTL_BRANCH}"
KRKNCTL_DIR="${KRKNCTL_DIR:-$DEFAULT_KRKNCTL_DIR}"
RETRIEVER_BACKEND="${RETRIEVER_BACKEND:-$DEFAULT_RETRIEVER_BACKEND}"

export KRKNCTL_REPO
export RETRIEVER_BACKEND

exec "$SCRIPT_DIR/setup_krknctl_assist.sh" \
  --launch \
  --krknctl-dir "$KRKNCTL_DIR" \
  --krknctl-branch "$KRKNCTL_BRANCH" \
  "$@"
