#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Backward-compatible shim: the launcher is now host-agnostic.
exec "$SCRIPT_DIR/run_krknctl_launch.sh" "$@"
