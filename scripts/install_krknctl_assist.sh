#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="$ROOT_DIR/krknctl-assist"
TARGET_DIR=""
preferred_dirs=(
  "/opt/homebrew/bin"
  "/usr/local/bin"
  "$HOME/.local/bin"
)

for dir in "${preferred_dirs[@]}"; do
  if [[ "$dir" == "$HOME/.local/bin" ]]; then
    mkdir -p "$dir"
  fi
  if [[ -d "$dir" && -w "$dir" ]]; then
    TARGET_DIR="$dir"
    break
  fi
done

if [[ -z "$TARGET_DIR" ]]; then
  for dir in ${PATH//:/ }; do
    if [[ -n "$dir" && -d "$dir" && -w "$dir" ]]; then
      TARGET_DIR="$dir"
      break
    fi
  done
fi

[[ -n "$TARGET_DIR" ]] || {
  echo "Could not find a writable install directory." >&2
  exit 1
}

ln -sf "$SOURCE" "$TARGET_DIR/krknctl-assist"

cat <<EOF
Installed: $TARGET_DIR/krknctl-assist

Run:
  krknctl-assist
EOF

if ! command -v krknctl-assist >/dev/null 2>&1; then
  cat <<EOF

Note:
  $TARGET_DIR is not currently on your PATH in this shell.
  Add it to your shell config, then open a new shell.
EOF
fi
