#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR=""
HOST_OS="$(uname -s)"

case "$HOST_OS" in
  Darwin)
    preferred_dirs=(
      "/opt/homebrew/bin"
      "/usr/local/bin"
      "$HOME/.local/bin"
    )
    ;;
  *)
    preferred_dirs=(
      "/usr/local/bin"
      "$HOME/.local/bin"
    )
    ;;
esac

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

target="$TARGET_DIR/krknctl-assist"

# Install an executable wrapper instead of a symlink so we don't depend on
# git tracking +x bits for repo files.
tmp_template="$(mktemp "$TARGET_DIR/krknctl-assist.tmp.XXXXXX")"
tmp_out="$(mktemp "$TARGET_DIR/krknctl-assist.tmp.XXXXXX")"

cat >"$tmp_template" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

assist_dir="${KRKNCTL_ASSIST_HOME:-__KRKNCTL_ASSIST_ROOT__}"

if [[ ! -d "$assist_dir" ]]; then
  if [[ -z "${KRKNCTL_ASSIST_HOME:-}" && -d "$HOME/krknctl-assist" ]]; then
    assist_dir="$HOME/krknctl-assist"
  else
    echo "krknctl-assist: repo not found at $assist_dir" >&2
    echo "Set KRKNCTL_ASSIST_HOME to your clone, or clone it to ~/krknctl-assist." >&2
    exit 1
  fi
fi

exec bash "$assist_dir/scripts/run_krknctl_launch.sh" "$@"
EOF

# Substitute the install-time repo path into the wrapper, without expanding
# any of the runtime variables ($HOME, $@, etc).
sed "s|__KRKNCTL_ASSIST_ROOT__|$ROOT_DIR|g" "$tmp_template" >"$tmp_out"
rm -f "$tmp_template"
chmod 0755 "$tmp_out"
mv -f "$tmp_out" "$target"
chmod 0755 "$target"

cat <<EOF
Installed: $target

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
