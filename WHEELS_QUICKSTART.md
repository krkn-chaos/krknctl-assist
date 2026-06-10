# Quick Start: Pre-compiled Wheels

## TL;DR

To avoid 1.5 hours of compilation in GitHub Actions, build the ARM64 wheel locally and upload it:

```bash
# 1. Build wheel (10-15 min on native ARM64)
./scripts/build_wheel_local.sh 0.3.19

# 2. Upload to GitHub (requires gh CLI)
./scripts/upload_wheel.sh 0.3.19
```

Done! Docker builds will now complete in ~5-10 min instead of ~1.5h.

## Prerequisites

**Hardware Requirements:**
- **Apple Silicon Mac (M1/M2/M3/M4)** OR
- **ARM64 Linux machine** (e.g., AWS Graviton, Oracle Cloud ARM)

**Software Requirements:**
- **podman** or **docker** installed
- **gh CLI** installed and authenticated
  ```bash
  # macOS
  brew install gh
  gh auth login

  # Linux
  # See https://github.com/cli/cli/blob/trunk/docs/install_linux.md
  ```

**Note:** Building on x86_64/AMD64 machines will use QEMU emulation and take ~1.5 hours (same as GitHub Actions). Native ARM64 hardware is strongly recommended.

## When to Use

- **First build** of the project
- **Every time** you update llama-cpp-python version
- When GitHub Actions fails due to compilation timeout

## Build Time Comparison

| Platform | Time | Notes |
|----------|------|-------|
| Apple Silicon (native) | 10-15 min | ✅ Recommended |
| ARM64 Linux (native) | 10-20 min | ✅ Recommended |
| x86_64 + QEMU | ~1.5 hours | ❌ Slow, not recommended |
| GitHub Actions ARM64 | N/A | ❌ Runners not available |

## Details

See [.github/WHEELS_README.md](.github/WHEELS_README.md) for complete documentation.
