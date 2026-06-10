# Pre-compiled Wheels Setup

## Overview

To avoid compiling `llama-cpp-python` with Vulkan backend on every build (~1.5 hours on ARM64 via QEMU emulation in GitHub Actions), we use pre-compiled wheels built locally on native ARM64 hardware and uploaded manually.

## How It Works

1. **Local build** - Compile wheel locally on native ARM64 hardware (much faster than emulation)
2. **Manual upload** - Upload wheel to GitHub Releases using `gh` CLI
3. **Automatic download** - Dockerfiles download pre-compiled wheels if available
4. **Fallback** - If wheel doesn't exist, compile from source

## Requirements for Building Wheels

**You need native ARM64 hardware:**
- ✅ Apple Silicon Mac (M1/M2/M3/M4)
- ✅ ARM64 Linux machine (AWS Graviton, Oracle Cloud ARM, Raspberry Pi 4/5, etc.)
- ❌ x86_64/AMD64 machines will use QEMU emulation and be very slow

**Software:**
- `podman` or `docker`
- `gh` CLI (GitHub CLI)

## Steps to Create Wheel (ARM64 only)

### 1. Build Wheel Locally

On an Apple Silicon Mac or ARM64 Linux machine:

```bash
./scripts/build_wheel_local.sh 0.3.19
```

Build times:
- **Apple Silicon (native)**: ~10-15 minutes
- **ARM64 Linux (native)**: ~10-20 minutes
- **x86_64 + QEMU emulation**: ~1.5 hours (not recommended)

### 2. Upload to GitHub Release

```bash
./scripts/upload_wheel.sh 0.3.19
```

Requires `gh` CLI installed and authenticated:
```bash
# macOS
brew install gh
gh auth login

# Linux - see https://github.com/cli/cli/blob/trunk/docs/install_linux.md
```

### 3. Subsequent Docker Builds Will Be Fast

Once the wheel is available on GitHub Releases:
- **With wheel**: ~5-10 minutes total (download only)
- **Without wheel**: ~1.5 hours (build + compile from source)

## Release Structure

```
Release: llama-cpp-0.3.19
└── llama_cpp_python-0.3.19-cp3XX-cp3XX-linux_aarch64.whl  # ARM64
```

**Note**: Only ARM64 wheel is needed because:
- ARM64 builds on GitHub Actions use QEMU emulation and are extremely slow (~1.5h)
- AMD64 builds are native and fast (~5-10 min), so pre-compilation is unnecessary

The wheel filename includes the Python version (e.g., cp313 = Python 3.13). Dockerfiles automatically try multiple Python versions (cp313, cp312, cp311, cp310) to find a compatible wheel.

## Updating llama-cpp-python Version

When updating the version (e.g., from 0.3.19 to 0.3.20):

1. Update `ARG LLAMA_CPP_VERSION=0.3.20` in both Dockerfiles:
   - `krkn-assist/Dockerfile`
   - `krkn-assist/Dockerfile.linux`

2. Build new wheel locally (requires ARM64 hardware):
   ```bash
   ./scripts/build_wheel_local.sh 0.3.20
   ```

3. Upload to GitHub:
   ```bash
   ./scripts/upload_wheel.sh 0.3.20
   ```

4. Subsequent builds will use the new wheel

5. Push a git tag to trigger Docker builds:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

## Troubleshooting

### Wheels Not Downloaded

Check that:
- Release exists: `https://github.com/krkn-chaos/krknctl-assist/releases/tag/llama-cpp-0.3.19`
- Wheel filename matches: `llama_cpp_python-0.3.19-cp3XX-cp3XX-linux_aarch64.whl`
- Dockerfiles search for multiple Python versions (cp313, cp312, cp311, cp310)

### Force Rebuild from Source

Temporarily delete the GitHub release or rename the tag

### Building on Different Architectures

The `build_wheel_local.sh` script auto-detects architecture:
- **Mac ARM64**: builds `linux/arm64` wheel (Apple Silicon)
- **Linux ARM64**: builds `linux/arm64` wheel (native)
- **Linux x86_64**: builds `linux/amd64` wheel (native, fast)
- **Mac x86_64**: uses QEMU to build `linux/amd64` wheel (slow)

### I Don't Have ARM64 Hardware

Options:
1. **Use QEMU emulation** (slow but works):
   ```bash
   ./scripts/build_wheel_local.sh 0.3.19
   # Takes ~1.5 hours on x86_64 hardware
   ```
2. **Ask someone with ARM64 hardware** to build it for you
3. **Use cloud ARM64 instance**:
   - AWS Graviton (t4g instances)
   - Oracle Cloud ARM (always-free tier available)
   - GitHub Codespaces (if ARM64 runners become available)
4. **Skip pre-compilation**: Let GitHub Actions compile from source on first build (only affects first build)
