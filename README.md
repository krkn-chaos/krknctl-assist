
# krknctl-assist

A natural language based chaos scenario discovery assistant

## What is krknctl-assist?

`krknctl-assist` makes scenario discovery faster by translating plain-English testing intent into relevant scenario recommendations. It works alongside [krknctl](https://krkn-chaos.dev/docs/krknctl/) by providing:

- **Natural-language queries** - Ask in plain English, get relevant scenario matches
- **Local execution** - Runs locally via container runtime (no hosted dependency)
- **Seamless integration** - Works neatly in your `krknctl` workflow

## Quick Start

### Prerequisites
`podman`, `python3`, `go`

### Install

```bash
git clone https://github.com/krkn-chaos/krknctl-assist.git ~/krknctl-assist
cd ~/krknctl-assist
bash ./scripts/install_krknctl_assist.sh
```

### Run

```bash
krknctl-assist
```

The first run may take longer as it builds the container image, prepares indexes, and sets up a local `krknctl` checkout.

## Usage Example

```bash
> "Block a pod's outgoing MySQL and PostgreSQL traffic to disrupt aurora database connections"
⏱️  Response time: 0.37s

⚡ AI assist Response

Scenario: aurora-disruption

🧭 matched scenario context: aurora-disruption

?Aurora Disruption Scenario
"This scenario blocks a pod's outgoing MySQL and PostgreSQL traffic, effectively preventing it from connecting to any AWS Aurora SQL engine."

Suggested command: krknctl run pod-network-filter
```

## Useful Commands

| Command | Purpose |
|---------|---------|
| `krknctl-assist` | Start the interactive assistant |
| `krknctl-assist --force-build` | Force a rebuild |
| `krknctl-assist --cleanup` | Clean up local containers |

## For Maintainers

### Building Docker Images

Docker images are automatically built and pushed to Quay.io on every tag push via GitHub Actions.

#### Pre-compiled Wheels for ARM64

To speed up ARM64 builds (which otherwise take ~1.5 hours due to QEMU emulation), we use pre-compiled llama-cpp-python wheels:

**Requirements:**
- Apple Silicon Mac (M1/M2/M3) OR ARM64 Linux machine
- `podman` or `docker` installed
- `gh` CLI installed and authenticated

**Process:**
```bash
# 1. Build wheel locally (10-15 min on native ARM64)
./scripts/build_wheel_local.sh 0.3.19

# 2. Upload to GitHub Releases
./scripts/upload_wheel.sh 0.3.19
```

See [WHEELS_QUICKSTART.md](WHEELS_QUICKSTART.md) for details.

#### Updating llama-cpp-python Version

1. Update `ARG LLAMA_CPP_VERSION=` in both Dockerfiles
2. Build and upload new ARM64 wheel (see above)
3. Push a new tag to trigger CI builds

The Dockerfiles will automatically use the pre-compiled wheel if available, otherwise fall back to building from source.
