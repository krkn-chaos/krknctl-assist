
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

## Running Benchmarks

The benchmark uses the debug API, not the OpenAI-compatible API. Keep
`./scripts/pipeline.sh --verbose` running in one terminal so the debug endpoint
stays available on `http://127.0.0.1:18080/retrieve`, then run the benchmark in a
second terminal.

```bash
# Terminal 1: start the debug API and leave it open
./scripts/pipeline.sh --verbose

# Terminal 2: run a 100-query benchmark
mkdir -p bench-res
RUN_DIR="bench-res/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
python3 scripts/benchmark.py --n 100 --fr 10 --out "$RUN_DIR/benchmark.json"
```

For a full run:

```bash
python3 scripts/benchmark.py --n 1000 --fr 100 --out "$RUN_DIR/final.json"
```