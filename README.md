# krknctl Assist

> **Note:** `krknctl-assist` is still under active development. Commands, branches, and setup behavior may change. Use this flow with caution, especially outside local testing or integration work.

Guide to build the assist service image used by `krknctl assist`:

## Quick Start

Clone the assist service from the organization repository:

```bash
git clone https://github.com/krkn-chaos/krknctl-assist.git ~/krknctl-assist
cd ~/krknctl-assist
```

For the current assist integration work, use Aryan's `krknctl` fork as the
`krknctl` source of truth:

```bash
export KRKNCTL_REPO=https://github.com/AryanPrakhar/krknctl.git
export KRKNCTL_BRANCH=fix/assist-source-doc-details
```

Clone the forked `krknctl` checkout used by the wrapper:

```bash
git clone --branch "$KRKNCTL_BRANCH" "$KRKNCTL_REPO" ~/krknctl-fork
```

Run the full verification flow:

```bash
cd ~/krknctl-assist
./scripts/setup_krknctl_assist.sh --cleanup

./scripts/setup_krknctl_assist.sh \
  --verify \
  --krknctl-dir ~/krknctl-fork \
  --krknctl-branch "$KRKNCTL_BRANCH"
```

## Interactive Launch

Use the wrapper for manual testing:

```bash
./scripts/setup_krknctl_assist.sh \
  --launch \
  --krknctl-dir ~/krknctl-fork \
  --krknctl-branch "$KRKNCTL_BRANCH"
```

The wrapper is preferred over running `krknctl assist run` directly because it cleans stale containers first.


## Rebuild Assist Only

Use this when you only want to rebuild and smoke-test the assist image, without letting the wrapper clone, build, or run `krknctl`:

```bash
cd ~/krknctl-assist
./scripts/setup_krknctl_assist.sh --cleanup
./scripts/setup_krknctl_assist.sh --setup-only --force-build --skip-krknctl
```

## Common Commands

```bash
# Clean stale krknctl assist containers and processes.
cd ~/krknctl-assist
./scripts/setup_krknctl_assist.sh --cleanup

# Rebuild and verify everything from scratch (using Aryan's krknctl fork).
KRKNCTL_REPO=https://github.com/AryanPrakhar/krknctl.git \
  ./scripts/setup_krknctl_assist.sh \
  --verify \
  --force-build \
  --krknctl-dir ~/krknctl-fork \
  --krknctl-branch fix/assist-source-doc-details

# Start the debug and compat APIs for retrieval work.
./scripts/pipeline.sh --verbose

# Run a benchmark after the debug API is healthy at http://127.0.0.1:18080.
./scripts/benchmark.py --n 100 --fr 25 --out bench.json --clear-cache
```

## API Endpoints

When `./scripts/pipeline.sh --verbose` is running:

- Debug API: `http://127.0.0.1:18080`
- Compat API: `http://127.0.0.1:8080`

```bash
curl -s http://127.0.0.1:18080/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"network latency between services","k":5,"rerank_k":3}' | jq

curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"krkn-assist","messages":[{"role":"user","content":"network latency between services"}]}' | jq
```

## Troubleshooting

If port `8080` is already in use or `krknctl` reports `proxy already running`, clean up stale assist containers first:

```bash
cd ~/krknctl-assist
./scripts/setup_krknctl_assist.sh --cleanup
```
