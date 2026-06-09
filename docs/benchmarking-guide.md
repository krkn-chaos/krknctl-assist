
## Benchmark Metrics Guide


| Metric | What it measures | Consequence if it drifts down |
| --- | --- | --- |
| `precision@1` | How often the top suggestion is the correct scenario. This is the MOST important metric, because this ensures that the user sees the most relevant result. Constraining the latency, all measures should be taken to bring it closer to 1.0 | Users see the wrong command first, which is the fastest way to lose trust in the assistant. |
| `recall@k` | How often the correct scenario appears in the top `k` results. | Worse candidate set for the reranker, affecting precision |
| `mrr` | How early the correct scenario appears on average. | Even when the right scenario is present, it may be buried too deep to be useful. |
| `latency_ms.avg` | Average end-to-end response time. | The assistant feels sluggish |
| `latency_ms.p95` | Tail response time for the slowest 5% of queries. | A minority of queries become visibly painful, which is usually where users notice regressions first. |
| `stage_timing_ms.retrieval.avg` | Time spent in vector search and candidate gathering. | Retrieval becomes the bottleneck and every downstream stage inherits the delay. |
| `stage_timing_ms.rerank.avg` | Time spent in the cross-encoder rerank step. | The system still finds candidates, but ranking quality becomes expensive enough to threaten responsiveness. |

### Notes
- Reranking is important because it boosts precision (our core metric), but it is very expensive in terms of latency if not set properly
- The quality of reranking will be of no use if the retriever does not provide high recall

### Current Baseline

These are the latest benchmark values from `benchmark-results/20260609T143153Z/benchmark.json`:

| Metric | Latest value |
| --- | --- |
| `precision@1` | `0.910345` |
| `recall@k` | `0.965517` |
| `mrr` | `0.937931` |
| `latency_ms.avg` | `1045.521 ms` |
| `latency_ms.p50` | `1021.874 ms` |
| `latency_ms.p95` | `1136.891 ms` |
| `latency_ms.p99` | `1265.980 ms` |
| `stage_timing_ms.retrieval.avg` | `155.507 ms` |
| `stage_timing_ms.rerank.avg` | `867.752 ms` |
| `stage_timing_ms.total.avg` | `1029.421 ms` |
| `failures` | `26 / 290` |


### Drift Control Limits

Treat these values as the current control limits for this benchmark run. If any metric falls below the minimum shown here, or latency rises above the stated ceiling, investigate before merging the change. There are conservative limits.

| Metric | Control limit |
| --- | --- |
| `precision@1` | Keep at or above `0.82` |
| `recall@k` | Keep at or above `0.90` |
| `mrr` | Keep at or above `0.90` |
| `latency_ms.p95` | Keep at or below `1200 ms` |
| `latency_ms.avg` | Keep at or below `1400 ms` |

### How To Read Drift

Small changes are normal, especially when the model backend or CPU features differ between machines. The important thing is direction:

* A drop in `precision@1` and `mrr` usually means the assistant is surfacing the wrong scenario first OR the reranker is not performing well.
* A drop in `recall@k` usually means the correct scenario is not being surfaced in the candidate set i.e. bad hybrid retrieval
* A rise in `latency_ms.p95` usually means a subset of queries is getting much slower, often because reranking or model loading changed, or user-input drift.
* A rise in `stage_timing_ms.rerank.avg` usually points to cross-encoder backend, quantization, CPU capability changes, ONNX Runtime unavailability
* A rise in `stage_timing_ms.retrieval.avg` usually measures steady-state retrieval latency after indexes are already built and loaded. Index build time should be tracked separately, if needed, as an offline pipeline metric.



## Running Benchmarks

The benchmark uses the debug API, not the OpenAI-compatible API. Keep
`./scripts/pipeline.sh --verbose` running in one terminal so the debug endpoint stays available on `http://127.0.0.1:18080/retrieve`, then run the benchmark in a second terminal.

```bash
# Terminal 1: start the debug API and leave it open
./scripts/pipeline.sh --verbose

# Terminal 2: run the benchmark
mkdir -p benchmark-results
RUN_DIR="benchmark-results/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN_DIR"
python3 scripts/benchmark.py \
  --csv krkn-assist/mini-bench.csv \
  --base-url http://127.0.0.1:18080 \
  --n 0 \
  --fr 100 \
  --out "$RUN_DIR/benchmark.json"
```
