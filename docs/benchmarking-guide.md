
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


### Drift Control Limits

Treat these values as the current control limits for this benchmark run. If any metric falls below the minimum shown here, or latency rises above the stated ceiling, investigate before merging the change. There are conservative limits.

| Metric | Control limit |
| --- | --- |
| `precision@1` | Keep at or above `0.74` |
| `recall@k` | Keep at or above `0.92` |
| `mrr` | Keep at or above `0.80` |
| `latency_ms.p95` | Keep at or below `2000 ms` |
| `latency_ms.avg` | Keep at or below `1500 ms` |

### How To Read Drift

Small changes are normal, especially when the model backend or CPU features differ between machines. The important thing is direction:

* A drop in `precision@1` and `mrr` usually means the assistant is surfacing the wrong scenario first OR the reranker is not performing well.
* A drop in `recall@k` usually means the correct scenario is not being surfaced in the candidate set i.e. bad hybrid retrieval
* A rise in `latency_ms.p95` usually means a subset of queries is getting much slower, often because reranking or model loading changed, or user-input drift.
* A rise in `stage_timing_ms.rerank.avg` usually points to cross-encoder backend, quantization, CPU capability changes, ONNX Runtime unavailability
* A rise in `stage_timing_ms.retrieval.avg` usually measures steady-state retrieval latency after indexes are already built and loaded. Index build time should be tracked separately, if needed, as an offline pipeline metric.
