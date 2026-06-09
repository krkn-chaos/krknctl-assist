# Benchmark Results

The current `krknctl-assist` benchmark run completed **290 queries** with **0 errors**.

## Summary

- **Precision@1:** 91.03%
- **Recall@K:** 96.55%
- **MRR:** 93.79%
- **No-match rate:** 0.00%
- **Average latency:** 1045 ms
- **p95 latency:** 1137 ms
- **p99 latency:** 1266 ms

## Current State

The benchmark shows strong retrieval quality overall, with most scenario categories returning the expected result at rank 1. The main remaining misses are concentrated in a few overlapping scenario groups, especially pod and network-related scenarios.
