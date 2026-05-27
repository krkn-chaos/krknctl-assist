#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Sample:
    row_id: str
    query: str
    label: str


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    if p <= 0:
        return float(min(values))
    if p >= 100:
        return float(max(values))
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[int(k)])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)


def _http_post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url=url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _http_get_json(url: str, timeout_s: float) -> dict[str, Any]:
    req = Request(url=url, headers={"accept": "application/json"}, method="GET")
    with urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _load_samples(csv_path: Path) -> list[Sample]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "input", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"data.csv missing columns: {sorted(missing)}")

        samples: list[Sample] = []
        for row in reader:
            query = (row.get("input") or "").strip()
            label = (row.get("label") or "").strip()
            row_id = str(row.get("id") or "").strip()
            if not query or not label:
                continue
            samples.append(Sample(row_id=row_id, query=query, label=label))
        return samples


def _rank_of(label: str, predicted_ids: list[str]) -> Optional[int]:
    label = (label or "").strip()
    for idx, pred in enumerate(predicted_ids):
        if pred == label:
            return idx + 1
    return None


def _rate(part: int, total: int) -> float:
    return round(part / total, 6) if total else 0.0


def _score(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _ranked_scores(results: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(results, start=1):
        rows.append(
            {
                "rank": idx,
                "id": str(row.get("id") or ""),
                "score": _score(row.get(score_key)),
            }
        )
    return rows


def _reranker_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(results, start=1):
        rows.append(
            {
                "rank": idx,
                "id": str(row.get("id") or ""),
                "score": _score(row.get("rerank_score")),
                "calibrated_score": _score(row.get("rerank_score_calibrated")),
            }
        )
    return rows


def _query_log(
    *,
    sample: Sample,
    resp: dict[str, Any],
    results: list[dict[str, Any]],
    rank: Optional[int],
    elapsed_ms: float,
    k: int,
) -> dict[str, Any]:
    top_rows = results[:k]
    return {
        "row_id": sample.row_id,
        "query": sample.query,
        "label": sample.label,
        "policy": resp.get("policy") or {},
        "retriever": _ranked_scores(top_rows, "retrieval_score"),
        "reranker": _reranker_scores(top_rows),
        "final": _ranked_scores(top_rows, "final_score"),
        "top_match": resp.get("top_match"),
        "rank": rank,
        "hit_at_1": rank == 1,
        "hit_at_k": (rank is not None and rank <= k),
        "e2e_latency_ms": round(elapsed_ms, 2),
        "message": resp.get("message"),
    }


def _progress_summary(
    *,
    done: int,
    total: int,
    hits_at1: int,
    hits_atk: int,
    reciprocal_ranks: list[float],
    e2e_ms: list[float],
    no_match_count: int,
) -> str:
    mrr = sum(reciprocal_ranks) / done if done else 0.0
    p_at1 = hits_at1 / done if done else 0.0
    r_atk = hits_atk / done if done else 0.0
    no_match_rate = no_match_count / done if done else 0.0
    avg_ms = statistics.fmean(e2e_ms) if e2e_ms else 0.0
    p95 = _percentile(e2e_ms, 95) or 0.0
    return (
        f"[{done}/{total}] "
        f"precision@1={p_at1:.3f} recall@k={r_atk:.3f} mrr={mrr:.3f} "
        f"no_match={no_match_count} ({no_match_rate:.1%}) "
        f"avg_ms={avg_ms:.1f} p95_ms={p95:.1f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark krkn-assist on krkn-assist/data.csv via debug API (/retrieve)."
    )
    parser.add_argument(
        "--csv",
        default=str(Path("krkn-assist") / "data.csv"),
        help="Path to data.csv (default: krkn-assist/data.csv)",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:18080",
        help="Debug API base URL (default: http://127.0.0.1:18080)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of tests to run (default: 100, 0 means all)",
    )
    parser.add_argument(
        "--fr",
        type=int,
        default=25,
        help="Reporting frequency (print every FR samples, default: 25)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Recall@K window size / top-k requested from server (default: 5)",
    )
    parser.add_argument(
        "--retrieve-k",
        type=int,
        default=10,
        help="FAISS retrieval candidates to fetch before reranking (default: 10)",
    )
    parser.add_argument(
        "--rerank-k",
        type=int,
        default=5,
        help="Cross-encoder reranking window (default: 5)",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed (default: 0; no shuffle when 0)",
    )
    parser.add_argument(
        "--out",
        default="benchmark_results.json",
        help="Output JSON path (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="Include per-sample predictions in the JSON output (can be large)",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear server query cache via debug endpoint before benchmarking",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 2

    samples = _load_samples(csv_path)
    if not samples:
        print("No usable rows found in CSV", file=sys.stderr)
        return 2

    if args.seed:
        rng = random.Random(args.seed)
        rng.shuffle(samples)

    total = len(samples) if args.n == 0 else min(len(samples), args.n)
    samples = samples[:total]

    endpoint = args.base_url.rstrip("/") + "/retrieve"
    health_endpoint = args.base_url.rstrip("/") + "/health"
    clear_cache_endpoint = args.base_url.rstrip("/") + "/debug/cache/clear"
    k = max(1, int(args.k))
    retrieve_k = max(int(args.retrieve_k), k)
    rerank_k = max(int(args.rerank_k), k)

    hits_at1 = 0
    hits_atk = 0
    reciprocal_ranks: list[float] = []

    e2e_lat_ms: list[float] = []
    stage_retrieve_ms: list[float] = []
    stage_rerank_ms: list[float] = []
    stage_total_ms: list[float] = []

    errors: list[dict[str, Any]] = []
    query_logs: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    no_match_count = 0
    policy_reason_counts: dict[str, int] = {}

    started_wall = time.time()
    try:
        _http_get_json(health_endpoint, timeout_s=min(float(args.timeout_s), 10.0))
    except Exception as exc:
        print(
            "Benchmark API is not reachable. Start the debug API first with:\n"
            "  ./scripts/pipeline.sh --verbose\n\n"
            f"Health check failed for {health_endpoint}: {exc}",
            file=sys.stderr,
        )
        return 2

    if args.clear_cache:
        try:
            _http_post_json(clear_cache_endpoint, {}, timeout_s=float(args.timeout_s))
        except Exception as exc:
            print(f"Warning: failed to clear cache: {exc}", file=sys.stderr)
    for idx, sample in enumerate(samples, start=1):
        payload = {"query": sample.query, "k": k, "retrieve_k": retrieve_k, "rerank_k": rerank_k}

        t0 = time.perf_counter()
        try:
            resp = _http_post_json(endpoint, payload, timeout_s=float(args.timeout_s))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            e2e_lat_ms.append(elapsed_ms)
            errors.append(
                {
                    "row_id": sample.row_id,
                    "query": sample.query,
                    "label": sample.label,
                    "error": str(exc),
                    "elapsed_ms": round(elapsed_ms, 2),
                }
            )
            if args.fr > 0 and (idx % args.fr == 0 or idx == total):
                print(
                    _progress_summary(
                        done=idx,
                        total=total,
                        hits_at1=hits_at1,
                        hits_atk=hits_atk,
                        reciprocal_ranks=reciprocal_ranks,
                        e2e_ms=e2e_lat_ms,
                        no_match_count=no_match_count,
                    )
                )
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        e2e_lat_ms.append(elapsed_ms)

        results = resp.get("results") or []
        predicted_ids = [str(row.get("id") or "") for row in results if str(row.get("id") or "").strip()]
        policy = resp.get("policy") or {}
        policy_reason = str(policy.get("reason") or "unknown")
        policy_reason_counts[policy_reason] = policy_reason_counts.get(policy_reason, 0) + 1
        if policy.get("accepted") is False:
            no_match_count += 1

        rank = _rank_of(sample.label, predicted_ids)
        if rank == 1:
            hits_at1 += 1
        if rank is not None and rank <= k:
            hits_atk += 1
            reciprocal_ranks.append(1.0 / float(rank))
        else:
            reciprocal_ranks.append(0.0)

        # Stage timing (from server), if present on top row.
        top_timing = (results[0].get("timing_ms") if results else None) or {}
        if isinstance(top_timing, dict):
            if "retrieve" in top_timing:
                stage_retrieve_ms.append(float(top_timing["retrieve"]))
            if "rerank" in top_timing:
                stage_rerank_ms.append(float(top_timing["rerank"]))
            if "total" in top_timing:
                stage_total_ms.append(float(top_timing["total"]))

        query_logs.append(
            _query_log(
                sample=sample,
                resp=resp,
                results=results,
                rank=rank,
                elapsed_ms=elapsed_ms,
                k=k,
            )
        )

        if args.include_samples:
            sample_rows.append(
                {
                    "row_id": sample.row_id,
                    "query": sample.query,
                    "label": sample.label,
                    "predicted": [
                        {
                            "id": str(row.get("id") or ""),
                            "final_score": row.get("final_score"),
                        }
                        for row in results
                    ],
                    "rank": rank,
                    "hit_at_1": rank == 1,
                    "hit_at_k": (rank is not None and rank <= k),
                    "e2e_latency_ms": round(elapsed_ms, 2),
                    "timing_ms": top_timing if isinstance(top_timing, dict) else None,
                }
            )

        if args.fr > 0 and (idx % args.fr == 0 or idx == total):
            print(
                _progress_summary(
                    done=idx,
                    total=total,
                    hits_at1=hits_at1,
                    hits_atk=hits_atk,
                    reciprocal_ranks=reciprocal_ranks,
                    e2e_ms=e2e_lat_ms,
                    no_match_count=no_match_count,
                )
            )

    duration_s = time.time() - started_wall
    n_done = len(samples)
    precision_at_1 = hits_at1 / n_done if n_done else 0.0
    recall_at_k = hits_atk / n_done if n_done else 0.0
    mrr = sum(reciprocal_ranks) / n_done if n_done else 0.0

    summary = {
        "quality": {
            "precision_at_1": round(precision_at_1, 6),
            "recall_at_k": round(recall_at_k, 6),
            "mrr": round(mrr, 6),
        },
        "no_match": {
            "count": no_match_count,
            "rate": _rate(no_match_count, n_done),
        },
        "latency_ms": {
            "avg": round(statistics.fmean(e2e_lat_ms), 3) if e2e_lat_ms else None,
            "p50": round(_percentile(e2e_lat_ms, 50) or 0.0, 3) if e2e_lat_ms else None,
            "p95": round(_percentile(e2e_lat_ms, 95) or 0.0, 3) if e2e_lat_ms else None,
            "p99": round(_percentile(e2e_lat_ms, 99) or 0.0, 3) if e2e_lat_ms else None,
        },
        "stage_timing_ms": {
            "retrieval": {
                "avg": round(statistics.fmean(stage_retrieve_ms), 3) if stage_retrieve_ms else None,
                "p50": round(_percentile(stage_retrieve_ms, 50) or 0.0, 3) if stage_retrieve_ms else None,
                "p95": round(_percentile(stage_retrieve_ms, 95) or 0.0, 3) if stage_retrieve_ms else None,
                "p99": round(_percentile(stage_retrieve_ms, 99) or 0.0, 3) if stage_retrieve_ms else None,
            },
            "rerank": {
                "avg": round(statistics.fmean(stage_rerank_ms), 3) if stage_rerank_ms else None,
                "p50": round(_percentile(stage_rerank_ms, 50) or 0.0, 3) if stage_rerank_ms else None,
                "p95": round(_percentile(stage_rerank_ms, 95) or 0.0, 3) if stage_rerank_ms else None,
                "p99": round(_percentile(stage_rerank_ms, 99) or 0.0, 3) if stage_rerank_ms else None,
            },
            "total": {
                "avg": round(statistics.fmean(stage_total_ms), 3) if stage_total_ms else None,
                "p50": round(_percentile(stage_total_ms, 50) or 0.0, 3) if stage_total_ms else None,
                "p95": round(_percentile(stage_total_ms, 95) or 0.0, 3) if stage_total_ms else None,
                "p99": round(_percentile(stage_total_ms, 99) or 0.0, 3) if stage_total_ms else None,
            },
        },
    }

    report = {
        "config": {
            "csv": str(csv_path),
            "endpoint": endpoint,
            "n": args.n,
            "k": k,
            "retrieve_k": retrieve_k,
            "rerank_k": rerank_k,
            "timeout_s": float(args.timeout_s),
            "seed": int(args.seed),
            "report_every": int(args.fr),
        },
        "run": {
            "started_at_unix": int(started_wall),
            "duration_s": round(duration_s, 3),
            "completed": n_done,
            "errors": len(errors),
        },
        "final_recommended_benchmark_metrics": summary,
        "policy_reason_counts": dict(sorted(policy_reason_counts.items())),
        "query_logs": query_logs,
        "errors_detail": errors,
    }
    if args.include_samples:
        report["samples"] = sample_rows

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")

    print("\nFinal Recommended Benchmark Metrics")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote JSON: {out_path}")
    if errors:
        print(f"Errors: {len(errors)} (see errors_detail in {out_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
