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
    accepted_labels: tuple[str, ...]

    @property
    def expected_labels(self) -> tuple[str, ...]:
        labels = [self.label]
        labels.extend(label for label in self.accepted_labels if label and label != self.label)
        return tuple(labels)


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


def _parse_labels(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    labels: list[str] = []
    for chunk in value.replace("|", ",").replace(";", ",").split(","):
        label = chunk.strip()
        if label:
            labels.append(label)
    return tuple(labels)


def _load_samples(csv_path: Path) -> list[Sample]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "input", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"benchmark CSV missing columns: {sorted(missing)}")

        samples: list[Sample] = []
        for row in reader:
            query = (row.get("input") or "").strip()
            label = (row.get("label") or "").strip()
            row_id = str(row.get("id") or "").strip()
            if not query or not label:
                continue
            samples.append(
                Sample(
                    row_id=row_id,
                    query=query,
                    label=label,
                    accepted_labels=_parse_labels(row.get("accepted_labels")),
                )
            )
        return samples


def _prediction_keys(row: dict[str, Any], match_mode: str) -> list[str]:
    scenario_id = str(row.get("id") or row.get("name") or "").strip()
    runnable = str(row.get("runnable_name") or "").strip()

    if match_mode == "scenario":
        keys = [scenario_id]
    elif match_mode == "runnable":
        keys = [runnable]
    else:
        keys = [scenario_id, runnable]

    deduped: list[str] = []
    for key in keys:
        if key and key not in deduped:
            deduped.append(key)
    return deduped


def _rank_of(expected_labels: tuple[str, ...], predicted_rows: list[dict[str, Any]], match_mode: str) -> Optional[int]:
    expected = {label.strip() for label in expected_labels if label.strip()}
    for idx, row in enumerate(predicted_rows):
        if expected.intersection(_prediction_keys(row, match_mode)):
            return idx + 1
    return None


def _display_id(row: dict[str, Any], match_mode: str) -> str:
    keys = _prediction_keys(row, match_mode)
    return keys[0] if keys else ""


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
                "id": str(row.get("id") or row.get("name") or ""),
                "runnable_name": row.get("runnable_name"),
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
                "id": str(row.get("id") or row.get("name") or ""),
                "score": _score(row.get("rerank_score")),
                "calibrated_score": _score(row.get("rerank_score_calibrated")),
            }
        )
    return rows


def _policy_rows(resp: dict[str, Any], k: int) -> list[dict[str, Any]]:
    policy = resp.get("policy") or {}
    if policy.get("accepted") is False:
        return []
    rows = policy.get("scenarios") or []
    return rows[:k] if isinstance(rows, list) else []


def _raw_rows(resp: dict[str, Any], k: int) -> list[dict[str, Any]]:
    rows = resp.get("results") or []
    return rows[:k] if isinstance(rows, list) else []


def _scored_rows(resp: dict[str, Any], k: int, score_source: str) -> list[dict[str, Any]]:
    return _policy_rows(resp, k) if score_source == "policy" else _raw_rows(resp, k)


def _query_log(
    *,
    sample: Sample,
    resp: dict[str, Any],
    results: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
    rank: Optional[int],
    elapsed_ms: float,
    k: int,
    match_mode: str,
) -> dict[str, Any]:
    top_rows = results[:k]
    return {
        "row_id": sample.row_id,
        "query": sample.query,
        "label": sample.label,
        "accepted_labels": list(sample.accepted_labels),
        "expected_labels": list(sample.expected_labels),
        "policy": resp.get("policy") or {},
        "retriever": _ranked_scores(top_rows, "retrieval_score"),
        "reranker": _reranker_scores(top_rows),
        "final": _ranked_scores(top_rows, "final_score"),
        "scored_predictions": [_display_id(row, match_mode) for row in scored_rows],
        "top_match": _display_id(scored_rows[0], match_mode) if scored_rows else None,
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


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "avg": round(statistics.fmean(values), 3) if values else None,
        "p50": round(_percentile(values, 50) or 0.0, 3) if values else None,
        "p95": round(_percentile(values, 95) or 0.0, 3) if values else None,
        "p99": round(_percentile(values, 99) or 0.0, 3) if values else None,
    }


def _quality_summary(
    *,
    total: int,
    hits_at1: int,
    hits_atk: int,
    reciprocal_ranks: list[float],
) -> dict[str, float]:
    return {
        "precision_at_1": round(hits_at1 / total, 6) if total else 0.0,
        "recall_at_k": round(hits_atk / total, 6) if total else 0.0,
        "mrr": round(sum(reciprocal_ranks) / total, 6) if total else 0.0,
    }


def _category_metrics(categories: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label in sorted(categories):
        row = categories[label]
        count = int(row["count"])
        no_match = int(row["no_match"])
        output[label] = {
            "count": count,
            **_quality_summary(
                total=count,
                hits_at1=int(row["hits_at1"]),
                hits_atk=int(row["hits_atk"]),
                reciprocal_ranks=row["reciprocal_ranks"],
            ),
            "no_match": {"count": no_match, "rate": _rate(no_match, count)},
            "errors": int(row["errors"]),
            "failure_count": int(row["failure_count"]),
            "latency_ms": _latency_summary(row["latency_ms"]),
        }
    return output


def _macro_quality(category_metrics: dict[str, Any]) -> dict[str, float]:
    if not category_metrics:
        return {"precision_at_1": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
    keys = ("precision_at_1", "recall_at_k", "mrr")
    return {
        key: round(statistics.fmean(float(row[key]) for row in category_metrics.values()), 6)
        for key in keys
    }


def _failure_type(policy: dict[str, Any], rank: Optional[int], k: int) -> str:
    if policy.get("accepted") is False:
        return "policy_rejected"
    if rank is None or rank > k:
        return "topk_miss"
    return "top1_miss"


def _failed_metrics(policy: dict[str, Any], rank: Optional[int], k: int) -> list[str]:
    failed: list[str] = []
    if policy.get("accepted") is False:
        failed.append("policy_rejected")
    if rank != 1:
        failed.append("precision@1_miss")
    if rank is None or rank > k:
        failed.append("recall@k_miss")
    return failed


def _empty_category_row() -> dict[str, Any]:
    return {
        "count": 0,
        "hits_at1": 0,
        "hits_atk": 0,
        "reciprocal_ranks": [],
        "latency_ms": [],
        "no_match": 0,
        "errors": 0,
        "failure_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark krkn-assist via debug API (/retrieve)."
    )
    parser.add_argument(
        "--csv",
        default=str(Path("krkn-assist") / "benchmark-docs-gnd.csv"),
        help="Benchmark CSV path (default: krkn-assist/benchmark-docs-gnd.csv)",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:18080",
        help="Debug API base URL (default: http://127.0.0.1:18080)",
    )
    parser.add_argument("--n", type=int, default=100, help="Rows to run (default: 100, 0 means all)")
    parser.add_argument("--fr", type=int, default=25, help="Report every FR rows (default: 25)")
    parser.add_argument("--k", type=int, default=5, help="Recall@K / top-k requested (default: 5)")
    parser.add_argument("--retrieve-k", type=int, default=10, help="Retrieval candidates (default: 10)")
    parser.add_argument("--rerank-k", type=int, default=5, help="Rerank window (default: 5)")
    parser.add_argument("--timeout-s", type=float, default=60.0, help="Per-request timeout (default: 60)")
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed (default: 0 means no shuffle)")
    parser.add_argument("--out", default="benchmark_results.json", help="Output JSON path")
    parser.add_argument("--include-samples", action="store_true", help="Include compact sample rows")
    parser.add_argument("--clear-cache", action="store_true", help="Clear server query cache first")
    parser.add_argument(
        "--score-source",
        choices=("policy", "raw"),
        default="policy",
        help="Score final policy.scenarios or raw reranked results (default: policy)",
    )
    parser.add_argument(
        "--match-mode",
        choices=("either", "scenario", "runnable"),
        default="either",
        help="Match expected label against scenario id, runnable_name, or either (default: either)",
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
    failures: list[dict[str, Any]] = []
    categories: dict[str, dict[str, Any]] = {}
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
        category = categories.setdefault(sample.label, _empty_category_row())
        category["count"] += 1
        payload = {"query": sample.query, "k": k, "retrieve_k": retrieve_k, "rerank_k": rerank_k}

        t0 = time.perf_counter()
        try:
            resp = _http_post_json(endpoint, payload, timeout_s=float(args.timeout_s))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            e2e_lat_ms.append(elapsed_ms)
            category["latency_ms"].append(elapsed_ms)
            category["errors"] += 1
            category["failure_count"] += 1
            error_row = {
                "row_id": sample.row_id,
                "query": sample.query,
                "label": sample.label,
                "error": str(exc),
                "elapsed_ms": round(elapsed_ms, 2),
            }
            errors.append(error_row)
            failures.append({**error_row, "failure_type": "error", "failed_metrics": ["error"]})
            reciprocal_ranks.append(0.0)
            category["reciprocal_ranks"].append(0.0)
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
        category["latency_ms"].append(elapsed_ms)

        results = _raw_rows(resp, k)
        scored_rows = _scored_rows(resp, k, args.score_source)
        policy = resp.get("policy") or {}
        policy_reason = str(policy.get("reason") or "unknown")
        policy_reason_counts[policy_reason] = policy_reason_counts.get(policy_reason, 0) + 1
        if policy.get("accepted") is False:
            no_match_count += 1
            category["no_match"] += 1

        rank = _rank_of(sample.expected_labels, scored_rows, args.match_mode)
        if rank == 1:
            hits_at1 += 1
            category["hits_at1"] += 1
        if rank is not None and rank <= k:
            hits_atk += 1
            category["hits_atk"] += 1
            rr = 1.0 / float(rank)
        else:
            rr = 0.0
        reciprocal_ranks.append(rr)
        category["reciprocal_ranks"].append(rr)

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
                scored_rows=scored_rows,
                rank=rank,
                elapsed_ms=elapsed_ms,
                k=k,
                match_mode=args.match_mode,
            )
        )

        if rank != 1:
            category["failure_count"] += 1
            failures.append(
                {
                    "row_id": sample.row_id,
                    "label": sample.label,
                    "query": sample.query,
                    "failure_type": _failure_type(policy, rank, k),
                    "failed_metrics": _failed_metrics(policy, rank, k),
                    "rank": rank,
                    "expected_label": sample.label,
                    "expected_labels": list(sample.expected_labels),
                    "predicted_top1": _display_id(scored_rows[0], args.match_mode) if scored_rows else None,
                    "predicted_topk": [_display_id(row, args.match_mode) for row in scored_rows],
                    "raw_topk": [_display_id(row, "scenario") for row in results],
                    "policy": policy,
                    "top_match": resp.get("top_match"),
                    "e2e_latency_ms": round(elapsed_ms, 2),
                    "error": None,
                }
            )

        if args.include_samples:
            sample_rows.append(
                {
                    "row_id": sample.row_id,
                    "query": sample.query,
                    "label": sample.label,
                    "expected_labels": list(sample.expected_labels),
                    "predicted": [_display_id(row, args.match_mode) for row in scored_rows],
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
    category_metrics = _category_metrics(categories)
    summary = {
        "quality": _quality_summary(
            total=n_done,
            hits_at1=hits_at1,
            hits_atk=hits_atk,
            reciprocal_ranks=reciprocal_ranks,
        ),
        "macro_quality_by_label": _macro_quality(category_metrics),
        "no_match": {"count": no_match_count, "rate": _rate(no_match_count, n_done)},
        "latency_ms": _latency_summary(e2e_lat_ms),
        "stage_timing_ms": {
            "retrieval": _latency_summary(stage_retrieve_ms),
            "rerank": _latency_summary(stage_rerank_ms),
            "total": _latency_summary(stage_total_ms),
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
            "score_source": args.score_source,
            "match_mode": args.match_mode,
        },
        "run": {
            "started_at_unix": int(started_wall),
            "duration_s": round(duration_s, 3),
            "completed": n_done,
            "errors": len(errors),
        },
        "final_recommended_benchmark_metrics": summary,
        "category_metrics": category_metrics,
        "failure_summary": {
            "total": len(failures),
            "by_label": {
                label: int(row["failure_count"])
                for label, row in sorted(categories.items())
            },
        },
        "policy_reason_counts": dict(sorted(policy_reason_counts.items())),
        "query_logs": query_logs,
        "failures": failures,
        "errors_detail": errors,
    }
    if args.include_samples:
        report["samples"] = sample_rows

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")

    print("\nFinal Recommended Benchmark Metrics")
    print(json.dumps(summary, indent=2))
    print(f"\nFailures: {len(failures)}")
    print(f"Wrote JSON: {out_path}")
    if errors:
        print(f"Errors: {len(errors)} (see errors_detail in {out_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
