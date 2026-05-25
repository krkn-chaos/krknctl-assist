import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark.py"


def _assert_metric_floor(metric_name, actual_value, minimum_value):
    if actual_value >= minimum_value:
        return

    drift = minimum_value - actual_value
    pytest.fail(
        f"Benchmark regression detected for {metric_name}: "
        f"actual={actual_value:.6f}, floor={minimum_value:.6f}, drift={drift:.6f}"
    )


def test_debug_api_benchmark_stays_above_regression_floor(tmp_path):
    if os.environ.get("KRKN_RUN_BENCHMARK") != "1":
        pytest.skip("Set KRKN_RUN_BENCHMARK=1 to run the 1000-sample debug API regression benchmark.")

    base_url = os.environ.get("KRKN_BENCHMARK_BASE_URL", "http://127.0.0.1:18080")
    sample_count = int(os.environ.get("KRKN_BENCHMARK_N", "1000"))
    precision_floor = float(os.environ.get("KRKN_BENCHMARK_MIN_P1", "0.73"))
    recall_floor = float(os.environ.get("KRKN_BENCHMARK_MIN_RK", "0.94"))
    mrr_floor = float(os.environ.get("KRKN_BENCHMARK_MIN_MRR", "0.82"))

    out_path = tmp_path / "benchmark_results.json"
    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--base-url",
        base_url,
        "--n",
        str(sample_count),
        "--fr",
        "100",
        "--out",
        str(out_path),
        "--clear-cache",
    ]

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    if completed.returncode != 0:
        pytest.fail(
            "Benchmark run failed.\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    quality = report["final_recommended_benchmark_metrics"]["quality"]

    _assert_metric_floor("precision_at_1", quality["precision_at_1"], precision_floor)
    _assert_metric_floor("recall_at_k", quality["recall_at_k"], recall_floor)
    _assert_metric_floor("mrr", quality["mrr"], mrr_floor)
