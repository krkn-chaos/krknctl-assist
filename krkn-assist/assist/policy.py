from __future__ import annotations

from dataclasses import dataclass

from .settings import (
    CE_TOP2_GAP_THRESHOLD,
    FAISS_TOP2_GAP_THRESHOLD,
    MAX_MULTI_SCENARIOS,
    MIN_FAISS_SCORE,
    MIN_MATCH_SCORE,
    MIN_MULTI_SCORE,
    MIN_QUERY_WORDS,
    MULTI_MATCH_SCORE_GAP,
)


@dataclass(frozen=True)
class PolicyDecision:
    accepted: bool
    reason: str
    scenarios: list[dict]
    timing_ms: int | None = None


def _score_key(row: dict) -> float:
    return float(row.get("final_score", row.get("score", 0.0)))


def _dedupe_evidence(evidence: list[dict]) -> list[dict]:
    best_by_id: dict[str, dict] = {}
    for row in evidence:
        scenario_id = str(row.get("id") or "").strip()
        if not scenario_id:
            continue
        existing = best_by_id.get(scenario_id)
        if existing is None or _score_key(row) > _score_key(existing):
            best_by_id[scenario_id] = row
    return sorted(best_by_id.values(), key=_score_key, reverse=True)


def decide_scenarios(
    *,
    query: str,
    evidence: list[dict],
    threshold: float | None = None,
    allow_multi: bool = True,
) -> PolicyDecision:
    """
    Convert raw ranked evidence into a final decision.

    evidence: list of dicts containing:
      - id
      - final_score
      - score (cross-encoder raw)
      - retrieval_score
      - timing_ms (optional)
    """
    threshold = float(MIN_MATCH_SCORE if threshold is None else threshold)
    query = (query or "").strip()
    evidence = _dedupe_evidence(evidence)

    if len(query.split()) < MIN_QUERY_WORDS:
        return PolicyDecision(
            accepted=False,
            reason="query_too_short",
            scenarios=[],
            timing_ms=None,
        )

    if not evidence:
        return PolicyDecision(
            accepted=False,
            reason="no_candidates",
            scenarios=[],
            timing_ms=None,
        )

    top = evidence[0]
    top_timing = top.get("timing_ms") or {}
    timing_ms = int(top_timing.get("total")) if "total" in top_timing else None

    top_faiss = float(top.get("retrieval_score", 0.0))
    top_final = float(top.get("final_score", 0.0))
    if top_faiss < MIN_FAISS_SCORE and top_final < max(threshold, MIN_MULTI_SCORE):
        return PolicyDecision(
            accepted=False,
            reason="min_faiss_score",
            scenarios=[],
            timing_ms=timing_ms,
        )

    if top_final < threshold:
        return PolicyDecision(
            accepted=False,
            reason="final_score_below_threshold",
            scenarios=[],
            timing_ms=timing_ms,
        )

    ambiguous = False
    if allow_multi and len(evidence) >= 2:
        second = evidence[1]
        second_final = float(second.get("final_score", 0.0))
        faiss_gap = abs(float(top.get("retrieval_score", 0.0)) - float(second.get("retrieval_score", 0.0)))
        ce_gap = abs(float(top.get("score", 0.0)) - float(second.get("score", 0.0)))
        final_gap = abs(top_final - second_final)
        ambiguous = second_final >= max(threshold, MIN_MULTI_SCORE) and (
            faiss_gap < FAISS_TOP2_GAP_THRESHOLD
            or ce_gap < CE_TOP2_GAP_THRESHOLD
            or final_gap < MULTI_MATCH_SCORE_GAP
        )

    accepted_rows: list[dict] = []
    if ambiguous and allow_multi:
        multi_threshold = max(threshold, MIN_MULTI_SCORE)
        for row in evidence[: max(1, MAX_MULTI_SCENARIOS)]:
            if float(row.get("final_score", 0.0)) >= multi_threshold:
                accepted_rows.append(row)
    else:
        accepted_rows.append(top)

    scenarios = [
        {
            "name": str(row.get("id") or ""),
            "runnable_name": str(row.get("runnable_name") or row.get("id") or ""),
            "title": str(row.get("title") or row.get("name") or row.get("id") or ""),
            "summary": str(row.get("summary") or ""),
            "run_command": str(row.get("run_command") or ""),
            "score": round(float(row.get("final_score", 0.0)), 4),
            "rank": idx,
            "is_primary": idx == 1,
        }
        for idx, row in enumerate(accepted_rows, start=1)
        if str(row.get("id") or "").strip()
    ]

    return PolicyDecision(
        accepted=bool(scenarios),
        reason="accepted_multi" if len(scenarios) > 1 else ("accepted" if scenarios else "filtered"),
        scenarios=scenarios,
        timing_ms=timing_ms,
    )
