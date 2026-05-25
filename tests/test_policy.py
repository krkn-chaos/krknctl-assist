from assist.policy import decide_scenarios


def evidence_row(
    scenario_id,
    *,
    final_score=0.8,
    rerank_score=2.0,
    retrieval_score=0.9,
    timing_ms=None,
):
    return {
        "id": scenario_id,
        "title": scenario_id.replace("-", " ").title(),
        "summary": f"{scenario_id} summary",
        "run_command": f"krknctl run {scenario_id}",
        "final_score": final_score,
        "score": rerank_score,
        "retrieval_score": retrieval_score,
        "timing_ms": timing_ms or {},
    }


def test_rejects_short_queries():
    decision = decide_scenarios(
        query="pod outage",
        evidence=[evidence_row("pod-scenarios")],
    )

    assert decision.accepted is False
    assert decision.reason == "query_too_short"
    assert decision.scenarios == []


def test_accepts_best_deduped_evidence():
    decision = decide_scenarios(
        query="create node memory pressure scenario",
        evidence=[
            evidence_row("node-memory-hog", final_score=0.4, timing_ms={"total": 15}),
            evidence_row("node-memory-hog", final_score=0.91, timing_ms={"total": 27}),
        ],
    )

    assert decision.accepted is True
    assert decision.reason == "accepted"
    assert decision.timing_ms == 27
    assert decision.scenarios[0]["name"] == "node-memory-hog"
    assert decision.scenarios[0]["score"] == 0.91


def test_rejects_low_faiss_score():
    decision = decide_scenarios(
        query="create node cpu hog scenario",
        evidence=[evidence_row("node-cpu-hog", retrieval_score=0.1)],
    )

    assert decision.accepted is False
    assert decision.reason == "min_faiss_score"


def test_accepts_when_final_score_is_good_even_if_ce_is_low():
    decision = decide_scenarios(
        query="create node cpu hog scenario",
        evidence=[
            evidence_row(
                "node-cpu-hog",
                final_score=0.8,
                rerank_score=-20.0,
                retrieval_score=0.85,
            )
        ],
    )

    assert decision.accepted is True
    assert decision.reason == "accepted"
    assert [row["name"] for row in decision.scenarios] == ["node-cpu-hog"]


def test_multi_match_filters_conflicting_intent_family():
    decision = decide_scenarios(
        query="create container memory stress scenario",
        evidence=[
            evidence_row("container-scenarios", final_score=0.8),
            evidence_row("pod-scenarios", final_score=0.79),
        ],
    )

    assert decision.accepted is True
    assert decision.reason == "accepted"
    assert [row["name"] for row in decision.scenarios] == ["container-scenarios"]


def test_returns_multiple_scenarios_for_same_intent_family():
    decision = decide_scenarios(
        query="create node network chaos scenario",
        evidence=[
            evidence_row("node-network-filter", final_score=0.82, rerank_score=3.0, retrieval_score=0.88),
            evidence_row("node-interface-down", final_score=0.8, rerank_score=2.7, retrieval_score=0.84),
        ],
    )

    assert decision.accepted is True
    assert decision.reason == "accepted_multi"
    assert [row["name"] for row in decision.scenarios] == [
        "node-network-filter",
        "node-interface-down",
    ]
    assert decision.scenarios[0]["is_primary"] is True
    assert decision.scenarios[1]["is_primary"] is False
