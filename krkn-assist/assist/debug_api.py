import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import service
from .policy import decide_scenarios

logger = logging.getLogger(__name__)


class RetrieveRequest(BaseModel):
    query: str
    k: Optional[int] = None
    retrieve_k: Optional[int] = None
    rerank_k: Optional[int] = None


class DebugRetrieveResponse(BaseModel):
    query: str
    results: list[dict]
    top_match: Optional[str] = None
    policy: dict
    message: str


app = FastAPI(
    title="krkn Retriever Debug Service",
    version="1.0.0",
    description="Debug endpoints for retrieval evidence and policy decisions",
    lifespan=service.lifespan,
)


@app.get("/")
async def root():
    return {
        "service": f"{service.SERVICE_NAME}-debug",
        "model": service.SERVICE_MODEL,
        "backend": service.BACKEND,
        "endpoints": {
            "health": "/health",
            "retrieve": "/retrieve",
            "debug_retrieve": "/debug/retrieve",
            "debug_policy": "/debug/policy",
        },
    }


@app.get("/health")
async def health_check():
    if service.ranker is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return {
        "status": "healthy",
        "service": f"{service.SERVICE_NAME}-debug",
        "model": service.SERVICE_MODEL,
        "backend": service.BACKEND,
        "index_loaded": service.ranker.faiss_index is not None,
        "documents_indexed": service.documents_indexed(),
    }


def _format_debug_rows(rows: list[dict]) -> list[dict]:
    payload: list[dict] = []
    for row in rows:
        payload.append(
            {
                "id": row.get("id"),
                "name": row.get("name", row.get("id")),
                "retrieval_score": round(float(row.get("retrieval_score", 0.0)), 4),
                "rerank_score": round(float(row.get("score", 0.0)), 4),
                "rerank_score_calibrated": round(float(row.get("calibrated_score", 0.0)), 4),
                "final_score": round(float(row.get("final_score", 0.0)), 4),
                "score_percent": round(float(row.get("final_score", 0.0)) * 100.0, 1),
                "timing_ms": row.get("timing_ms"),
            }
        )
    return payload


@app.post("/retrieve", response_model=DebugRetrieveResponse)
async def retrieve(request: RetrieveRequest):
    if service.ranker is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    retrieve_k = request.retrieve_k or service.RETRIEVE_K
    rerank_k = request.rerank_k or service.RERANK_K
    final_k = request.k or rerank_k
    if final_k < 1:
        raise HTTPException(status_code=400, detail="k must be >= 1")
    if retrieve_k < final_k:
        retrieve_k = final_k

    evidence, elapsed_ms, cache_hit = service.rank_with_cache(query, retrieve_k, rerank_k)
    decision = decide_scenarios(query=query, evidence=evidence, threshold=service.RELEVANCE_THRESHOLD)
    evidence = evidence[:final_k]

    logger.info(
        "debug_retrieve_time_ms=%d cache_hit=%s scenarios=%s query=%s",
        elapsed_ms,
        cache_hit,
        ",".join([row.get("name", "") for row in decision.scenarios]),
        query[:120],
    )

    payload = _format_debug_rows(evidence)
    return DebugRetrieveResponse(
        query=query,
        results=payload,
        top_match=payload[0]["id"] if payload else None,
        policy={
            "accepted": decision.accepted,
            "reason": decision.reason,
            "scenarios": decision.scenarios,
            "timing_ms": decision.timing_ms,
            "threshold": service.RELEVANCE_THRESHOLD,
        },
        message=f"Found {len(payload)} relevant scenarios" if payload else "No matching chaos scenarios found",
    )


@app.post("/debug/retrieve")
async def debug_retrieve(request: RetrieveRequest):
    if service.ranker is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    retrieve_k = request.retrieve_k or service.RETRIEVE_K
    rerank_k = request.rerank_k or service.RERANK_K

    evidence, elapsed_ms, cache_hit = service.rank_with_cache(query, retrieve_k, rerank_k)
    return {
        "query": query,
        "cache_hit": cache_hit,
        "elapsed_ms": elapsed_ms,
        "evidence": _format_debug_rows(evidence),
    }


@app.post("/debug/policy")
async def debug_policy(request: RetrieveRequest):
    if service.ranker is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    retrieve_k = request.retrieve_k or service.RETRIEVE_K
    rerank_k = request.rerank_k or service.RERANK_K

    evidence, elapsed_ms, cache_hit = service.rank_with_cache(query, retrieve_k, rerank_k)
    decision = decide_scenarios(query=query, evidence=evidence, threshold=service.RELEVANCE_THRESHOLD)
    current_time = int(time.time())
    return {
        "id": f"debug-{current_time}",
        "query": query,
        "cache_hit": cache_hit,
        "elapsed_ms": elapsed_ms,
        "decision": {
            "accepted": decision.accepted,
            "reason": decision.reason,
            "scenarios": decision.scenarios,
            "timing_ms": decision.timing_ms,
        },
        "evidence": _format_debug_rows(evidence),
    }


@app.post("/debug/cache/clear")
async def debug_clear_cache():
    service.clear_cache()
    return {"cleared": True, "cache_size": 0}
