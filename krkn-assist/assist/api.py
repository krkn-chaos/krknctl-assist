import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import service


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETRIEVE_K = service.RETRIEVE_K
RERANK_K = service.RERANK_K
RELEVANCE_THRESHOLD = service.RELEVANCE_THRESHOLD
SERVICE_NAME = service.SERVICE_NAME
SERVICE_MODEL = service.SERVICE_MODEL


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False


class QueryChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class QueryResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[QueryChoice]
    usage: Usage
    scenario_name: Optional[str] = None
    scenario_names: list[str] = Field(default_factory=list)
    scenarios: list[dict] = Field(default_factory=list)
    timing_ms: Optional[int] = None


class ScenarioQueryRequest(BaseModel):
    query: str


class ScenarioQueryResponse(BaseModel):
    query: str
    scenario_name: Optional[str] = None
    relevance_score: Optional[float] = None


app = FastAPI(
    title="krkn Retriever Service",
    version="1.0.0",
    description="Scenario retrieval service",
    lifespan=service.lifespan,
)


def get_user_query_from_messages(messages: list[ChatMessage]) -> str:
    for message in reversed(messages or []):
        if message.role == "user":
            return (message.content or "").strip()
    return ""


@app.get("/")
async def root():
    return {
        "service": SERVICE_NAME,
        "model": SERVICE_MODEL,
        "backend": service.BACKEND,
        "retrieve_k": RETRIEVE_K,
        "rerank_k": RERANK_K,
        "relevance_threshold": RELEVANCE_THRESHOLD,
        "endpoints": {
            "health": "/health",
            "query": "/v1/chat/completions",
        },
    }


@app.get("/health")
async def health_check():
    if service.ranker is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "model": SERVICE_MODEL,
        "backend": service.BACKEND,
        "index_loaded": service.ranker.faiss_index is not None,
        "documents_indexed": service.documents_indexed(),
    }


@app.post("/v1/chat/completions", response_model=QueryResponse)
async def chat_completions(request: QueryRequest):
    if service.ranker is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported")

    user_query = get_user_query_from_messages(request.messages)
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found in request")

    results, elapsed_ms, cache_hit = service.rank_with_cache(user_query, RETRIEVE_K, RERANK_K)
    decision = service.policy_decision(user_query, results)

    scenario_name: str | None = None
    if decision.scenarios:
        scenario_name = (
            decision.scenarios[0].get("runnable_name")
            or decision.scenarios[0].get("name")
            or None
        )
    scenario_names = [
        row.get("runnable_name") or row.get("name", "")
        for row in decision.scenarios
        if row.get("runnable_name") or row.get("name")
    ]

    logger.info(
        "chat_completion_time_ms=%d cache_hit=%s scenarios=%s query=%s",
        elapsed_ms,
        cache_hit,
        ",".join([row.get("name", "") for row in decision.scenarios]),
        user_query[:120],
    )

    current_time = int(time.time())
    response_id = f"chatcmpl-{current_time}"

    if not decision.scenarios:
        response_content = "No confident scenario match found"
    else:
        # Keep multi-match selection internally, but only surface the primary
        # scenario in chat responses for now.
        response_content = f"Scenario: {decision.scenarios[0]['name']}"

    prompt_tokens = len(user_query.split())
    completion_tokens = len(response_content.split())

    return QueryResponse(
        id=response_id,
        object="chat.completion",
        created=current_time,
        model=request.model or SERVICE_MODEL,
        choices=[
            QueryChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_content),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        scenario_name=scenario_name,
        scenario_names=scenario_names,
        scenarios=decision.scenarios,
        timing_ms=decision.timing_ms,
    )


@app.post("/query", response_model=ScenarioQueryResponse)
async def legacy_query(request: ScenarioQueryRequest):
    if service.ranker is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    query = (request.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    results, elapsed_ms, cache_hit = service.rank_with_cache(query, RETRIEVE_K, RERANK_K)
    decision = service.policy_decision(query, results)
    scenario_name: str | None = decision.scenarios[0]["name"] if decision.scenarios else None

    logger.info(
        "legacy_query_time_ms=%d cache_hit=%s scenarios=%s query=%s",
        elapsed_ms,
        cache_hit,
        ",".join([row.get("name", "") for row in decision.scenarios]),
        query[:120],
    )

    return ScenarioQueryResponse(
        query=query,
        scenario_name=scenario_name,
        relevance_score=round(float(decision.scenarios[0]["score"]), 4) if decision.scenarios else 0.0,
    )


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()
