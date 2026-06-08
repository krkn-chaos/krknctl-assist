from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI

from .policy import decide_scenarios
from .ranking import create_ranker, reset_ranker
from .settings import (
    DEFAULT_BACKEND,
    DEFAULT_CPU_ONLY,
    DEFAULT_DEVICE,
    DEFAULT_LLAMA_GPU_LAYERS,
    DEFAULT_LLAMA_MODEL,
    DOCS_CACHE_PATH,
    DOCS_DIR,
    INDEX_PATH,
    INDEX_TTL_DAYS,
    META_PATH,
    MIN_MATCH_SCORE,
)

logger = logging.getLogger(__name__)

DEVICE = os.environ.get("RETRIEVER_DEVICE", DEFAULT_DEVICE)
CPU_ONLY = False
BACKEND = os.environ.get("RETRIEVER_BACKEND", DEFAULT_BACKEND)
LLAMA_MODEL = os.environ.get("LLAMA_EMBED_MODEL", DEFAULT_LLAMA_MODEL)
LLAMA_GPU_LAYERS = int(os.environ.get("LLAMA_GPU_LAYERS", str(DEFAULT_LLAMA_GPU_LAYERS)))

RETRIEVE_K = int(os.environ.get("RETRIEVE_K", "10"))
RERANK_K = int(os.environ.get("RERANK_K", "5"))
FORCE_REINDEX = os.environ.get("FORCE_REINDEX", "false").lower() == "true"
QUERY_CACHE_SIZE = int(os.environ.get("RETRIEVER_QUERY_CACHE_SIZE", "256"))

RELEVANCE_THRESHOLD = float(os.environ.get("RELEVANCE_THRESHOLD", str(MIN_MATCH_SCORE)))
SERVICE_NAME = os.environ.get("RETRIEVER_SERVICE_NAME", "krknctl-assist")
SERVICE_MODEL = os.environ.get("RETRIEVER_SERVICE_MODEL", "krkn-assist")
INDEX_TTL_SECONDS = max(0.0, float(INDEX_TTL_DAYS)) * 86400.0

ranker = None
query_cache: OrderedDict[tuple[str, int, int], list[dict]] = OrderedDict()
_service_ready = False
_init_lock = asyncio.Lock()


def _index_last_modified() -> float:
    timestamps = []
    for path in (INDEX_PATH, META_PATH, DOCS_CACHE_PATH):
        if os.path.exists(path):
            try:
                timestamps.append(os.path.getmtime(path))
            except OSError:
                continue
    return max(timestamps) if timestamps else 0.0


def _index_is_stale() -> bool:
    if INDEX_TTL_SECONDS <= 0:
        return False
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        return True
    if not os.path.exists(DOCS_CACHE_PATH):
        return True
    last_modified = _index_last_modified()
    if last_modified <= 0:
        return True
    return (time.time() - last_modified) >= INDEX_TTL_SECONDS


def cache_get(cache_key: tuple[str, int, int]) -> list[dict] | None:
    if QUERY_CACHE_SIZE <= 0:
        return None
    cached = query_cache.get(cache_key)
    if cached is None:
        return None
    query_cache.move_to_end(cache_key)
    return [dict(row) for row in cached]


def cache_put(cache_key: tuple[str, int, int], results: list[dict]) -> None:
    if QUERY_CACHE_SIZE <= 0:
        return
    query_cache[cache_key] = [dict(row) for row in results]
    query_cache.move_to_end(cache_key)
    while len(query_cache) > QUERY_CACHE_SIZE:
        query_cache.popitem(last=False)


def clear_cache() -> None:
    query_cache.clear()


def rank_with_cache(query: str, retrieve_k: int, rerank_k: int) -> tuple[list[dict], int, bool]:
    if ranker is None:
        raise RuntimeError("Retriever not initialized")
    cache_key = (query, retrieve_k, rerank_k)
    cached = cache_get(cache_key)
    if cached is not None:
        return cached, 0, True
    started = time.perf_counter()
    results = ranker.find_match(query, retrieve_k=retrieve_k, rerank_k=rerank_k)
    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    cache_put(cache_key, results)
    return results, elapsed_ms, False


def documents_indexed() -> int:
    if ranker is None:
        return 0
    doc_ids = getattr(ranker, "doc_ids", None) or []
    return len(doc_ids)


def policy_decision(query: str, evidence: list[dict]):
    return decide_scenarios(
        query=query,
        evidence=evidence,
        threshold=RELEVANCE_THRESHOLD,
        allow_multi=True,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global ranker, _service_ready

    async with _init_lock:
        if not (_service_ready and ranker is not None):
            reset_ranker()
            query_cache.clear()
            ranker = create_ranker(
                device_preference=DEVICE,
                cpu_only=CPU_ONLY,
                backend=BACKEND,
                llama_model_path=LLAMA_MODEL,
                llama_gpu_layers=LLAMA_GPU_LAYERS,
            )

            reindex_reason = None
            if FORCE_REINDEX:
                reindex_reason = "force"
            elif not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
                reindex_reason = "missing"
            elif not os.path.exists(DOCS_CACHE_PATH):
                reindex_reason = "cache_missing"
            elif _index_is_stale():
                reindex_reason = "stale"

            if reindex_reason:
                logger.info("Building FAISS index (reason=%s)", reindex_reason)
                ranker.build_index(DOCS_DIR)

            ranker._init_index_models()
            ranker._load_doc_texts()
            _service_ready = True
            logger.info("Retriever ready")
    yield
