import os


CROSS_ENCODER_MODEL = os.environ.get(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RETRIEVER_MODEL = os.environ.get(
    "RETRIEVER_MODEL",
    "Qwen/Qwen3-Embedding-0.6B",
)

MIN_FAISS_SCORE = float(os.environ.get("MIN_FAISS_SCORE", "0.30"))
FAISS_TOP2_GAP_THRESHOLD = 0.07
CE_TOP2_GAP_THRESHOLD = 1.0
FINAL_CE_WEIGHT = 0.6
FINAL_FAISS_WEIGHT = 0.2
FINAL_BM25_WEIGHT = 0.2
FINAL_LEXICAL_WEIGHT = float(os.environ.get("FINAL_LEXICAL_WEIGHT", "0"))
FINAL_INTENT_WEIGHT = float(os.environ.get("FINAL_INTENT_WEIGHT", "0"))
MIN_QUERY_WORDS = 4
RERANK_SCORE_FLOOR = float(os.environ.get("RERANK_SCORE_FLOOR", "-11.47"))
RERANK_SCORE_CEILING = float(os.environ.get("RERANK_SCORE_CEILING", "9.0"))
RERANK_TOP_FRACTION = float(os.environ.get("RERANK_TOP_FRACTION", "0.25"))
MIN_CE_SCORE = float(os.environ.get("MIN_CE_SCORE", "-7.0"))
MIN_MATCH_SCORE = float(os.environ.get("MIN_MATCH_SCORE", "0.30"))
MIN_MULTI_SCORE = float(os.environ.get("MIN_MULTI_SCORE", "0.28"))
MULTI_MATCH_SCORE_GAP = float(os.environ.get("MULTI_MATCH_SCORE_GAP", "0.08"))
MAX_MULTI_SCENARIOS = int(os.environ.get("MAX_MULTI_SCENARIOS", "2"))
INTENT_MATCH_BOOST = float(os.environ.get("INTENT_MATCH_BOOST", "0"))
INTENT_MISMATCH_PENALTY = float(os.environ.get("INTENT_MISMATCH_PENALTY", "1"))

RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "192"))
RERANK_BATCH_SIZE = int(os.environ.get("RERANK_BATCH_SIZE", "16"))
RERANK_DOC_CHARS = int(os.environ.get("RERANK_DOC_CHARS", "1800"))
RERANK_THREADS = int(
    os.environ.get("RERANK_THREADS", str(min(4, os.cpu_count() or 4)))
)
RERANK_CANDIDATE_K = int(os.environ.get("RERANK_CANDIDATE_K", "0"))
RERANK_ONNX_QUANTIZE = os.environ.get("RERANK_ONNX_QUANTIZE", "1") == "1"
RETRIEVAL_CANDIDATE_K = int(os.environ.get("RETRIEVAL_CANDIDATE_K", "24"))
VECTOR_SEARCH_MULTIPLIER = int(os.environ.get("VECTOR_SEARCH_MULTIPLIER", "6"))
INDEX_CHUNK_SIZE_CHARS = int(os.environ.get("INDEX_CHUNK_SIZE_CHARS", "1200"))
INDEX_CHUNK_OVERLAP_CHARS = int(os.environ.get("INDEX_CHUNK_OVERLAP_CHARS", "200"))
INDEX_SCENARIO_CHUNKS = os.environ.get("INDEX_SCENARIO_CHUNKS", "0") == "1"
RERANK_SUPPORT_PASSAGES = int(os.environ.get("RERANK_SUPPORT_PASSAGES", "2"))
RETRIEVER_BATCH_SIZE = int(os.environ.get("RETRIEVER_BATCH_SIZE", "8"))

BM25_K1 = float(os.environ.get("BM25_K1", "1.5"))
BM25_B = float(os.environ.get("BM25_B", "0.75"))

DEFAULT_BACKEND = os.environ.get("RETRIEVER_BACKEND", "vulkan")
DEFAULT_DEVICE = "auto"
DEFAULT_CPU_ONLY = False
DEFAULT_LLAMA_MODEL = os.environ.get("LLAMA_EMBED_MODEL", "")
DEFAULT_LLAMA_GPU_LAYERS = int(os.environ.get("LLAMA_GPU_LAYERS", "-1"))

DOCS_DIR = os.environ.get("DOCS_DIR", "../docs")
INDEX_DIR = os.environ.get("INDEX_DIR", "faiss-index")
INDEX_PATH = f"{INDEX_DIR}/krkn-scenarios.index"
META_PATH = f"{INDEX_DIR}/krkn-scenarios.meta"
DOCS_CACHE_PATH = f"{INDEX_DIR}/krkn-scenarios.docs.json"
MERGED_DOCS_DEBUG_DIR = os.environ.get(
    "MERGED_DOCS_DEBUG_DIR",
    f"{INDEX_DIR}/merged-scenarios",
)

GITHUB_REPO = os.environ.get("GITHUB_REPO", "https://github.com/krkn-chaos/website")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")
REPO_PATH = os.environ.get("REPO_PATH", "content/en/docs")
KRKN_HUB_REPO = os.environ.get("KRKN_HUB_REPO", "https://github.com/krkn-chaos/krkn-hub")
KRKN_HUB_BRANCH = os.environ.get("KRKN_HUB_BRANCH")
LOCAL_DOCS_PATH = os.environ.get("LOCAL_DOCS_PATH")
try:
    INDEX_TTL_DAYS = float(os.environ.get("INDEX_TTL_DAYS", "7"))
except ValueError:
    INDEX_TTL_DAYS = 7.0

NON_SCENARIO_DOCS = {
    "all_scenarios_env.md",
    "contribute.md",
    "test_your_changes.md",
    "error_cases.md",
    "cerberus.md",
    "chaos-recommender.md",
    "aggregated_docs.md",
}

EXCLUDED_SCENARIO_IDS = {
    item.strip()
    for item in os.environ.get("EXCLUDED_SCENARIO_IDS", "dummy-scenario").split(",")
    if item.strip()
}
