import json
import logging
import os
import pickle
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from .settings import (
    BM25_B,
    BM25_K1,
    CROSS_ENCODER_MODEL,
    DEFAULT_BACKEND,
    DEFAULT_LLAMA_GPU_LAYERS,
    DEFAULT_LLAMA_MODEL,
    DOCS_DIR,
    DOCS_CACHE_PATH,
    FINAL_BM25_WEIGHT,
    FINAL_CE_WEIGHT,
    FINAL_FAISS_WEIGHT,
    FINAL_INTENT_WEIGHT,
    FINAL_LEXICAL_WEIGHT,
    GITHUB_BRANCH,
    GITHUB_REPO,
    INDEX_CHUNK_OVERLAP_CHARS,
    INDEX_SCENARIO_CHUNKS,
    INDEX_CHUNK_SIZE_CHARS,
    INDEX_DIR,
    INDEX_PATH,
    KRKN_HUB_BRANCH,
    KRKN_HUB_REPO,
    LOCAL_DOCS_PATH,
    META_PATH,
    RERANK_SCORE_CEILING,
    RERANK_SCORE_FLOOR,
    REPO_PATH,
    INTENT_MATCH_BOOST,
    INTENT_MISMATCH_PENALTY,
    RERANK_BATCH_SIZE,
    RERANK_CANDIDATE_K,
    RERANK_DOC_CHARS,
    RERANK_MAX_LENGTH,
    RERANK_ONNX_QUANTIZE,
    RERANK_SUPPORT_PASSAGES,
    RERANK_THREADS,
    RERANK_TOP_FRACTION,
    RETRIEVAL_CANDIDATE_K,
    RETRIEVER_BATCH_SIZE,
    RETRIEVER_MODEL,
    VECTOR_SEARCH_MULTIPLIER,
)
from .ingestion import build_scenario_documents, load_local_scenario_docs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexEntry:
    scenario_id: str
    chunk_id: str
    text: str
    passage: str


_LEXICAL_STOPWORDS = {
    "about",
    "and",
    "are",
    "can",
    "create",
    "from",
    "how",
    "into",
    "label",
    "labeled",
    "labels",
    "the",
    "this",
    "through",
    "while",
    "with",
}

_INTENT_FAMILY_KEYWORDS = {
    "pod": {"pod", "pods"},
    "container": {"container", "containers"},
    "node": {"node", "nodes", "worker", "workers"},
    "service": {"service", "services"},
    "network": {"network", "latency", "packet", "packets", "bandwidth", "ingress", "egress"},
    "dns": {"dns"},
    "pvc": {"pvc", "persistentvolumeclaim", "persistent-volume-claim"},
    "time": {"time", "clock"},
    "zone": {"zone", "zones", "zonal"},
    "power": {"power", "outage", "outages"},
    "memory": {"memory", "ram"},
    "cpu": {"cpu"},
    "io": {"io", "disk", "storage"},
    "http": {"http"},
    "vmi": {"vmi", "vm", "vms", "virtualmachine", "virtualmachines", "kubevirt"},
    "application": {"application", "applications", "app", "apps"},
    "aurora": {"aurora"},
    "efs": {"efs"},
    "etcd": {"etcd"},
}


def probe_vulkan_devices() -> list[dict]:
    devices: list[dict] = []
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except FileNotFoundError:
        return devices
    except Exception as exc:
        logger.warning("vulkaninfo probe failed: %s", exc)
        return devices

    for line in (result.stdout + result.stderr).splitlines():
        summary = line.strip()
        if "deviceName" not in summary or "=" not in summary:
            continue
        name = summary.split("=", 1)[1].strip()
        normalized_name = name.lower()
        is_software = any(
            renderer in normalized_name
            for renderer in ("llvmpipe", "lavapipe", "softpipe", "swrast", "swiftshader")
        )
        devices.append(
            {
                "index": len(devices),
                "name": name,
                "software": is_software,
            }
        )
    return devices


def log_vulkan_device_status(devices: list[dict]) -> None:
    if not devices:
        logger.warning("vulkaninfo unavailable; unable to confirm Vulkan GPU device")
        return

    hardware_devices = [device for device in devices if not device.get("software")]
    if hardware_devices:
        active = hardware_devices[0]
        logger.info(
            "Vulkan hardware device active: %s (index=%s)",
            active.get("name"),
            active.get("index"),
        )
        return

    renderer_names = ", ".join(str(device.get("name")) for device in devices)
    logger.warning(
        "Vulkan software renderer active: %s",
        renderer_names,
    )


def score_to_match(
    ce_score: float,
    faiss_score: float,
    bm25_score: float,
    lexical_score: float = 0.0,
    intent_score: float = 0.0,
) -> float:
    ce_calibrated = calibrate_rerank_score(ce_score)
    faiss_score = max(0.0, min(1.0, float(faiss_score)))
    bm25_score = max(0.0, min(1.0, float(bm25_score)))
    lexical_score = max(0.0, min(1.0, float(lexical_score)))
    intent_score = max(0.0, min(1.0, float(intent_score)))
    weight_total = (
        FINAL_CE_WEIGHT
        + FINAL_FAISS_WEIGHT
        + FINAL_BM25_WEIGHT
        + FINAL_LEXICAL_WEIGHT
        + FINAL_INTENT_WEIGHT
    )
    if weight_total <= 0:
        return 0.0
    return (
        (FINAL_CE_WEIGHT * ce_calibrated)
        + (FINAL_FAISS_WEIGHT * faiss_score)
        + (FINAL_BM25_WEIGHT * bm25_score)
        + (FINAL_LEXICAL_WEIGHT * lexical_score)
        + (FINAL_INTENT_WEIGHT * intent_score)
    ) / weight_total


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _lexical_terms(text: str) -> set[str]:
    return {
        token
        for token in _tokenize(text)
        if len(token) >= 3 and token not in _LEXICAL_STOPWORDS
    }


def _lexical_overlap_score(query: str, text: str) -> float:
    query_terms = _lexical_terms(query)
    if not query_terms:
        return 0.0
    text_terms = set(_tokenize(text))
    return len(query_terms & text_terms) / len(query_terms)


def _detect_query_intent_families(query: str) -> set[str]:
    query_terms = set(_tokenize(query))
    families = {
        family
        for family, keywords in _INTENT_FAMILY_KEYWORDS.items()
        if query_terms & keywords
    }
    if "container" in families:
        families.discard("pod")
    return families


def _scenario_intent_families(scenario_id: str, text: str) -> set[str]:
    family_terms = set(_tokenize(f"{scenario_id} {_extract_title(text, scenario_id)}"))
    families = {
        family
        for family, keywords in _INTENT_FAMILY_KEYWORDS.items()
        if family_terms & keywords
    }

    if scenario_id.startswith("node-memory"):
        families.add("memory")
    if scenario_id.startswith("node-cpu"):
        families.add("cpu")
    if scenario_id.startswith("node-io"):
        families.add("io")
    if scenario_id.startswith("pod-"):
        families.add("pod")
    if scenario_id.startswith("container-"):
        families.add("container")
    if scenario_id.startswith("node-"):
        families.add("node")
    if "network" in scenario_id:
        families.add("network")
    return families


def _intent_alignment_score(query: str, scenario_id: str, text: str) -> float:
    query_families = _detect_query_intent_families(query)
    if not query_families:
        return 0.0
    scenario_families = _scenario_intent_families(scenario_id, text)
    if not scenario_families:
        return 0.0
    return 1.0 if query_families & scenario_families else 0.0


def _filter_conflicting_intent_results(
    query: str,
    results: list[dict],
    doc_texts: dict[str, str],
) -> list[dict]:
    query_families = _detect_query_intent_families(query)
    if not query_families:
        return results

    has_positive_intent_match = any(float(row.get("intent_score", 0.0)) > 0.0 for row in results)
    if not has_positive_intent_match:
        return results

    filtered: list[dict] = []
    for row in results:
        scenario_id = str(row.get("id") or "")
        scenario_families = _scenario_intent_families(scenario_id, doc_texts.get(scenario_id, ""))
        if (
            float(row.get("intent_score", 0.0)) > 0.0
            or not scenario_families
            or (query_families & scenario_families)
        ):
            filtered.append(row)
    return filtered or results


def _bm25_scores(query: str, doc_ids: list[str], doc_texts: dict[str, str]) -> dict[str, float]:
    terms = _tokenize(query)
    if not terms or not doc_ids:
        return {doc_id: 0.0 for doc_id in doc_ids}

    query_terms = set(terms)
    doc_term_freqs: list[dict[str, int]] = []
    doc_lengths: list[int] = []
    doc_freqs = {term: 0 for term in query_terms}

    for doc_id in doc_ids:
        tokens = _tokenize(doc_texts.get(doc_id, ""))
        length = len(tokens)
        doc_lengths.append(length)
        term_counts: dict[str, int] = {}
        if tokens:
            for token in tokens:
                if token in query_terms:
                    term_counts[token] = term_counts.get(token, 0) + 1
        for term in query_terms:
            if term_counts.get(term, 0) > 0:
                doc_freqs[term] += 1
        doc_term_freqs.append(term_counts)

    total_docs = len(doc_ids)
    avgdl = sum(doc_lengths) / total_docs if total_docs else 0.0
    scores: dict[str, float] = {doc_id: 0.0 for doc_id in doc_ids}

    for idx, doc_id in enumerate(doc_ids):
        dl = float(doc_lengths[idx])
        tf_map = doc_term_freqs[idx]
        if not tf_map:
            continue
        score = 0.0
        for term in query_terms:
            tf = float(tf_map.get(term, 0))
            if tf <= 0:
                continue
            df = float(doc_freqs.get(term, 0))
            idf = np.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (dl / avgdl if avgdl else 0.0))
            score += idf * ((tf * (BM25_K1 + 1.0)) / (denom if denom else 1.0))
        scores[doc_id] = score
    return scores


def _normalize_score(score: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 1.0 if score > 0 else 0.0
    return max(0.0, min(1.0, (score - min_value) / (max_value - min_value)))


def calibrate_rerank_score(score: float) -> float:
    # Normalize the cross-encoder logit into [0, 1] using the observed score floor.
    floor = float(RERANK_SCORE_FLOOR)
    ceiling = float(RERANK_SCORE_CEILING)
    if ceiling <= floor:
        return 1.0 / (1.0 + np.exp(-float(score - floor)))
    return max(0.0, min(1.0, (float(score) - floor) / (ceiling - floor)))


def _strip_frontmatter(text: str) -> str:
    if not text:
        return ""
    if not text.startswith("---"):
        return text.strip()
    parts = text.split("\n---", 1)
    if len(parts) != 2:
        return text.strip()
    return parts[1].strip()


def _extract_title(text: str, scenario_id: str) -> str:
    match = re.search(r"(?im)^title:\s*(.+?)\s*$", text or "")
    if match:
        return match.group(1).strip().strip("'\"")
    return scenario_id.replace("-", " ").title()


def _scenario_aliases(scenario_id: str, title: str) -> list[str]:
    aliases = {
        scenario_id.strip(),
        scenario_id.replace("-", " ").strip(),
        title.strip(),
        f"krknctl run {scenario_id}".strip(),
    }
    return [alias for alias in sorted(aliases) if alias]


def _summary_paragraph(text: str) -> str:
    body = _strip_frontmatter(text)
    body = re.sub(r"```.*?```", " ", body, flags=re.DOTALL)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    for paragraph in paragraphs:
        cleaned = re.sub(r"\s+", " ", paragraph).strip()
        if cleaned:
            return cleaned[:600]
    return ""


def _extract_run_command(text: str, scenario_id: str) -> str:
    match = re.search(r"(?im)\bkrknctl\s+run\s+[a-z0-9][a-z0-9-]*(?:[^\n`]*)", text or "")
    if not match:
        return f"krknctl run {scenario_id}"
    return re.sub(r"\s+", " ", match.group(0)).strip()


def _extract_run_scenario_id(text: str, scenario_id: str) -> str:
    command = _extract_run_command(text, scenario_id)
    match = re.search(r"(?i)\bkrknctl\s+run\s+([a-z0-9][a-z0-9-]*)", command)
    if not match:
        return scenario_id
    return match.group(1).strip() or scenario_id


def _search_text_for_scenario(scenario_id: str, text: str) -> str:
    title = _extract_title(text, scenario_id)
    aliases = _scenario_aliases(scenario_id, title)
    summary = _summary_paragraph(text)
    header_parts = [
        f"Scenario ID: {scenario_id}",
        f"Scenario Name: {title}",
        f"Aliases: {', '.join(aliases)}",
        f"Command: krknctl run {scenario_id}",
    ]
    if summary:
        header_parts.append(f"Summary: {summary}")
    body = _strip_frontmatter(text)
    return "\n".join(header_parts) + "\n\n" + body


def _chunk_paragraphs(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        paragraph_len = len(paragraph)
        projected_len = current_len + paragraph_len + (2 if current else 0)
        if current and projected_len > chunk_size:
            chunk = "\n\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
            if overlap > 0:
                overlap_parts: list[str] = []
                overlap_len = 0
                for existing in reversed(current):
                    overlap_parts.insert(0, existing)
                    overlap_len += len(existing) + 2
                    if overlap_len >= overlap:
                        break
                current = overlap_parts
                current_len = len("\n\n".join(current))
            else:
                current = []
                current_len = 0
        current.append(paragraph)
        current_len = len("\n\n".join(current))

    final_chunk = "\n\n".join(current).strip()
    if final_chunk:
        chunks.append(final_chunk)
    return chunks


def _build_index_entries(docs: list[tuple[str, str]]) -> list[IndexEntry]:
    entries: list[IndexEntry] = []
    for scenario_id, text in docs:
        search_text = _search_text_for_scenario(scenario_id, text)
        title = _extract_title(text, scenario_id)
        aliases = _scenario_aliases(scenario_id, title)
        synopsis = "\n".join(
            [
                f"Scenario ID: {scenario_id}",
                f"Scenario Name: {title}",
                f"Aliases: {', '.join(aliases)}",
                f"Command: krknctl run {scenario_id}",
            ]
        )
        body = _strip_frontmatter(text)
        passages = _chunk_paragraphs(body, INDEX_CHUNK_SIZE_CHARS, INDEX_CHUNK_OVERLAP_CHARS)
        if not passages:
            passages = [body or search_text]

        overview = _summary_paragraph(text) or passages[0]
        entries.append(
            IndexEntry(
                scenario_id=scenario_id,
                chunk_id=f"{scenario_id}::overview",
                passage=overview,
                text=search_text,
            )
        )
        if not INDEX_SCENARIO_CHUNKS:
            continue

        for idx, passage in enumerate(passages, start=1):
            entries.append(
                IndexEntry(
                    scenario_id=scenario_id,
                    chunk_id=f"{scenario_id}::chunk::{idx}",
                    passage=passage,
                    text=f"{synopsis}\n\nReference:\n{passage}",
                )
            )
    return entries


def _scenario_support_text(
    *,
    scenario_id: str,
    scenario_text: str,
    support_passages: list[str],
) -> str:
    title = _extract_title(scenario_text, scenario_id)
    aliases = _scenario_aliases(scenario_id, title)
    parts = [
        f"Scenario ID: {scenario_id}",
        f"Scenario Name: {title}",
        f"Aliases: {', '.join(aliases)}",
        f"Command: krknctl run {scenario_id}",
    ]
    summary = _summary_paragraph(scenario_text)
    if summary:
        parts.append(f"Summary: {summary}")

    if support_passages:
        parts.append("Top matching passages:")
        parts.extend(support_passages[: max(1, RERANK_SUPPORT_PASSAGES)])
    else:
        parts.append(compact_for_reranking(_strip_frontmatter(scenario_text)))

    return compact_for_reranking("\n\n".join(parts))


def resolve_backend(backend: str, llama_model_path: str) -> str:
    normalized_backend = (backend or "vulkan").lower()
    if normalized_backend == "torch":
        raise ValueError("torch retriever backend is disabled; use RETRIEVER_BACKEND=vulkan")
    if normalized_backend not in {"auto", "vulkan"}:
        raise ValueError(f"Unsupported retriever backend: {backend!r}")
    if not llama_model_path:
        raise ValueError("Vulkan backend requires LLAMA_EMBED_MODEL")
    return "vulkan"


def compact_for_reranking(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text or "").strip()
    if RERANK_DOC_CHARS > 0 and len(text) > RERANK_DOC_CHARS:
        head = text[:RERANK_DOC_CHARS]
        last_break = max(head.rfind("\n"), head.rfind(". "), head.rfind(" "))
        if last_break > 300:
            head = head[:last_break]
        return head
    return text


class OnnxCrossEncoder:
    def __init__(self, model_name: str = CROSS_ENCODER_MODEL, cache_dir: str | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )
        self._session = None
        self._tokenizer = None
        self._hf_model = None
        self._backend = None
        self._input_names: set[str] = set()
        self._init()

    def _onnx_model_dir(self) -> Path:
        return Path(self.cache_dir) / "onnx_rerankers" / self.model_name.replace("/", "__")

    def _init(self) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        errors: list[str] = []

        try:
            import onnxruntime as ort

            onnx_dir = self._onnx_model_dir()
            onnx_file = onnx_dir / "model.onnx"
            runtime_file = onnx_dir / "model-int8.onnx"

            if not onnx_file.exists():
                self._export_to_onnx(onnx_dir)

            if onnx_file.exists():
                runtime_file = (
                    self._quantize_onnx(onnx_file, runtime_file)
                    if RERANK_ONNX_QUANTIZE
                    else onnx_file
                )
                options = ort.SessionOptions()
                options.intra_op_num_threads = max(1, RERANK_THREADS)
                options.inter_op_num_threads = 1
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._session = ort.InferenceSession(
                    str(runtime_file),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                self._input_names = {item.name for item in self._session.get_inputs()}
                self._backend = "onnx"
                return
        except Exception as exc:
            errors.append(str(exc))

        detail = "; ".join(errors) if errors else "unknown error"
        raise RuntimeError(
            "ONNX reranker initialization failed and transformers inference fallback is disabled: "
            f"{detail}"
        )

    def _export_to_onnx(self, out_dir: Path) -> None:
        errors: list[str] = []
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification

            model = ORTModelForSequenceClassification.from_pretrained(
                self.model_name,
                export=True,
            )
            model.save_pretrained(str(out_dir))
            return
        except Exception as exc:
            errors.append(f"optimum export failed: {exc}")

        try:
            import torch
            from transformers import AutoModelForSequenceClassification

            model = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval()
            dummy = self._tokenizer(
                "query",
                "document",
                return_tensors="pt",
                truncation=True,
                max_length=RERANK_MAX_LENGTH,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            input_names = ["input_ids", "attention_mask"]
            model_args = [dummy["input_ids"], dummy["attention_mask"]]
            dynamic_axes = {
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            }
            if "token_type_ids" in dummy:
                input_names.append("token_type_ids")
                model_args.append(dummy["token_type_ids"])
                dynamic_axes["token_type_ids"] = {0: "batch", 1: "seq"}
            torch.onnx.export(
                model,
                tuple(model_args),
                str(out_dir / "model.onnx"),
                input_names=input_names,
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=14,
            )
            return
        except Exception as exc:
            errors.append(f"torch onnx export failed: {exc}")

        raise RuntimeError("; ".join(errors))

    @staticmethod
    def _quantize_onnx(src: Path, dst: Path) -> Path:
        if dst.exists():
            return dst
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
            return dst
        except Exception:
            return src

    def compute_score(self, pairs: list[list[str]], batch_size: int | None = None) -> list[float]:
        batch_size = batch_size or RERANK_BATCH_SIZE
        if self._backend == "onnx":
            return self._score_onnx(pairs, batch_size)
        raise RuntimeError("ONNX reranker is not initialized")

    def _score_onnx(self, pairs: list[list[str]], batch_size: int) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = self._tokenizer(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch],
                padding=True,
                truncation=True,
                max_length=RERANK_MAX_LENGTH,
                return_tensors="np",
            )
            ort_inputs = {
                name: encoded[name].astype(np.int64)
                for name in self._input_names
                if name in encoded
            }
            logits = self._session.run(None, ort_inputs)[0]
            batch_scores = (
                (logits[:, 1] - logits[:, 0]).tolist()
                if logits.ndim == 2 and logits.shape[1] == 2
                else logits[:, 0].tolist()
            )
            scores.extend(batch_scores)
        return scores

class BaseRanker:
    def __init__(self) -> None:
        self.faiss_index = None
        self.doc_ids: list[str] = []
        self.doc_texts: dict[str, str] = {}
        self.index_entries: list[IndexEntry] = []
        self.search_texts: dict[str, str] = {}
        self.cross_encoder: OnnxCrossEncoder | None = None
        self._load_index()

    def _init_models(self) -> None:
        raise NotImplementedError

    def _init_index_models(self) -> None:
        self._init_models()

    def _embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError

    def _embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def _scenario_docs(self, docs_dir: str) -> list[tuple[str, str]]:
        return load_local_scenario_docs(docs_dir)

    def _load_index(self) -> None:
        if Path(INDEX_PATH).exists() and Path(META_PATH).exists():
            self.faiss_index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as handle:
                payload = pickle.load(handle)
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                self.index_entries = [
                    IndexEntry(
                        scenario_id=str(row.get("scenario_id") or ""),
                        chunk_id=str(row.get("chunk_id") or ""),
                        text=str(row.get("text") or ""),
                        passage=str(row.get("passage") or ""),
                    )
                    for row in payload
                    if row.get("scenario_id")
                ]
            elif isinstance(payload, list):
                self.index_entries = [
                    IndexEntry(
                        scenario_id=str(doc_id),
                        chunk_id=f"{doc_id}::legacy",
                        text="",
                        passage="",
                    )
                    for doc_id in payload
                    if str(doc_id).strip()
                ]
            self.doc_ids = sorted({entry.scenario_id for entry in self.index_entries})

    def _load_doc_texts(self, docs_dir: str = DOCS_DIR) -> None:
        if self.doc_texts:
            return
        docs_cache = Path(DOCS_CACHE_PATH)
        if docs_cache.exists():
            try:
                payload = json.loads(docs_cache.read_text(encoding="utf-8"))
                docs = payload.get("docs", []) if isinstance(payload, dict) else []
                self.doc_texts = {
                    str(row.get("id")): str(row.get("text"))
                    for row in docs
                    if row.get("id") and row.get("text")
                }
                if self.doc_texts:
                    self.search_texts = {
                        doc_id: _search_text_for_scenario(doc_id, text)
                        for doc_id, text in self.doc_texts.items()
                    }
                    return
            except Exception:
                self.doc_texts = {}

        self.doc_texts = {doc_id: text for doc_id, text in self._scenario_docs(docs_dir)}
        self.search_texts = {
            doc_id: _search_text_for_scenario(doc_id, text)
            for doc_id, text in self.doc_texts.items()
        }

    def build_index(self, docs_dir: str = DOCS_DIR) -> None:
        self._init_index_models()
        docs = build_scenario_documents(
            docs_dir=docs_dir,
            github_repo=GITHUB_REPO,
            repo_path=REPO_PATH,
            github_branch=GITHUB_BRANCH,
            krkn_hub_repo=KRKN_HUB_REPO,
            krkn_hub_branch=KRKN_HUB_BRANCH,
            local_docs_path=LOCAL_DOCS_PATH,
        )
        if not docs:
            raise RuntimeError("No documents available for indexing")
        scenario_ids = [doc_id for doc_id, _ in docs]
        self.doc_texts = dict(docs)
        self.search_texts = {
            doc_id: _search_text_for_scenario(doc_id, text)
            for doc_id, text in docs
        }
        self.index_entries = _build_index_entries(docs)
        texts = [entry.text for entry in self.index_entries]
        logger.info("Scenario IDs (%d):\n%s", len(scenario_ids), "\n".join(scenario_ids))
        embeddings = self._embed_documents(texts)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        Path(INDEX_DIR).mkdir(exist_ok=True, parents=True)
        try:
            scenario_list_path = Path(INDEX_DIR) / "krkn-scenarios.list.txt"
            scenario_list_path.write_text(
                "\n".join(scenario_ids) + "\n",
                encoding="utf-8",
            )
            logger.info("Scenario list written to %s", scenario_list_path)
        except OSError as exc:
            logger.warning("Failed to write scenario list: %s", exc)
        faiss.write_index(index, INDEX_PATH)
        with open(META_PATH, "wb") as handle:
            pickle.dump(
                [
                    {
                        "scenario_id": entry.scenario_id,
                        "chunk_id": entry.chunk_id,
                        "text": entry.text,
                        "passage": entry.passage,
                    }
                    for entry in self.index_entries
                ],
                handle,
            )
        docs_payload = {
            "docs": [{"id": doc_id, "text": text} for doc_id, text in docs],
            "entries": [
                {
                    "scenario_id": entry.scenario_id,
                    "chunk_id": entry.chunk_id,
                    "passage": entry.passage,
                }
                for entry in self.index_entries
            ],
        }
        Path(DOCS_CACHE_PATH).write_text(
            json.dumps(docs_payload, indent=2),
            encoding="utf-8",
        )
        self.faiss_index = index
        self.doc_ids = scenario_ids

    def find_match(self, query: str, retrieve_k: int = 10, rerank_k: int = 5) -> list[dict]:
        if self.faiss_index is None:
            raise RuntimeError("Index not found. Build it first.")
        self._init_models()
        self._load_doc_texts()
        return run_retrieval(
            query=query,
            retrieve_k=retrieve_k,
            rerank_k=rerank_k,
            faiss_index=self.faiss_index,
            index_entries=self.index_entries,
            doc_ids=self.doc_ids,
            doc_texts=self.doc_texts,
            search_texts=self.search_texts,
            embed_query=self._embed_query,
            reranker=self.cross_encoder,
        )


class VulkanRanker(BaseRanker):
    def __init__(
        self,
        model_path: str,
        gpu_layers: int = -1,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
    ) -> None:
        if not model_path:
            raise ValueError("Vulkan backend requires LLAMA_EMBED_MODEL")
        from llama_cpp import Llama

        self.model_path = model_path
        self.vulkan_devices = probe_vulkan_devices()
        log_vulkan_device_status(self.vulkan_devices)
        hardware_devices = [device for device in self.vulkan_devices if not device.get("software")]
        preferred_devices = hardware_devices or self.vulkan_devices
        self.main_gpu = preferred_devices[0]["index"] if preferred_devices else None
        self.gpu_layers = gpu_layers
        self.cross_encoder_model_name = cross_encoder_model
        llama_kwargs = {
            "model_path": model_path,
            "embedding": True,
            "n_gpu_layers": self.gpu_layers,
            "verbose": False,
            "n_batch": 512,
        }
        if self.main_gpu is not None and self.gpu_layers != 0:
            llama_kwargs["main_gpu"] = int(self.main_gpu)
        try:
            self.llm = Llama(**llama_kwargs)
        except TypeError:
            llama_kwargs.pop("main_gpu", None)
            self.llm = Llama(**llama_kwargs)
        super().__init__()

    def _init_models(self) -> None:
        if self.cross_encoder is None:
            self.cross_encoder = OnnxCrossEncoder(self.cross_encoder_model_name)

    def _init_index_models(self) -> None:
        return None

    @staticmethod
    def _extract_embedding(response):
        if isinstance(response, dict):
            if "data" in response and response["data"]:
                embedding = response["data"][0].get("embedding")
                if embedding is not None:
                    return embedding
            if "embedding" in response:
                return response["embedding"]
        if isinstance(response, list) and response and isinstance(response[0], (float, int)):
            return response
        raise RuntimeError("Cannot parse embedding from llama.cpp response")

    def _embed_text(self, text: str) -> np.ndarray:
        response = self.llm.create_embedding(text)
        vector = np.array(self._extract_embedding(response), dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def _embed_query(self, query: str) -> np.ndarray:
        return self._embed_text(query)

    def _embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed_text(text) for text in texts]).astype(np.float32)


def run_retrieval(
    query: str,
    retrieve_k: int,
    rerank_k: int,
    faiss_index,
    index_entries: list[IndexEntry],
    doc_ids: list[str],
    doc_texts: dict[str, str],
    search_texts: dict[str, str],
    embed_query,
    reranker: OnnxCrossEncoder,
) -> list[dict]:
    started = time.perf_counter()
    retrieval_started = time.perf_counter()
    query_embedding = embed_query(query).reshape(1, -1)
    vector_top_k = min(
        len(index_entries),
        max(
            retrieve_k,
            RETRIEVAL_CANDIDATE_K,
            rerank_k * VECTOR_SEARCH_MULTIPLIER,
            retrieve_k * VECTOR_SEARCH_MULTIPLIER,
        ),
    )
    scores, indices = faiss_index.search(query_embedding, vector_top_k)

    vector_scores: dict[str, list[float]] = defaultdict(list)
    support_passages: dict[str, list[tuple[float, str]]] = defaultdict(list)
    if vector_top_k > 0:
        for offset, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(index_entries):
                continue
            entry = index_entries[idx]
            raw_score = float(scores[0][offset])
            vector_scores[entry.scenario_id].append(raw_score)
            if entry.passage:
                support_passages[entry.scenario_id].append((raw_score, entry.passage))

    faiss_scores: dict[str, float] = {}
    for scenario_id, values in vector_scores.items():
        ranked = sorted(values, reverse=True)
        best = ranked[0]
        mean_top = float(sum(ranked[: min(3, len(ranked))])) / float(min(3, len(ranked)))
        faiss_scores[scenario_id] = (0.8 * best) + (0.2 * mean_top)

    scenario_ids = sorted(set(doc_ids) | set(doc_texts.keys()))
    bm25_scores = _bm25_scores(query, scenario_ids, search_texts)
    candidate_k = min(
        len(scenario_ids),
        max(retrieve_k, RETRIEVAL_CANDIDATE_K, rerank_k * 4),
    )
    bm25_top_ids = [
        doc_id
        for doc_id, _ in sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)[:candidate_k]
    ]
    faiss_top_ids = [
        doc_id
        for doc_id, _ in sorted(faiss_scores.items(), key=lambda item: item[1], reverse=True)[:candidate_k]
    ]

    candidate_ids = list(dict.fromkeys(faiss_top_ids + bm25_top_ids))
    retrieve_ms = (time.perf_counter() - retrieval_started) * 1000
    query_intent_families = _detect_query_intent_families(query)

    faiss_values = list(faiss_scores.values()) or [0.0]
    min_faiss = min(faiss_values)
    max_faiss = max(faiss_values)
    bm25_values = [bm25_scores.get(doc_id, 0.0) for doc_id in candidate_ids]
    min_bm25 = min(bm25_values) if bm25_values else 0.0
    max_bm25 = max(bm25_values) if bm25_values else 0.0
    retriever_weight_total = FINAL_FAISS_WEIGHT + FINAL_BM25_WEIGHT + FINAL_LEXICAL_WEIGHT

    candidates = []
    for doc_id in candidate_ids:
        faiss_raw = faiss_scores.get(doc_id, 0.0)
        bm25_raw = bm25_scores.get(doc_id, 0.0)
        faiss_norm = _normalize_score(faiss_raw, min_faiss, max_faiss)
        bm25_norm = _normalize_score(bm25_raw, min_bm25, max_bm25)
        lexical_score = _lexical_overlap_score(query, search_texts.get(doc_id, ""))
        intent_score = _intent_alignment_score(query, doc_id, doc_texts.get(doc_id, ""))
        if retriever_weight_total > 0:
            retrieval_score = (
                (FINAL_FAISS_WEIGHT * faiss_norm)
                + (FINAL_BM25_WEIGHT * bm25_norm)
                + (FINAL_LEXICAL_WEIGHT * lexical_score)
            ) / retriever_weight_total
        else:
            retrieval_score = faiss_norm
        ranked_support = sorted(
            support_passages.get(doc_id, []),
            key=lambda item: item[0],
            reverse=True,
        )
        unique_support: list[str] = []
        seen_support = set()
        query_terms = set(_tokenize(query))
        rerank_candidates = sorted(
            ranked_support,
            key=lambda item: (
                len(query_terms & set(_tokenize(item[1]))),
                item[0],
            ),
            reverse=True,
        )
        for _, passage in rerank_candidates:
            cleaned = passage.strip()
            if not cleaned or cleaned in seen_support:
                continue
            unique_support.append(cleaned)
            seen_support.add(cleaned)
            if len(unique_support) >= max(1, RERANK_SUPPORT_PASSAGES):
                break
        candidates.append(
            {
                "id": doc_id,
                "text": doc_texts.get(doc_id, ""),
                "retrieval_score": retrieval_score,
                "faiss_score": faiss_norm,
                "bm25_score": bm25_norm,
                "lexical_score": lexical_score,
                "intent_score": intent_score,
                "support_passages": unique_support,
            }
        )

    candidates.sort(key=lambda row: row["retrieval_score"], reverse=True)
    if not candidates:
        return []

    rerank_window = max(1, int(np.ceil(len(candidates) * max(0.0, min(1.0, float(RERANK_TOP_FRACTION))))))
    if RERANK_CANDIDATE_K > 0:
        rerank_window = min(rerank_window, RERANK_CANDIDATE_K)
    expensive_k = min(len(candidates), rerank_window)
    candidates = candidates[: max(1, min(len(candidates), expensive_k))]

    rerank_started = time.perf_counter()
    pairs = [
        [
            query,
            _scenario_support_text(
                scenario_id=candidate["id"],
                scenario_text=candidate["text"],
                support_passages=list(candidate.get("support_passages") or []),
            ),
        ]
        for candidate in candidates
    ]
    rerank_scores = reranker.compute_score(pairs, batch_size=RERANK_BATCH_SIZE)
    rerank_ms = (time.perf_counter() - rerank_started) * 1000

    results = [
        {
            "id": candidate["id"],
            "name": candidate["id"].replace("-", " ").title(),
            "title": _extract_title(candidate.get("text", ""), candidate["id"]),
            "summary": _summary_paragraph(candidate.get("text", "")),
            "run_command": _extract_run_command(candidate.get("text", ""), candidate["id"]),
            "runnable_name": _extract_run_scenario_id(candidate.get("text", ""), candidate["id"]),
            "score": float(score),
            "calibrated_score": calibrate_rerank_score(score),
            "retrieval_score": candidate["retrieval_score"],
            "faiss_score": candidate["faiss_score"],
            "bm25_score": candidate["bm25_score"],
            "lexical_score": candidate["lexical_score"],
            "intent_score": candidate.get("intent_score", 0.0),
            "support_passages": candidate.get("support_passages", []),
        }
        for candidate, score in zip(candidates, rerank_scores)
    ]
    for row in results:
        base_final_score = score_to_match(
            row["score"],
            row.get("faiss_score", 0.0),
            row.get("bm25_score", 0.0),
            row.get("lexical_score", 0.0),
            row.get("intent_score", 0.0),
        )
        if query_intent_families:
            scenario_families = _scenario_intent_families(row["id"], doc_texts.get(row["id"], ""))
            if row.get("intent_score", 0.0) > 0:
                row["final_score"] = min(1.0, base_final_score + INTENT_MATCH_BOOST)
            elif scenario_families and not (query_intent_families & scenario_families):
                row["final_score"] = max(0.0, base_final_score * INTENT_MISMATCH_PENALTY)
            else:
                row["final_score"] = base_final_score
        else:
            row["final_score"] = base_final_score
    results.sort(key=lambda row: row["final_score"], reverse=True)
    results = _filter_conflicting_intent_results(query, results, doc_texts)

    total_ms = (time.perf_counter() - started) * 1000
    final_results = results[:rerank_k]
    for row in final_results:
        row["timing_ms"] = {
            "retrieve": int(round(retrieve_ms)),
            "rerank": int(round(rerank_ms)),
            "total": int(round(total_ms)),
            "reranked": len(candidates),
        }
    return final_results


_ranker_instance = None
_ranker_config = None


def reset_ranker() -> None:
    global _ranker_instance, _ranker_config
    _ranker_instance = None
    _ranker_config = None


def create_ranker(
    device_preference: str = "auto",
    cpu_only: bool = False,
    backend: str = DEFAULT_BACKEND,
    llama_model_path: str = DEFAULT_LLAMA_MODEL,
    llama_gpu_layers: int = DEFAULT_LLAMA_GPU_LAYERS,
):
    global _ranker_instance, _ranker_config

    if cpu_only:
        raise ValueError("CPU-only retriever mode is disabled; use the Vulkan backend")

    resolved_backend = resolve_backend(backend, llama_model_path)
    config = (
        resolved_backend,
        device_preference,
        cpu_only,
        llama_model_path,
        int(llama_gpu_layers),
    )
    if _ranker_instance is not None and _ranker_config == config:
        return _ranker_instance

    _ranker_instance = VulkanRanker(
        model_path=llama_model_path,
        gpu_layers=int(llama_gpu_layers),
    )
    _ranker_config = config
    return _ranker_instance
