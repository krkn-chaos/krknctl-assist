#!/usr/bin/env python3
"""
Pre-build FAISS index during Docker image build.

This script generates the FAISS index, metadata, and document cache
at build time to eliminate startup indexing delay.

Usage:
    python3 prebuild_index.py
"""
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

def main():
    # Work from script directory (supports both Docker and GitHub Actions)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    from assist.ranking import create_ranker
    from assist.settings import (
        DOCS_DIR,
        INDEX_DIR,
        INDEX_PATH,
        META_PATH,
        DOCS_CACHE_PATH,
    )

    logger.info("Creating ranker for index pre-build...")
    logger.info("  Backend: torch (CPU-only for build-time indexing)")
    logger.info("  Device: cpu")

    os.makedirs(INDEX_DIR, exist_ok=True)

    # Use torch backend with CPU for build-time indexing
    # Vulkan/GPU backends are not available during Docker build
    ranker = create_ranker(
        device_preference="cpu",
        cpu_only=True,
        backend="torch",
        llama_model_path="",
        llama_gpu_layers=0,
    )

    logger.info("Building FAISS index from %s...", DOCS_DIR)
    ranker.build_index(DOCS_DIR)

    if not os.path.exists(INDEX_PATH):
        logger.error("Index file not created at %s", INDEX_PATH)
        sys.exit(1)
    if not os.path.exists(META_PATH):
        logger.error("Meta file not created at %s", META_PATH)
        sys.exit(1)
    if not os.path.exists(DOCS_CACHE_PATH):
        logger.error("Docs cache not created at %s", DOCS_CACHE_PATH)
        sys.exit(1)

    index_size_mb = os.path.getsize(INDEX_PATH) / (1024 * 1024)
    meta_size_mb = os.path.getsize(META_PATH) / (1024 * 1024)
    docs_size_mb = os.path.getsize(DOCS_CACHE_PATH) / (1024 * 1024)
    total_size_mb = index_size_mb + meta_size_mb + docs_size_mb

    logger.info("Index pre-build completed successfully:")
    logger.info("  %s: %.2f MB", INDEX_PATH, index_size_mb)
    logger.info("  %s: %.2f MB", META_PATH, meta_size_mb)
    logger.info("  %s: %.2f MB", DOCS_CACHE_PATH, docs_size_mb)
    logger.info("  Total: %.2f MB", total_size_mb)

    doc_count = len(ranker.doc_texts) if hasattr(ranker, 'doc_texts') else 0
    entry_count = len(ranker.index_entries) if hasattr(ranker, 'index_entries') else 0
    logger.info("  Documents indexed: %d", doc_count)
    logger.info("  Index entries: %d", entry_count)

if __name__ == "__main__":
    main()
