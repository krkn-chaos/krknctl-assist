#!/usr/bin/env python3
"""
Pre-export ONNX reranker model during Docker image build.

This script exports the cross-encoder model to ONNX format at build time,
eliminating the need for torch dependencies at runtime.

Usage:
    python3 prebuild_onnx_reranker.py
"""
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

def main():
    os.chdir("/app")

    from assist.settings import CROSS_ENCODER_MODEL
    from assist.ranking import OnnxCrossEncoder

    logger.info("Pre-exporting ONNX reranker model...")
    logger.info("  Model: %s", CROSS_ENCODER_MODEL)

    try:
        cross_encoder = OnnxCrossEncoder(CROSS_ENCODER_MODEL)

        onnx_dir = cross_encoder._onnx_model_dir()
        onnx_file = onnx_dir / "model.onnx"
        runtime_file = onnx_dir / "model-int8.onnx"

        if not onnx_file.exists():
            logger.error("ONNX export failed - model.onnx not found at %s", onnx_file)
            sys.exit(1)

        onnx_size_mb = onnx_file.stat().st_size / (1024 * 1024)
        logger.info("  ONNX model exported: %s (%.2f MB)", onnx_file, onnx_size_mb)

        if runtime_file.exists():
            runtime_size_mb = runtime_file.stat().st_size / (1024 * 1024)
            logger.info("  Quantized model: %s (%.2f MB)", runtime_file, runtime_size_mb)

        logger.info("ONNX reranker pre-export completed successfully")
        logger.info("  Cache directory: %s", onnx_dir)

    except Exception as exc:
        logger.error("Failed to pre-export ONNX reranker: %s", exc)
        sys.exit(1)

if __name__ == "__main__":
    main()
