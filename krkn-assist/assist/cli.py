import argparse
import logging
import os

from .ranking import create_ranker
from .settings import (
    DEFAULT_BACKEND,
    DEFAULT_CPU_ONLY,
    DEFAULT_DEVICE,
    DEFAULT_LLAMA_GPU_LAYERS,
    DEFAULT_LLAMA_MODEL,
    DOCS_DIR,
)


def main() -> None:
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=log_level)
    for logger_name in ("httpx", "huggingface_hub", "transformers", "optimum", "onnxruntime"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description="krkn retriever maintenance CLI")
    parser.add_argument(
        "--device",
        choices=["auto"],
        default=DEFAULT_DEVICE,
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        default=DEFAULT_CPU_ONLY,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "vulkan"],
        default=DEFAULT_BACKEND,
    )
    parser.add_argument("--llama-model", default=DEFAULT_LLAMA_MODEL)
    parser.add_argument("--llama-gpu-layers", type=int, default=DEFAULT_LLAMA_GPU_LAYERS)

    subcommands = parser.add_subparsers(dest="command", required=True)
    index_cmd = subcommands.add_parser("index", help="Build the FAISS index")
    index_cmd.add_argument("--docs", default=DOCS_DIR)

    args = parser.parse_args()
    if args.cpu_only:
        parser.error("--cpu-only is disabled; the retriever always uses Vulkan")

    ranker = create_ranker(
        device_preference=args.device,
        cpu_only=args.cpu_only,
        backend=args.backend,
        llama_model_path=args.llama_model,
        llama_gpu_layers=args.llama_gpu_layers,
    )

    if args.command == "index":
        ranker.build_index(args.docs)
