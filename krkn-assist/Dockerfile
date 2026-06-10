FROM --platform=linux/arm64 fedora:42

USER 0

# ── [1/4] Mesa libkrun Vulkan drivers (locked before anything else) ──────────
#
# Must install before any other mesa package lands via transitive deps, or dnf
# will pull the stock mesa build and the versionlock will conflict.
RUN dnf -y install \
        dnf-plugins-core \
        dnf-plugin-versionlock && \
    dnf -y copr enable slp/mesa-libkrun-vulkan && \
    dnf -y install \
        mesa-vulkan-drivers.aarch64 \
        vulkan-loader-devel.aarch64 \
        vulkan-headers \
        vulkan-tools.aarch64 \
        vulkan-loader.aarch64 \
        glslc && \
    dnf versionlock add mesa-vulkan-drivers && \
    dnf clean all

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

ARG GGML_BACKEND=vulkan
ENV GGML_BACKEND_BUILT=${GGML_BACKEND}

# ── [2/4] Python + build tools ───────────────────────────────────────────────
RUN dnf -y install --setopt=install_weak_deps=False \
    git \
    python3 python3-pip python3-devel \
    gcc gcc-c++ cmake ninja-build make pkgconf-pkg-config \
    spirv-headers-devel \
    ccache && \
    dnf clean all

# ── [3/4] llama-cpp-python with Vulkan backend ───────────────────────────────
#
# GGML_VULKAN_COOPMAT* disabled: cooperative matrix extensions require
# Vulkan 1.3 features not guaranteed on Mesa Venus (virtio-gpu).
RUN --mount=type=cache,target=/root/.cache/ccache \
    pip install --upgrade pip setuptools wheel && \
    export PATH="/usr/lib64/ccache:$PATH" && \
    CMAKE_ARGS="\
      -DGGML_VULKAN=on \
      -DGGML_VULKAN_COOPMAT=OFF \
      -DGGML_VULKAN_COOPMAT2=OFF \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    " FORCE_CMAKE=1 \
      pip install --no-binary=llama-cpp-python "llama-cpp-python==0.3.19"

# ── [4/4] Python runtime dependencies ────────────────────────────────────────
#
# torch: PyTorch CPU wheel used only for ONNX export when a cached reranker is not
#        available. Embeddings stay on llama.cpp/Vulkan.
#
# onnxruntime: used by OnnxCrossEncoder for cross-encoder reranking.
#              No GPU dependency required for 20-document workloads.
#
# optimum[onnxruntime]: preferred ONNX export path from HuggingFace models.
#
# FlagEmbedding intentionally removed: FlagReranker replaced by OnnxCrossEncoder.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install \
      fastapi \
      uvicorn \
      "transformers>=4.52.4,<5" \
      "huggingface_hub[cli]>=0.30,<1" \
      faiss-cpu \
      onnx \
      onnxscript \
      onnxruntime \
      "optimum[onnxruntime]>=1.27,<2" \
      requests

ARG LLAMA_EMBED_MODEL=/models/Qwen3-Embedding-0.6B-f16.gguf
ARG GGUF_REPO=Qwen/Qwen3-Embedding-0.6B-GGUF
ARG GGUF_FILE=Qwen3-Embedding-0.6B-f16.gguf

ENV RETRIEVER_BACKEND=vulkan \
    LLAMA_EMBED_MODEL=${LLAMA_EMBED_MODEL} \
    LLAMA_GPU_LAYERS=-1

# Download GGUF model from HuggingFace
RUN mkdir -p /models && \
    curl -L --fail --retry 3 --continue-at - \
    -o /models/${GGUF_FILE} \
    "https://huggingface.co/${GGUF_REPO}/resolve/main/${GGUF_FILE}"

COPY krkn-assist/ /app/

EXPOSE 8080 18080
CMD ["python3", "api_server.py"]
