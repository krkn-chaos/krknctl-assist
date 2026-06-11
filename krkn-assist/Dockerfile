# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: BUILDER - Build dependencies and compile/download packages
# ═══════════════════════════════════════════════════════════════════════════
FROM --platform=linux/arm64 fedora:42 AS builder

USER 0
WORKDIR /build

# ── [1/5] Mesa libkrun Vulkan drivers + development headers ─────────────────
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

# ── [2/5] Build tools + Python development ───────────────────────────────────
RUN dnf -y install --setopt=install_weak_deps=False \
    git curl \
    python3 python3-pip python3-devel \
    gcc gcc-c++ cmake ninja-build make pkgconf-pkg-config \
    spirv-headers-devel \
    ccache && \
    dnf clean all

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG GGML_BACKEND=vulkan
ENV GGML_BACKEND_BUILT=${GGML_BACKEND}

# ── [3/5] llama-cpp-python with Vulkan backend ───────────────────────────────
ARG LLAMA_CPP_VERSION=0.3.19
ARG GITHUB_REPO=krkn-chaos/krknctl-assist

RUN pip install --upgrade pip setuptools wheel && \
    BASE_URL="https://github.com/${GITHUB_REPO}/releases/download/llama-cpp-${LLAMA_CPP_VERSION}" && \
    echo "Searching for pre-built wheel at ${BASE_URL}..." && \
    WHEEL_FOUND=0 && \
    for PYVER in cp313 cp312 cp311 cp310; do \
      WHEEL_URL="${BASE_URL}/llama_cpp_python-${LLAMA_CPP_VERSION}-${PYVER}-${PYVER}-linux_aarch64.whl"; \
      echo "Checking: ${WHEEL_URL}"; \
      if curl -fsSL --head --max-time 30 "$WHEEL_URL" >/dev/null 2>&1; then \
        echo "Found pre-built wheel (${PYVER}), downloading and installing..."; \
        pip install "$WHEEL_URL" && \
        WHEEL_FOUND=1 && \
        break; \
      else \
        echo "Not found for ${PYVER}"; \
      fi; \
    done && \
    if [ "$WHEEL_FOUND" = "0" ]; then \
      echo "No pre-built wheel found, building from source (this will take ~1.5 hours)..." && \
      export PATH="/usr/lib64/ccache:$PATH" && \
      CMAKE_ARGS="\
        -DGGML_VULKAN=on \
        -DGGML_VULKAN_COOPMAT=OFF \
        -DGGML_VULKAN_COOPMAT2=OFF \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      " FORCE_CMAKE=1 \
        pip install --no-binary=llama-cpp-python "llama-cpp-python==${LLAMA_CPP_VERSION}"; \
    fi

# ── [4/5] Python runtime dependencies ────────────────────────────────────────
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install \
      fastapi \
      uvicorn \
      "transformers>=4.52.4,<5" \
      huggingface_hub \
      faiss-cpu \
      onnx \
      onnxscript \
      onnxruntime \
      "optimum[onnxruntime]>=1.27,<2" \
      requests

# Cleanup Python cache
RUN find /usr/local/lib/python*/site-packages -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python*/site-packages -type f -name '*.pyc' -delete 2>/dev/null || true

# ── [5/5] Download GGUF model ────────────────────────────────────────────────
ARG GGUF_REPO=Qwen/Qwen3-Embedding-0.6B-GGUF
ARG GGUF_FILE=Qwen3-Embedding-0.6B-f16.gguf

RUN mkdir -p /models && \
    curl -L --fail --retry 3 --continue-at - \
    -o /models/${GGUF_FILE} \
    "https://huggingface.co/${GGUF_REPO}/resolve/main/${GGUF_FILE}"

# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: RUNTIME - Minimal runtime image
# ═══════════════════════════════════════════════════════════════════════════
FROM --platform=linux/arm64 fedora:42 AS runtime

USER 0
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG GGML_BACKEND=vulkan
ENV GGML_BACKEND_BUILT=${GGML_BACKEND}

ARG LLAMA_EMBED_MODEL=/models/Qwen3-Embedding-0.6B-f16.gguf
ENV RETRIEVER_BACKEND=vulkan \
    LLAMA_EMBED_MODEL=${LLAMA_EMBED_MODEL} \
    LLAMA_GPU_LAYERS=-1

# ── Install ONLY runtime dependencies ────────────────────────────────────────
RUN dnf -y install \
        dnf-plugins-core \
        dnf-plugin-versionlock && \
    dnf -y copr enable slp/mesa-libkrun-vulkan && \
    dnf -y install --setopt=install_weak_deps=False \
        mesa-vulkan-drivers.aarch64 \
        vulkan-loader.aarch64 \
        vulkan-tools.aarch64 \
        python3 \
        python3-pip \
        curl \
        git && \
    dnf versionlock add mesa-vulkan-drivers && \
    dnf clean all && \
    rm -rf /var/cache/dnf /var/lib/dnf /var/cache/yum && \
    rm -rf /tmp/* /var/tmp/* && \
    rm -rf /usr/share/doc/* /usr/share/man/* /usr/share/info/* && \
    find /usr/share/locale -mindepth 1 -maxdepth 1 ! -name 'en_US' -exec rm -rf {} +

# ── Copy GGUF model and llama-cpp-python wheel from builder ──────────────────
COPY --from=builder /models /models

# ── Install Python dependencies in runtime (ensures correct linkage) ─────────
ARG LLAMA_CPP_VERSION=0.3.19
ARG GITHUB_REPO=krkn-chaos/krknctl-assist

# Copy and install pre-built llama-cpp-python wheel from builder
RUN --mount=type=bind,from=builder,source=/usr/local/lib/python3.13/site-packages,target=/tmp/site-packages \
    cp /tmp/site-packages/llama_cpp_python*.dist-info -r /tmp/ 2>/dev/null || true && \
    pip install --upgrade pip setuptools wheel && \
    BASE_URL="https://github.com/${GITHUB_REPO}/releases/download/llama-cpp-${LLAMA_CPP_VERSION}" && \
    WHEEL_FOUND=0 && \
    for PYVER in cp313 cp312 cp311 cp310; do \
      WHEEL_URL="${BASE_URL}/llama_cpp_python-${LLAMA_CPP_VERSION}-${PYVER}-${PYVER}-linux_aarch64.whl"; \
      if curl -fsSL --head --max-time 30 "$WHEEL_URL" >/dev/null 2>&1; then \
        pip install "$WHEEL_URL" && \
        WHEEL_FOUND=1 && \
        break; \
      fi; \
    done && \
    if [ "$WHEEL_FOUND" = "0" ]; then \
      echo "WARNING: No pre-built wheel found for runtime stage"; \
    fi

# Install remaining Python dependencies
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install \
      fastapi \
      uvicorn \
      "transformers>=4.52.4,<5" \
      huggingface_hub \
      faiss-cpu \
      onnx \
      onnxscript \
      onnxruntime \
      "optimum[onnxruntime]>=1.27,<2" \
      requests && \
    find /usr/local/lib/python*/site-packages -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python*/site-packages -type f -name '*.pyc' -delete 2>/dev/null || true

# ── Copy application code (last to optimize cache) ───────────────────────────
COPY krkn-assist/ /app/

EXPOSE 8080 18080
CMD ["python3", "api_server.py"]
