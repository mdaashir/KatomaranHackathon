# syntax=docker/dockerfile:1.7
ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim-bookworm AS python-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

FROM python-base AS builder-base
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
RUN uv venv /opt/venv --python 3.11
ENV VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

FROM builder-base AS builder-cpu
RUN uv sync --frozen --extra cpu --no-dev --no-install-project \
    && (uv pip uninstall opencv-python || true) \
    && uv pip install --no-deps --force-reinstall "opencv-python-headless>=4.9"

FROM builder-base AS builder-gpu
RUN uv sync --frozen --extra gpu --no-dev --no-install-project \
    && uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu128 torch torchvision \
    && (uv pip uninstall onnxruntime opencv-python || true) \
    && uv pip install --no-deps --force-reinstall "opencv-python-headless>=4.9" \
    && uv pip install --no-deps --force-reinstall "onnxruntime-gpu==1.22.0"

FROM python-base AS cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder-cpu /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    CONFIG_PATH=/app/config.json
COPY src/ ./src/
COPY main.py config.json ./
COPY scripts/download_models.py ./scripts/
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p data logs models \
    && chown -R 10001:10001 /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import pathlib, sys; sys.exit(0 if pathlib.Path('models/yolov8n-face.pt').is_file() else 1)"
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sh", "-c", "exec python main.py --config \"${CONFIG_PATH:-config.json}\""]

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS gpu
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        libffi8 \
        libcrypt1 \
        libexpat1 \
        zlib1g \
        libsqlite3-0 \
        libreadline8 \
        libbz2-1.0 \
        liblzma5 \
        libncursesw6 \
        libuuid1 \
        libmpdec3 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=python-base /usr/local /usr/local
WORKDIR /app
COPY --from=builder-gpu /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    CONFIG_PATH=/app/config.json \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/opt/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cudnn/lib
COPY src/ ./src/
COPY main.py config.json ./
COPY scripts/download_models.py ./scripts/
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p data logs models \
    && chown -R 10001:10001 /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import pathlib, sys; sys.exit(0 if pathlib.Path('models/yolov8n-face.pt').is_file() else 1)"
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sh", "-c", "exec python main.py --config \"${CONFIG_PATH:-config.json}\""]
