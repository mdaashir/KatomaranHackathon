"""ONNX Runtime provider selection.

The bundled PyTorch CUDA build ships its own CUDA and cuDNN DLLs. On Windows,
that directory must be on the DLL search path before ONNX Runtime is imported,
otherwise ``CUDAExecutionProvider`` silently falls back to CPU.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.logging_setup import get_logger

logger = get_logger(__name__)

CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


def prepare_cuda_dlls() -> None:
    """Add PyTorch's bundled CUDA/cuDNN directory to the DLL search path."""

    if os.name != "nt":
        return
    try:
        import torch
    except Exception:
        return

    lib_dir = Path(torch.__file__).parent / "lib"
    if lib_dir.is_dir():
        try:
            os.add_dll_directory(str(lib_dir))
        except (AttributeError, OSError):
            logger.debug("could not add torch DLL directory %s", lib_dir)


def resolve_providers(device: str) -> list[str]:
    """Return ONNX Runtime providers for *device*, with CPU fallback.

    ``auto`` prefers CUDA when the provider is available. An explicit ``cuda``
    request that cannot be satisfied logs a warning and uses CPU.
    """

    prepare_cuda_dlls()
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if device == "cpu":
        return [CPU_PROVIDER]
    if CUDA_PROVIDER in available:
        return [CUDA_PROVIDER, CPU_PROVIDER]
    if device.startswith("cuda"):
        logger.warning(
            "CUDA requested for InsightFace but %s is unavailable; using CPU",
            CUDA_PROVIDER,
        )
    return [CPU_PROVIDER]


def ctx_id_for(device: str, providers: list[str]) -> int:
    """Return the ONNX Runtime context id (CUDA index, or -1 for CPU)."""

    if CUDA_PROVIDER not in providers:
        return -1
    if device.startswith("cuda:"):
        suffix = device.split(":", 1)[1]
        return int(suffix) if suffix.isdigit() else 0
    return 0
