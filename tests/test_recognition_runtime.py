"""Tests for runtime error paths and provider selection helpers."""

from __future__ import annotations

import pytest

from src.config.models import RecognitionConfig
from src.recognition import InsightFaceRuntime, RecognitionError
from src.recognition.providers import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    ctx_id_for,
    resolve_providers,
)


def test_missing_models_raise_actionable_error(tmp_path):
    runtime = InsightFaceRuntime(
        RecognitionConfig(model_dir=str(tmp_path / "missing"), device="cpu")
    )

    with pytest.raises(RecognitionError, match="download_models.py") as error:
        _ = runtime.providers

    assert "missing" in str(error.value)


def test_cpu_provider_selection():
    assert resolve_providers("cpu") == [CPU_PROVIDER]


def test_cpu_context_id():
    assert ctx_id_for("cpu", [CPU_PROVIDER]) == -1


def test_cuda_context_ids():
    providers = [CUDA_PROVIDER, CPU_PROVIDER]
    assert ctx_id_for("cuda", providers) == 0
    assert ctx_id_for("cuda:1", providers) == 1
    assert ctx_id_for("auto", providers) == 0
