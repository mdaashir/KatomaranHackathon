"""Recognition layer: quality gate, alignment, embeddings and registry."""

from src.recognition.alignment import (
    ARCFACE_DST,
    align_face,
    five_points_from_106,
    pose_is_frontal,
)
from src.recognition.embedder import ArcFaceEmbedder, FaceRuntime
from src.recognition.providers import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    ctx_id_for,
    prepare_cuda_dlls,
    resolve_providers,
)
from src.recognition.quality import QualityGate, crop_face
from src.recognition.registry import FaceRegistry, RegistryError
from src.recognition.runtime import RecognitionError, InsightFaceRuntime
from src.recognition.types import EmbeddingResult, MatchResult, QualityResult

__all__ = [
    "ARCFACE_DST",
    "ArcFaceEmbedder",
    "CPU_PROVIDER",
    "CUDA_PROVIDER",
    "EmbeddingResult",
    "FaceRegistry",
    "FaceRuntime",
    "InsightFaceRuntime",
    "MatchResult",
    "QualityGate",
    "QualityResult",
    "RecognitionError",
    "RegistryError",
    "align_face",
    "crop_face",
    "ctx_id_for",
    "five_points_from_106",
    "pose_is_frontal",
    "prepare_cuda_dlls",
    "resolve_providers",
]
