"""Cosine-similarity face registry backed by SQLite embeddings."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.config.models import RecognitionConfig
from src.recognition.types import MatchResult
from src.storage.repositories import FaceRepository

_EMBEDDING_DIM = 512


class RegistryError(RuntimeError):
    """Raised when the registry receives an invalid embedding."""


def _normalise(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).flatten()
    if vector.size != _EMBEDDING_DIM:
        raise RegistryError(
            f"expected a {_EMBEDDING_DIM}-d embedding, got {vector.size}"
        )
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise RegistryError("cannot register a zero embedding")
    return vector / norm


class FaceRegistry:
    """In-memory cosine matrix kept in sync with the SQLite registry."""

    def __init__(
        self,
        faces: FaceRepository,
        config: RecognitionConfig,
    ) -> None:
        self._faces = faces
        self._config = config
        self._vectors = np.empty((0, _EMBEDDING_DIM), dtype=np.float32)
        self._face_ids = np.empty((0,), dtype=np.int64)
        self.reload()

    @property
    def size(self) -> int:
        """Number of stored embeddings (not unique faces)."""

        return int(self._face_ids.size)

    def reload(self) -> int:
        """Reload every embedding from SQLite and return the row count."""

        vectors: list[np.ndarray] = []
        face_ids: list[int] = []
        for face_id, blob, dim, _quality in self._faces.list_embeddings():
            if dim != _EMBEDDING_DIM:
                continue
            vector = np.frombuffer(blob, dtype=np.float32)
            if vector.size != _EMBEDDING_DIM:
                continue
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                continue
            vectors.append((vector / norm).astype(np.float32))
            face_ids.append(int(face_id))

        if vectors:
            self._vectors = np.vstack(vectors)
            self._face_ids = np.asarray(face_ids, dtype=np.int64)
        else:
            self._vectors = np.empty((0, _EMBEDDING_DIM), dtype=np.float32)
            self._face_ids = np.empty((0,), dtype=np.int64)
        return len(face_ids)

    def match(
        self,
        vector: np.ndarray,
        threshold: float | None = None,
    ) -> MatchResult:
        """Return the best match, or ``face_id=None`` below the threshold."""

        query = _normalise(vector)
        if self._vectors.shape[0] == 0:
            return MatchResult(None, -1.0)

        similarities = self._vectors @ query
        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])
        cutoff = (
            self._config.similarity_threshold
            if threshold is None
            else float(threshold)
        )
        if best_similarity >= cutoff:
            return MatchResult(int(self._face_ids[best_index]), best_similarity)
        return MatchResult(None, best_similarity)

    def register(
        self,
        vector: np.ndarray,
        quality: float,
        now: datetime,
    ) -> int:
        """Create a new Face ID and add its first embedding."""

        normalised = _normalise(vector)
        face_id = self._faces.create(
            vector=normalised.astype(np.float32).tobytes(),
            dim=_EMBEDDING_DIM,
            quality=float(quality),
            model_name=self._config.model_name,
            now=now,
        )
        self._append(face_id, normalised)
        return face_id

    def add_embedding(
        self,
        face_id: int,
        vector: np.ndarray,
        quality: float,
        now: datetime,
    ) -> bool:
        """Add a pose variant when the per-face cap allows it."""

        if (
            self._faces.count_embeddings(face_id)
            >= self._config.max_embeddings_per_face
        ):
            return False

        normalised = _normalise(vector)
        self._faces.add_embedding(
            face_id,
            vector=normalised.astype(np.float32).tobytes(),
            dim=_EMBEDDING_DIM,
            quality=float(quality),
            now=now,
        )
        self._append(face_id, normalised)
        return True

    def _append(self, face_id: int, vector: np.ndarray) -> None:
        self._vectors = np.vstack([self._vectors, vector.reshape(1, -1)])
        self._face_ids = np.append(self._face_ids, np.int64(face_id))
