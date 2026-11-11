"""Tests for the cosine face registry."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from src.config.models import RecognitionConfig
from src.recognition.registry import FaceRegistry, RegistryError
from src.storage import Database, FaceRepository

NOW = datetime(2025, 6, 20, 18, 39, 3, tzinfo=timezone.utc)


def _vector(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=512).astype(np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def faces(tmp_path):
    database = Database(tmp_path / "faces.db")
    database.initialize()
    repository = FaceRepository(database)
    yield repository
    database.close()


def _config(**overrides) -> RecognitionConfig:
    values = {
        "model_name": "buffalo_l",
        "model_dir": "unused",
        "similarity_threshold": 0.45,
        "max_embeddings_per_face": 2,
    }
    values.update(overrides)
    return RecognitionConfig(**values)


def test_empty_registry(faces):
    registry = FaceRegistry(faces, _config())

    assert registry.size == 0
    match = registry.match(_vector(1))
    assert match.face_id is None
    assert match.similarity == -1.0


def test_register_then_match_same_vector(faces):
    registry = FaceRegistry(faces, _config())
    vector = _vector(1)

    face_id = registry.register(vector, quality=0.8, now=NOW)
    match = registry.match(vector)

    assert match.face_id == face_id
    assert match.similarity == pytest.approx(1.0, abs=1e-5)
    assert faces.get(face_id).model_name == "buffalo_l"


def test_different_person_does_not_match(faces):
    registry = FaceRegistry(faces, _config())
    face_id = registry.register(_vector(1), quality=0.8, now=NOW)

    match = registry.match(_vector(2))

    assert match.face_id != face_id
    assert match.face_id is None
    assert match.similarity < 0.45


def test_vectors_are_normalised_before_storage(faces):
    registry = FaceRegistry(faces, _config())
    vector = _vector(3)

    face_id = registry.register(vector * 7.0, quality=0.8, now=NOW)
    match = registry.match(vector)

    assert match.face_id == face_id
    assert match.similarity == pytest.approx(1.0, abs=1e-5)


def test_registry_reloads_from_sqlite(faces):
    first = FaceRegistry(faces, _config())
    face_id = first.register(_vector(4), quality=0.8, now=NOW)

    second = FaceRegistry(faces, _config())
    match = second.match(_vector(4))

    assert second.size == 1
    assert match.face_id == face_id


def test_per_face_embedding_cap(faces):
    registry = FaceRegistry(faces, _config(max_embeddings_per_face=2))
    face_id = registry.register(_vector(5), quality=0.8, now=NOW)

    assert registry.add_embedding(face_id, _vector(6), 0.9, NOW) is True
    assert registry.add_embedding(face_id, _vector(7), 0.9, NOW) is False
    assert faces.count_embeddings(face_id) == 2
    assert registry.size == 2


def test_added_embedding_is_matchable(faces):
    registry = FaceRegistry(faces, _config())
    face_id = registry.register(_vector(8), quality=0.8, now=NOW)
    alternate = _vector(9)

    assert registry.add_embedding(face_id, alternate, 0.9, NOW) is True
    match = registry.match(alternate)

    assert match.face_id == face_id
    assert match.similarity == pytest.approx(1.0, abs=1e-5)


def test_threshold_override(faces):
    registry = FaceRegistry(faces, _config(similarity_threshold=0.99))
    face_id = registry.register(_vector(10), quality=0.8, now=NOW)

    strict = registry.match(_vector(10))
    relaxed = registry.match(_vector(10), threshold=0.1)

    assert strict.face_id == face_id  # identical vector still clears 0.99
    assert relaxed.face_id == face_id


def test_rejects_wrong_dimension(faces):
    registry = FaceRegistry(faces, _config())
    with pytest.raises(RegistryError, match="512"):
        registry.match(np.zeros(128, dtype=np.float32))


def test_rejects_zero_vector(faces):
    registry = FaceRegistry(faces, _config())
    with pytest.raises(RegistryError, match="zero"):
        registry.register(np.zeros(512, dtype=np.float32), 0.8, NOW)
