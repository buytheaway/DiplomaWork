"""Embedding extractor interface (port) and face-processing errors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class FaceProcessingError(Exception):
    """Base error for face-processing failures."""


class InvalidImageError(FaceProcessingError):
    """Image bytes cannot be decoded."""


class NoFaceDetectedError(FaceProcessingError):
    """Zero faces detected in image."""


class MultipleFacesDetectedError(FaceProcessingError):
    """More than one face was detected when strict single-face mode is active."""


@dataclass(frozen=True)
class FaceEmbedding:
    """Embedding extracted for a single detected face."""

    embedding: np.ndarray
    detection_score: float | None = None
    bbox: tuple[float, float, float, float] | None = None


class EmbeddingExtractor(ABC):
    """Port: convert raw image bytes into normalized face embeddings."""

    model_name: str
    dim: int

    @abstractmethod
    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        """Return a unit-normalized embedding for exactly one face."""
        raise NotImplementedError

    def extract_embeddings(self, image_bytes: bytes) -> list[FaceEmbedding]:
        """Return embeddings for all valid faces in the image."""
        return [FaceEmbedding(embedding=self.extract_embedding(image_bytes))]


class DummyEmbeddingExtractor(EmbeddingExtractor):
    """Deterministic extractor that needs zero ML dependencies."""

    def __init__(self, dim: int = 512) -> None:
        self.model_name = "dummy"
        self.dim = dim

    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise InvalidImageError("Empty image bytes")
        vector = np.zeros((self.dim,), dtype=np.float32)
        vector[0] = 1.0
        return vector


def create_extractor(settings: object) -> EmbeddingExtractor:
    """Instantiate the extractor chosen by ``settings.embedding_backend``."""

    backend: str = getattr(settings, "embedding_backend", "dummy")
    dim: int = getattr(settings, "embedding_dim", 512)

    if getattr(settings, "testing", False) or backend == "dummy":
        return DummyEmbeddingExtractor(dim=dim)

    if backend == "torch":
        import logging

        logging.getLogger(__name__).warning(
            "torch backend is experimental; use it as the custom model path, "
            "not as a production baseline"
        )
        from app.services.embeddings.torch_extractor import TorchEmbeddingExtractor

        return TorchEmbeddingExtractor(settings)

    if backend == "insightface":
        from app.services.embeddings.insightface_extractor import (
            InsightFaceEmbeddingExtractor,
        )

        return InsightFaceEmbeddingExtractor(settings)

    if backend == "onnx":
        from app.services.embeddings.onnx_extractor import OnnxEmbeddingExtractor

        return OnnxEmbeddingExtractor(settings)

    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r}")
