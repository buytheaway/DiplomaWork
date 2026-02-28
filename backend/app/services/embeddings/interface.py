"""Embedding extractor interface (port) and domain errors.

Every concrete extractor *must* implement :class:`EmbeddingExtractor`.
Backend code depends **only** on this module — never on a specific ML library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# ── domain errors ────────────────────────────────────────────────────────────


class FaceProcessingError(Exception):
    """Base error for face‑processing failures."""


class InvalidImageError(FaceProcessingError):
    """Image bytes cannot be decoded."""


class NoFaceDetectedError(FaceProcessingError):
    """Zero faces detected in image."""


class MultipleFacesDetectedError(FaceProcessingError):
    """More than one face detected when strict single‑face policy is active."""


# ── abstract extractor ───────────────────────────────────────────────────────


class EmbeddingExtractor(ABC):
    """Port: convert raw image bytes → normalised float32 embedding vector.

    Implementations:
    * ``DummyEmbeddingExtractor`` — always available, no ML deps.
    * ``InsightFaceEmbeddingExtractor`` — requires ``insightface`` + ``onnxruntime``.
    * ``TorchEmbeddingExtractor`` — requires ``torch`` + optional face detector.
    * ``OnnxEmbeddingExtractor`` — skeleton for any ONNX‑based model.
    """

    model_name: str
    dim: int

    @abstractmethod
    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        """Return a **unit‑normalised** ``float32`` vector of shape ``(dim,)``."""
        raise NotImplementedError


# ── dummy (always available) ─────────────────────────────────────────────────


class DummyEmbeddingExtractor(EmbeddingExtractor):
    """Deterministic extractor that needs zero ML dependencies.

    Returns a fixed unit vector (``[1, 0, 0, …]``).  Useful for:
    * Integration tests (``TESTING=true``)
    * Demo / smoke‑test without a real model
    """

    def __init__(self, dim: int = 512) -> None:
        self.model_name = "dummy"
        self.dim = dim

    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise InvalidImageError("Empty image bytes")
        vector = np.zeros((self.dim,), dtype=np.float32)
        vector[0] = 1.0
        return vector


# ── factory ──────────────────────────────────────────────────────────────────


def create_extractor(settings: object) -> EmbeddingExtractor:
    """Instantiate the extractor chosen by ``settings.embedding_backend``.

    Heavy ML libraries are imported **lazily** so that
    ``EMBEDDING_BACKEND=dummy`` never triggers an ``import torch`` / etc.
    """
    backend: str = getattr(settings, "embedding_backend", "dummy")
    dim: int = getattr(settings, "embedding_dim", 512)

    if getattr(settings, "testing", False) or backend == "dummy":
        return DummyEmbeddingExtractor(dim=dim)

    if backend == "torch":
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
