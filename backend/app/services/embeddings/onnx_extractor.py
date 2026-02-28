"""ONNX‑based embedding extractor — **skeleton / example**.

This file demonstrates how to plug a real ONNX face‑recognition model into the
system.  No model weights are shipped; set ``ONNX_MODEL_PATH`` to a valid
``.onnx`` file and adjust pre‑processing to your model's needs.

Usage::

    EMBEDDING_BACKEND=onnx
    ONNX_MODEL_PATH=path/to/model.onnx
    EMBEDDING_DIM=512
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.services.embeddings.interface import (
    EmbeddingExtractor,
    InvalidImageError,
    NoFaceDetectedError,
)

logger = logging.getLogger(__name__)


class OnnxEmbeddingExtractor(EmbeddingExtractor):
    """Skeleton ONNX extractor — replace ``_preprocess`` with your pipeline."""

    def __init__(self, settings: Settings) -> None:
        try:
            import onnxruntime as ort  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ONNX embedding backend. "
                "Install it with: pip install onnxruntime"
            ) from exc

        model_path = Path(settings.onnx_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.model_name = f"onnx_{model_path.stem}"
        self.dim = settings.embedding_dim
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info("ONNX model loaded: %s  dim=%d", model_path, self.dim)

    # ------------------------------------------------------------------
    # Override this method with your model's specific pre‑processing.
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(image_bytes: bytes) -> np.ndarray:
        """Decode + resize + normalise image to ``(1, 3, 112, 112)`` float32.

        This is a **placeholder** — adjust to your ONNX model's input spec.
        """
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python-headless is required for image decoding"
            ) from exc

        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise InvalidImageError("Cannot decode image bytes")

        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img / 255.0 - 0.5) / 0.5  # normalise to [-1, 1]
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # NCHW
        return img

    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise InvalidImageError("Empty image bytes")

        tensor = self._preprocess(image_bytes)
        outputs = self._session.run(None, {self._input_name: tensor})

        vector = outputs[0].flatten().astype(np.float32)
        if vector.shape[0] != self.dim:
            raise NoFaceDetectedError(
                f"Model returned dim={vector.shape[0]}, expected {self.dim}"
            )

        # L2‑normalise
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector
