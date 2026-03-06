"""InsightFace embedding extractor — uses ``buffalo_l`` (or any FaceAnalysis
model) for joint face detection + embedding in one pass.

Required extras::

    pip install insightface onnxruntime opencv-python-headless

Configure via ``.env``::

    EMBEDDING_BACKEND=insightface
    MODEL_NAME=buffalo_l
    EMBEDDING_DIM=512
"""

from __future__ import annotations

import logging

import numpy as np

from app.core.config import Settings
from app.services.embeddings.interface import (
    EmbeddingExtractor,
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)

logger = logging.getLogger(__name__)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes to a BGR ``np.ndarray`` using OpenCV (lazy import)."""
    try:
        import cv2  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "opencv-python-headless is required for image decoding. "
            "Install it with: pip install opencv-python-headless"
        ) from exc
    buf = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("Cannot decode image bytes")
    return img


class InsightFaceEmbeddingExtractor(EmbeddingExtractor):
    """Full-pipeline extractor powered by InsightFace ``FaceAnalysis``.

    * Lazy-imports ``insightface`` so the module never pollutes ``dummy`` mode.
    * Detects faces, enforces strict single-face policy, returns the normed embedding.
    """

    def __init__(self, settings: Settings) -> None:
        try:
            from insightface.app import FaceAnalysis  # noqa: WPS433
        except ImportError as exc:
            raise ImportError(
                "insightface is required for the InsightFace embedding backend. "
                "Install it with: pip install insightface onnxruntime"
            ) from exc

        self.model_name = settings.model_name
        self.dim = settings.embedding_dim
        self.strict_single_face = settings.strict_single_face
        self.min_det_score = settings.min_det_score

        self._app = FaceAnalysis(
            name=settings.model_name,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info(
            "InsightFace model loaded: name=%s dim=%d", self.model_name, self.dim,
        )

    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise InvalidImageError("Empty image bytes")

        image = _decode_image(image_bytes)
        faces = self._app.get(image)

        if not faces:
            raise NoFaceDetectedError("No face detected in image")

        if self.strict_single_face and len(faces) != 1:
            raise MultipleFacesDetectedError(
                f"Expected 1 face, detected {len(faces)}"
            )

        face = max(faces, key=lambda f: float(f.det_score))

        if float(face.det_score) < self.min_det_score:
            raise NoFaceDetectedError(
                f"Detection score {face.det_score:.3f} below threshold {self.min_det_score}"
            )

        # Prefer the pre-normed embedding produced by the recognition model
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        if embedding is None:
            raise NoFaceDetectedError("Model did not produce an embedding")

        vector = np.asarray(embedding, dtype=np.float32).ravel()

        # Safety: L2-normalise if not already unit-length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        if vector.shape[0] != self.dim:
            raise NoFaceDetectedError(
                f"Model returned dim={vector.shape[0]}, expected {self.dim}"
            )

        return vector
