from __future__ import annotations

import numpy as np
import pytest

from app.services.embeddings.interface import NoFaceDetectedError
from app.services.face.detector import DetectedFace
from app.services.face.quality import validate_face_quality


def _face(
    *,
    score: float = 0.9,
    bbox: tuple[float, float, float, float] = (10.0, 10.0, 70.0, 70.0),
) -> DetectedFace:
    return DetectedFace(
        bbox=np.asarray(bbox, dtype=np.float32),
        kps=None,
        det_score=score,
        embedding=None,
        normed_embedding=None,
    )


def test_validate_face_quality_accepts_valid_face():
    image = np.full((100, 100, 3), 127, dtype=np.uint8)

    validate_face_quality(
        _face(),
        min_score=0.5,
        image=image,
        min_face_size_px=40,
        min_face_area_ratio=0.01,
    )


def test_validate_face_quality_rejects_low_detection_score():
    with pytest.raises(NoFaceDetectedError, match="score"):
        validate_face_quality(_face(score=0.2), min_score=0.5)


def test_validate_face_quality_rejects_tiny_face_box():
    with pytest.raises(NoFaceDetectedError, match="smaller"):
        validate_face_quality(
            _face(bbox=(10.0, 10.0, 20.0, 20.0)),
            min_score=0.5,
            min_face_size_px=40,
        )


def test_validate_face_quality_rejects_small_face_area_ratio():
    image = np.full((200, 200, 3), 127, dtype=np.uint8)

    with pytest.raises(NoFaceDetectedError, match="area"):
        validate_face_quality(
            _face(bbox=(10.0, 10.0, 30.0, 30.0)),
            min_score=0.5,
            image=image,
            min_face_area_ratio=0.05,
        )
