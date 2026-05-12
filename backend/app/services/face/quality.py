from __future__ import annotations

import numpy as np

from app.services.embeddings.interface import NoFaceDetectedError
from app.services.face.detector import DetectedFace


def _bbox_size(face: DetectedFace) -> tuple[float, float] | None:
    if face.bbox is None or len(face.bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in face.bbox[:4]]
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def _face_crop(image: np.ndarray, face: DetectedFace) -> np.ndarray | None:
    if face.bbox is None or len(face.bbox) < 4:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in face.bbox[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _blur_variance(image: np.ndarray) -> float:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency guard
        raise NoFaceDetectedError("OpenCV is required for blur quality checks") from exc

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def validate_face_quality(
    face: DetectedFace,
    min_score: float,
    *,
    image: np.ndarray | None = None,
    min_face_size_px: int = 0,
    min_face_area_ratio: float = 0.0,
    min_blur_variance: float = 0.0,
) -> None:
    if face.det_score < min_score:
        raise NoFaceDetectedError("Face detection score below threshold")

    bbox_size = _bbox_size(face)
    if min_face_size_px > 0:
        if bbox_size is None:
            raise NoFaceDetectedError("Face bounding box is missing")
        box_width, box_height = bbox_size
        if box_width < min_face_size_px or box_height < min_face_size_px:
            raise NoFaceDetectedError("Face crop is smaller than the configured threshold")

    if image is None:
        return

    crop = _face_crop(image, face)
    if crop is None:
        raise NoFaceDetectedError("Face crop is empty")

    if min_face_area_ratio > 0.0:
        face_area = float(crop.shape[0] * crop.shape[1])
        image_area = float(image.shape[0] * image.shape[1])
        if image_area <= 0.0 or face_area / image_area < min_face_area_ratio:
            raise NoFaceDetectedError("Face area is below the configured threshold")

    if min_blur_variance > 0.0 and _blur_variance(crop) < min_blur_variance:
        raise NoFaceDetectedError("Face crop is too blurry")
