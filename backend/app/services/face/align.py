from __future__ import annotations

import numpy as np

from app.services.embeddings.interface import InvalidImageError


class AlignmentError(Exception):
    pass


def align_with_landmarks(
    image: np.ndarray,
    landmarks: np.ndarray | None,
    allow_center_crop: bool,
    bbox: np.ndarray | None = None,
) -> np.ndarray:
    if landmarks is not None and len(landmarks) >= 5:
        try:
            from insightface.utils import face_align
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise AlignmentError("insightface is required for landmark alignment") from exc

        aligned = face_align.norm_crop(image, landmarks)
        return aligned

    if not allow_center_crop:
        raise AlignmentError("Landmarks missing; center crop disabled")

    if bbox is not None and bbox.shape[0] >= 4:
        x1, y1, x2, y2 = bbox.astype(int)
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, image.shape[1])
        y2 = min(y2, image.shape[0])
        cropped = image[y1:y2, x1:x2]
    else:
        h, w, _ = image.shape
        size = min(h, w)
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        cropped = image[start_y : start_y + size, start_x : start_x + size]
    if cropped.size == 0:
        raise InvalidImageError("Invalid image for center crop")

    import cv2

    return cv2.resize(cropped, (112, 112))
