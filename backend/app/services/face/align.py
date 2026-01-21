from __future__ import annotations

import cv2
import numpy as np
from insightface.utils import face_align

from app.services.embeddings.interface import InvalidImageError


class AlignmentError(Exception):
    pass


def align_with_landmarks(
    image: np.ndarray, landmarks: np.ndarray | None, allow_center_crop: bool
) -> np.ndarray:
    if landmarks is not None and len(landmarks) >= 5:
        aligned = face_align.norm_crop(image, landmarks)
        return aligned

    if not allow_center_crop:
        raise AlignmentError("Landmarks missing; center crop disabled")

    h, w, _ = image.shape
    size = min(h, w)
    start_x = (w - size) // 2
    start_y = (h - size) // 2
    cropped = image[start_y : start_y + size, start_x : start_x + size]
    if cropped.size == 0:
        raise InvalidImageError("Invalid image for center crop")

    return cv2.resize(cropped, (112, 112))
