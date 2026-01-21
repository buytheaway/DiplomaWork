from dataclasses import dataclass

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.services.embeddings.interface import InvalidImageError


@dataclass
class DetectedFace:
    bbox: np.ndarray
    kps: np.ndarray | None
    det_score: float
    embedding: np.ndarray | None
    normed_embedding: np.ndarray | None


class InsightFaceDetector:
    def __init__(self, model_name: str, det_size: tuple[int, int] = (640, 640)) -> None:
        self.app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        faces = self.app.get(image)
        results: list[DetectedFace] = []
        for face in faces:
            results.append(
                DetectedFace(
                    bbox=face.bbox,
                    kps=getattr(face, "kps", None),
                    det_score=float(face.det_score),
                    embedding=getattr(face, "embedding", None),
                    normed_embedding=getattr(face, "normed_embedding", None),
                )
            )
        return results


def decode_image(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("Invalid or unreadable image")
    return img
