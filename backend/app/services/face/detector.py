from dataclasses import dataclass

import cv2
import numpy as np

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
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise ImportError("insightface is not installed") from exc

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


class OpenCVHaarDetector:
    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        results: list[DetectedFace] = []
        for (x, y, w, h) in faces:
            bbox = np.array([x, y, x + w, y + h], dtype=np.float32)
            results.append(
                DetectedFace(
                    bbox=bbox,
                    kps=None,
                    det_score=1.0,
                    embedding=None,
                    normed_embedding=None,
                )
            )
        return results


def decode_image(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("Invalid or unreadable image")
    return img
