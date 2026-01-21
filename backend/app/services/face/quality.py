from app.services.embeddings.interface import NoFaceDetectedError
from app.services.face.detector import DetectedFace


def validate_face_quality(face: DetectedFace, min_score: float) -> None:
    if face.det_score < min_score:
        raise NoFaceDetectedError("Face detection score below threshold")
