import numpy as np

from app.core.config import Settings
from app.services.embeddings.interface import (
    EmbeddingExtractor,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from app.services.face.detector import InsightFaceDetector, decode_image
from app.services.face.quality import validate_face_quality


class InsightFaceEmbeddingExtractor(EmbeddingExtractor):
    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.model_name
        self.dim = 512
        self.strict_single_face = settings.strict_single_face
        self.min_det_score = settings.min_det_score
        self.detector = InsightFaceDetector(model_name=settings.model_name)

    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        image = decode_image(image_bytes)
        faces = self.detector.detect(image)
        if not faces:
            raise NoFaceDetectedError("No face detected")

        if self.strict_single_face and len(faces) != 1:
            raise MultipleFacesDetectedError("Multiple faces detected")

        face = max(faces, key=lambda f: f.det_score)
        validate_face_quality(face, self.min_det_score)

        embedding = face.normed_embedding if face.normed_embedding is not None else face.embedding
        if embedding is None:
            raise NoFaceDetectedError("Face embedding not available")

        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape[0] != self.dim:
            raise NoFaceDetectedError("Unexpected embedding dimension")

        return vector
