from abc import ABC, abstractmethod

import numpy as np


class FaceProcessingError(Exception):
    pass


class InvalidImageError(FaceProcessingError):
    pass


class NoFaceDetectedError(FaceProcessingError):
    pass


class MultipleFacesDetectedError(FaceProcessingError):
    pass


class EmbeddingExtractor(ABC):
    model_name: str
    dim: int

    @abstractmethod
    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        raise NotImplementedError


class DummyEmbeddingExtractor(EmbeddingExtractor):
    def __init__(self, dim: int = 512) -> None:
        self.model_name = "dummy"
        self.dim = dim

    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        vector = np.zeros((self.dim,), dtype=np.float32)
        vector[0] = 1.0
        return vector
