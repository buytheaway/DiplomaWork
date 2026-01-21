from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VectorSearchResult:
    embedding_id: str
    score: float
    distance: float


class VectorIndex(ABC):
    @abstractmethod
    def add_embedding(self, embedding_id: str, vector: np.ndarray) -> int:
        raise NotImplementedError

    @abstractmethod
    def search(self, vector: np.ndarray, k: int) -> list[VectorSearchResult]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def train(self, vectors: np.ndarray) -> None:
        raise NotImplementedError
