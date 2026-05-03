"""FAISS adapter implementing :class:`VectorIndex`.

Supports three index types:
* ``flat``  — :class:`~faiss.IndexFlatIP` (brute‑force, exact).
* ``hnsw``  — :class:`~faiss.IndexHNSWFlat` (approximate, fast).
* ``ivfpq`` — :class:`~faiss.IndexIVFPQ` (approximate, compact; requires training).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.services.index.interface import VectorIndex, VectorSearchResult


class FaissIndex(VectorIndex):
    def __init__(self, dim: int, index_type: str, params: dict[str, Any], seed: int) -> None:
        self.dim = dim
        self.index_type = index_type
        self.params = params
        self.seed = seed
        self._index = self._build_index()
        self._id_map: dict[int, str] = {}
        self._next_id = 0
        self._lock = threading.Lock()

    def _build_index(self):
        metric = faiss.METRIC_INNER_PRODUCT
        if self.index_type == "flat":
            base = faiss.IndexFlatIP(self.dim)
        elif self.index_type == "hnsw":
            m = int(self.params.get("m", 32))
            base = faiss.IndexHNSWFlat(self.dim, m, metric)
            base.hnsw.efConstruction = int(self.params.get("ef_construction", 200))
            base.hnsw.efSearch = int(self.params.get("ef_search", 64))
        elif self.index_type == "ivfpq":
            nlist = int(self.params.get("nlist", 100))
            m = int(self.params.get("m", 16))
            nbits = int(self.params.get("nbits", 8))
            quantizer = faiss.IndexFlatIP(self.dim)
            base = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, nbits, metric)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        return faiss.IndexIDMap2(base)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        vec = vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def add_embedding(self, embedding_id: str, vector: np.ndarray) -> int:
        vec = self._normalize(vector).reshape(1, -1).astype(np.float32)
        with self._lock:
            vector_id = self._next_id
            self._next_id += 1
            self._id_map[vector_id] = embedding_id
            ids = np.array([vector_id], dtype=np.int64)
            self._index.add_with_ids(vec, ids)
        return vector_id

    def search(self, vector: np.ndarray, k: int) -> list[VectorSearchResult]:
        if self.count() == 0:
            return []
        vec = self._normalize(vector).reshape(1, -1).astype(np.float32)
        with self._lock:
            distances, ids = self._index.search(vec, k)
            id_map_snapshot = dict(self._id_map)
        results: list[VectorSearchResult] = []
        for score, vector_id in zip(distances[0], ids[0], strict=False):
            if vector_id < 0:
                continue
            embedding_id = id_map_snapshot.get(int(vector_id))
            if embedding_id is None:
                continue
            score_val = float(score)
            results.append(
                VectorSearchResult(
                    embedding_id=embedding_id,
                    score=score_val,
                    distance=float(1.0 - score_val),
                )
            )
        return results

    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            faiss.write_index(self._index, str(path_obj))
            map_path = path_obj.with_suffix(path_obj.suffix + ".map.json")
            with map_path.open("w", encoding="utf-8") as handle:
                json.dump(self._id_map, handle)

    def load(self, path: str) -> None:
        path_obj = Path(path)
        map_path = path_obj.with_suffix(path_obj.suffix + ".map.json")
        if not map_path.exists():
            raise FileNotFoundError(
                f"Missing FAISS sidecar map for {path_obj.name}: expected {map_path.name}"
            )

        with self._lock:
            loaded_index = faiss.read_index(str(path_obj))
            with map_path.open("r", encoding="utf-8") as handle:
                loaded_map = {int(k): v for k, v in json.load(handle).items()}
            if int(loaded_index.ntotal) != len(loaded_map):
                raise ValueError(
                    f"FAISS map mismatch for {path_obj.name}: "
                    f"index has {int(loaded_index.ntotal)} vectors, map has {len(loaded_map)} ids"
                )

            self._index = loaded_index
            self._id_map = loaded_map
            self._next_id = max(self._id_map.keys(), default=-1) + 1

    def count(self) -> int:
        return int(self._index.ntotal)

    def stats(self) -> dict[str, Any]:
        memory_bytes = int(self.count() * self.dim * 4)
        is_trained = True
        if self.index_type == "ivfpq":
            try:
                is_trained = bool(self._index.index.is_trained)
            except Exception:
                is_trained = False
        return {
            "index_type": self.index_type,
            "params": self.params,
            "embeddings_count": self.count(),
            "memory_estimate": memory_bytes,
            "memory_estimate_bytes": memory_bytes,
            "loaded": True,
            "is_trained": is_trained,
        }

    def train(self, vectors: np.ndarray) -> None:
        if self.index_type != "ivfpq":
            return
        if vectors.size == 0:
            return
        with self._lock:
            if not self._index.is_trained:
                self._index.train(vectors.astype(np.float32))
