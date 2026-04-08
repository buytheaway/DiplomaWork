"""Orchestrates the FAISS vector index lifecycle.

Responsibilities:
* Create / load / save / rebuild the index.
* Expose ``add_embedding``, ``search``, ``stats``, ``count``.
* Persist snapshots to disk and record metadata in DB.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import BASE_DIR, Settings
from app.security.crypto import decrypt_embedding_payload, decrypt_snapshot_payload, encrypt_snapshot_payload
from app.services.index.faiss_index import FaissIndex
from app.services.storage.repositories import EmbeddingRepo, IndexSnapshotRepo


@dataclass(frozen=True)
class IndexMatch:
    embedding_id: str
    score: float
    distance: float


class IndexManager:
    """High‑level wrapper around :class:`FaissIndex`."""

    def __init__(
        self,
        settings: Settings,
        model_name: str | None = None,
        index_path_override: str | None = None,
    ) -> None:
        self.settings = settings
        self.dim = settings.embedding_dim
        raw_index_path = Path(index_path_override or settings.index_path)
        if not raw_index_path.is_absolute():
            raw_index_path = BASE_DIR / raw_index_path
        self.index_path = raw_index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index = self._create_index(
            settings.index_type, self.default_params_for(settings.index_type)
        )
        self._index_type = settings.index_type
        self._params = self.default_params_for(settings.index_type)
        self.model_name = model_name
        self.last_snapshot_id: str | None = None
        self._logger = logging.getLogger(__name__)

    # ── helpers ───────────────────────────────────────────────────────────

    def default_params_for(self, index_type: str) -> dict[str, Any]:
        if index_type == "hnsw":
            return {
                "m": self.settings.hnsw_m,
                "ef_construction": self.settings.hnsw_ef_construction,
                "ef_search": self.settings.hnsw_ef_search,
            }
        if index_type == "ivfpq":
            return {
                "nlist": self.settings.ivfpq_nlist,
                "m": self.settings.ivfpq_m,
                "nbits": self.settings.ivfpq_nbits,
            }
        return {}

    def _create_index(self, index_type: str, params: dict[str, Any]) -> FaissIndex:
        return FaissIndex(
            dim=self.dim,
            index_type=index_type,
            params=params,
            seed=self.settings.seed,
        )

    def _map_path_for(self, path: Path) -> Path:
        return path.with_suffix(path.suffix + ".map.json")

    def _save_index_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / self.index_path.name
            raw_map_path = self._map_path_for(raw_path)
            target_map_path = self._map_path_for(self.index_path)

            self._index.save(str(raw_path))
            self.index_path.write_bytes(encrypt_snapshot_payload(raw_path.read_bytes()))
            target_map_path.write_bytes(encrypt_snapshot_payload(raw_map_path.read_bytes()))

    def _load_index_files(self, path: Path) -> None:
        map_path = self._map_path_for(path)
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / path.name
            raw_map_path = self._map_path_for(raw_path)
            raw_path.write_bytes(decrypt_snapshot_payload(path.read_bytes()))
            raw_map_path.write_bytes(decrypt_snapshot_payload(map_path.read_bytes()))
            self._index.load(str(raw_path))

    # ── snapshot management ──────────────────────────────────────────────

    def load_latest_snapshot(self, db: Session) -> bool:
        repo = IndexSnapshotRepo(db)
        snapshot = repo.get_latest_for_path(str(self.index_path))
        if snapshot and Path(snapshot.path).exists():
            self._index = self._create_index(snapshot.index_type, snapshot.params)
            self._load_index_files(Path(snapshot.path))
            self._index_type = snapshot.index_type
            self._params = snapshot.params
            self.last_snapshot_id = str(snapshot.id)
            self._logger.info(
                "Loaded index snapshot id=%s type=%s count=%d path=%s",
                snapshot.id,
                snapshot.index_type,
                self._index.count(),
                snapshot.path,
            )
            return True
        self._logger.info("No index snapshot found — starting with empty index")
        return False

    def save_snapshot(self, db: Session) -> None:
        self._save_index_files()
        repo = IndexSnapshotRepo(db)
        snapshot = repo.create(
            index_type=self._index_type,
            params=self._params,
            path=str(self.index_path),
            embeddings_count=self._index.count(),
        )
        self.last_snapshot_id = str(snapshot.id)

    # ── core operations ──────────────────────────────────────────────────

    def add_embedding(self, embedding_id: str, vector: np.ndarray) -> int:
        return self._index.add_embedding(embedding_id, vector)

    def search(self, vector: np.ndarray, k: int) -> list[IndexMatch]:
        results = self._index.search(vector, k)
        return [IndexMatch(r.embedding_id, r.score, r.distance) for r in results]

    def rebuild(self, db: Session, index_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if not params:
            params = self.default_params_for(index_type)
        embeddings = EmbeddingRepo(db).get_active_embeddings(model=self.model_name)
        self._index = self._create_index(index_type, params)
        self._index_type = index_type
        self._params = params

        if embeddings:
            vectors = np.vstack(
                [
                    np.frombuffer(decrypt_embedding_payload(e.vector), dtype=np.float32)
                    for e in embeddings
                ]
            )
            self._index.train(vectors)
            for embedding, vector in zip(embeddings, vectors):
                self._index.add_embedding(str(embedding.id), vector)

        self.save_snapshot(db)
        return self.stats()

    def rebuild_current(self, db: Session) -> dict[str, Any]:
        return self.rebuild(db, self._index_type, dict(self._params))

    def stats(self) -> dict[str, Any]:
        stats = self._index.stats()
        stats["loaded"] = self._index.count() > 0
        stats["file_path"] = str(self.index_path)
        stats["last_snapshot_id"] = self.last_snapshot_id
        stats["model_name"] = self.model_name
        return stats

    def count(self) -> int:
        return self._index.count()
