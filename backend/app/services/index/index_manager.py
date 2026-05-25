"""Orchestrates the FAISS vector index lifecycle.

Responsibilities:
* Create / load / save / rebuild the index.
* Expose ``add_embedding``, ``search``, ``stats``, ``count``.
* Persist snapshots to disk and record metadata in DB.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import BASE_DIR, Settings
from app.security.crypto import (
    decrypt_embedding_payload,
    decrypt_snapshot_payload,
    encrypt_snapshot_payload,
)
from app.services.index.faiss_index import FaissIndex
from app.services.storage.repositories import EmbeddingRepo, IndexSnapshotRepo


@dataclass(frozen=True)
class IndexMatch:
    embedding_id: str
    score: float
    distance: float


class IndexRebuildTooLargeError(RuntimeError):
    """Raised when API rebuild would load too many embeddings in-process."""

    def __init__(
        self,
        *,
        count: int,
        limit: int,
        pipeline: str | None,
        model_name: str | None,
    ) -> None:
        self.count = count
        self.limit = limit
        self.pipeline = pipeline
        self.model_name = model_name
        super().__init__(
            "Index rebuild is too large for the backend API path. "
            f"count={count} limit={limit} pipeline={pipeline} model={model_name}. "
            "Use scripts/build_scale_index_from_db.py for large indexes."
        )


class IndexManager:
    """High‑level wrapper around :class:`FaissIndex`."""

    def __init__(
        self,
        settings: Settings,
        model_name: str | None = None,
        pipeline: str | None = None,
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
        self.pipeline = pipeline
        self.last_snapshot_id: str | None = None
        self.current_snapshot_path: Path | None = None
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
                "nprobe": self.settings.ivfpq_nprobe,
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

    def _next_snapshot_path(self) -> Path:
        suffix = self.index_path.suffix or ".faiss"
        return self.index_path.with_name(
            f"{self.index_path.stem}.{uuid.uuid4().hex}{suffix}"
        )

    def _write_atomic(self, path: Path, payload: bytes) -> None:
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            tmp_path.write_bytes(payload)
            tmp_path.replace(path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _snapshot_files_exist(self, path: Path) -> bool:
        return path.exists() and self._map_path_for(path).exists()

    def _file_sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _snapshot_metadata_for(self, path: Path) -> dict[str, Any]:
        map_path = self._map_path_for(path)
        return {
            "pipeline": self.pipeline,
            "model_name": self.model_name,
            "dim": self.dim,
            "index_sha256": self._file_sha256(path),
            "map_sha256": self._file_sha256(map_path),
        }

    def _snapshot_matches_metadata(self, path: Path, metadata: dict[str, Any]) -> bool:
        expected_pipeline = metadata.get("pipeline")
        expected_model = metadata.get("model_name")
        expected_dim = metadata.get("dim")
        if expected_pipeline is not None and expected_pipeline != self.pipeline:
            return False
        if expected_model is not None and expected_model != self.model_name:
            return False
        if expected_dim is not None and int(expected_dim) != self.dim:
            return False

        expected_index_hash = metadata.get("index_sha256")
        if expected_index_hash and self._file_sha256(path) != expected_index_hash:
            return False

        map_path = self._map_path_for(path)
        expected_map_hash = metadata.get("map_sha256")
        if expected_map_hash and self._file_sha256(map_path) != expected_map_hash:
            return False

        return True

    def _snapshot_path_belongs_to_index(self, path: Path) -> bool:
        return (
            path.parent == self.index_path.parent
            and path.suffix == self.index_path.suffix
            and (
                path == self.index_path
                or path.name.startswith(f"{self.index_path.stem}.")
            )
        )

    def _prune_old_snapshots(self, db: Session) -> None:
        retention = self.settings.index_snapshot_retention
        if retention <= 0:
            return

        repo = IndexSnapshotRepo(db)
        snapshots = repo.list_latest_for_path(str(self.index_path))
        for snapshot in snapshots[retention:]:
            snapshot_path = Path(snapshot.path)
            if snapshot_path == self.current_snapshot_path:
                continue
            if not self._snapshot_path_belongs_to_index(snapshot_path):
                continue

            files_to_delete = [snapshot_path, self._map_path_for(snapshot_path)]
            can_delete_record = True
            deleted_filenames: list[str] = []
            for file_path in files_to_delete:
                try:
                    if file_path.exists():
                        file_path.unlink()
                        deleted_filenames.append(file_path.name)
                except OSError as exc:
                    can_delete_record = False
                    self._logger.warning(
                        "Failed to prune index snapshot file %s: %s",
                        file_path.name,
                        exc,
                    )

            if can_delete_record:
                repo.delete(snapshot)
                if deleted_filenames:
                    self._logger.info(
                        "Pruned index snapshot files: %s",
                        ", ".join(deleted_filenames),
                    )

    def _save_index_files(self) -> Path:
        snapshot_path = self._next_snapshot_path()
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / snapshot_path.name
            raw_map_path = self._map_path_for(raw_path)
            target_map_path = self._map_path_for(snapshot_path)

            self._index.save(str(raw_path))
            self._write_atomic(
                snapshot_path,
                encrypt_snapshot_payload(raw_path.read_bytes()),
            )
            self._write_atomic(
                target_map_path,
                encrypt_snapshot_payload(raw_map_path.read_bytes()),
            )
        return snapshot_path

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
        for snapshot in repo.list_latest_for_path(str(self.index_path)):
            snapshot_path = Path(snapshot.path)
            if not self._snapshot_files_exist(snapshot_path):
                self._logger.warning(
                    "Skipping incomplete index snapshot id=%s path=%s",
                    snapshot.id,
                    snapshot.path,
                )
                continue
            snapshot_params = dict(snapshot.params or {})
            metadata = snapshot_params.pop("_snapshot_meta", None)
            if metadata and not self._snapshot_matches_metadata(snapshot_path, metadata):
                self._logger.warning(
                    "Skipping index snapshot id=%s path=%s because metadata/checksum validation failed",
                    snapshot.id,
                    snapshot.path,
                )
                continue

            params = {
                **self.default_params_for(snapshot.index_type),
                **snapshot_params,
            }
            if snapshot.index_type == "ivfpq":
                params["nprobe"] = self.settings.ivfpq_nprobe
            elif snapshot.index_type == "hnsw":
                params["ef_search"] = self.settings.hnsw_ef_search
            self._index = self._create_index(snapshot.index_type, params)
            try:
                self._load_index_files(snapshot_path)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Skipping unreadable index snapshot id=%s path=%s: %s",
                    snapshot.id,
                    snapshot.path,
                    exc,
                )
                continue
            self._index_type = snapshot.index_type
            self._params = params
            self.last_snapshot_id = str(snapshot.id)
            self.current_snapshot_path = snapshot_path
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
        snapshot_path = self._save_index_files()
        repo = IndexSnapshotRepo(db)
        params = dict(self._params)
        params["_snapshot_meta"] = self._snapshot_metadata_for(snapshot_path)
        snapshot = repo.create(
            index_type=self._index_type,
            params=params,
            path=str(snapshot_path),
            embeddings_count=self._index.count(),
        )
        db.flush()
        self.last_snapshot_id = str(snapshot.id)
        self.current_snapshot_path = snapshot_path
        self._prune_old_snapshots(db)

    # ── core operations ──────────────────────────────────────────────────

    def add_embedding(self, embedding_id: str, vector: np.ndarray) -> int:
        return self._index.add_embedding(embedding_id, vector)

    def search(
        self,
        vector: np.ndarray,
        k: int,
        search_params: dict[str, Any] | None = None,
    ) -> list[IndexMatch]:
        results = self._index.search(vector, k, search_params=search_params)
        return [IndexMatch(r.embedding_id, r.score, r.distance) for r in results]

    def rebuild(self, db: Session, index_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if not params:
            params = self.default_params_for(index_type)
        repo = EmbeddingRepo(db)
        rebuild_count = repo.count_active_embeddings_for_index(
            model=self.model_name,
            pipeline=self.pipeline,
        )
        max_embeddings = self.settings.index_rebuild_max_embeddings
        if max_embeddings > 0 and rebuild_count > max_embeddings:
            raise IndexRebuildTooLargeError(
                count=rebuild_count,
                limit=max_embeddings,
                pipeline=self.pipeline,
                model_name=self.model_name,
            )

        embeddings = repo.get_active_embeddings(
            model=self.model_name,
            pipeline=self.pipeline,
        )
        next_index = self._create_index(index_type, params)

        if embeddings:
            valid_pairs = []
            for embedding in embeddings:
                try:
                    vector = np.frombuffer(
                        decrypt_embedding_payload(embedding.vector),
                        dtype=np.float32,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._logger.warning(
                        "Skipped unreadable embedding id=%s model=%s pipeline=%s: %s",
                        embedding.id,
                        self.model_name,
                        self.pipeline,
                        exc.__class__.__name__,
                    )
                    continue
                if vector.shape[0] != self.dim:
                    continue
                valid_pairs.append((embedding, vector))

            skipped = len(embeddings) - len(valid_pairs)
            if skipped:
                self._logger.warning(
                    "Skipped %d unreadable or malformed embeddings while rebuilding model=%s pipeline=%s",
                    skipped,
                    self.model_name,
                    self.pipeline,
                )
            if valid_pairs:
                vectors = np.vstack([vector for _embedding, vector in valid_pairs])
                next_index.train(vectors)
                for embedding, vector in valid_pairs:
                    next_index.add_embedding(str(embedding.id), vector)

        self._index = next_index
        self._index_type = index_type
        self._params = params
        self.save_snapshot(db)
        return self.stats()

    def rebuild_current(self, db: Session) -> dict[str, Any]:
        return self.rebuild(db, self._index_type, dict(self._params))

    def stats(self) -> dict[str, Any]:
        stats = self._index.stats()
        stats["loaded"] = self._index.count() > 0
        stats["file_path"] = str(self.current_snapshot_path or self.index_path)
        stats["last_snapshot_id"] = self.last_snapshot_id
        stats["model_name"] = self.model_name
        stats["pipeline"] = self.pipeline
        return stats

    def count(self) -> int:
        return self._index.count()
