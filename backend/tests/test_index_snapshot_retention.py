from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TESTING", "true")
os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")

from app.core.config import get_settings
from app.services.index.index_manager import IndexManager
from app.services.storage.repositories import IndexSnapshotRepo


def _settings_for(tmp_path: Path, *, filename: str, retention: int):
    return get_settings().model_copy(
        update={
            "index_path": str(tmp_path / filename),
            "embedding_dim": 3,
            "index_type": "flat",
            "index_snapshot_retention": retention,
        }
    )


def _map_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".map.json")


def _save_snapshot(manager: IndexManager, db_session, seq: int) -> Path:
    manager.add_embedding(
        f"vec-{seq}",
        np.array([1.0, float(seq), 0.0], dtype=np.float32),
    )
    manager.save_snapshot(db_session)
    db_session.commit()
    assert manager.current_snapshot_path is not None
    return manager.current_snapshot_path


def test_index_snapshot_retention_keeps_only_latest_n(tmp_path, db_session):
    settings = _settings_for(tmp_path, filename="retained.faiss", retention=2)
    manager = IndexManager(settings)

    snapshots = [_save_snapshot(manager, db_session, seq) for seq in range(5)]
    remaining_rows = IndexSnapshotRepo(db_session).list_latest_for_path(str(manager.index_path))

    assert len(remaining_rows) == 2
    for old_path in snapshots[:-2]:
        assert not old_path.exists()
        assert not _map_path(old_path).exists()
    for kept_path in snapshots[-2:]:
        assert kept_path.exists()
        assert _map_path(kept_path).exists()


def test_index_snapshot_retention_prunes_map_sidecars(tmp_path, db_session):
    settings = _settings_for(tmp_path, filename="with-map.faiss", retention=1)
    manager = IndexManager(settings)

    first = _save_snapshot(manager, db_session, 1)
    second = _save_snapshot(manager, db_session, 2)

    assert not first.exists()
    assert not _map_path(first).exists()
    assert second.exists()
    assert _map_path(second).exists()


def test_index_snapshot_retention_zero_disables_pruning(tmp_path, db_session):
    settings = _settings_for(tmp_path, filename="disabled.faiss", retention=0)
    manager = IndexManager(settings)

    snapshots = [_save_snapshot(manager, db_session, seq) for seq in range(3)]
    remaining_rows = IndexSnapshotRepo(db_session).list_latest_for_path(str(manager.index_path))

    assert len(remaining_rows) == 3
    for snapshot_path in snapshots:
        assert snapshot_path.exists()
        assert _map_path(snapshot_path).exists()


def test_index_snapshot_pruning_does_not_delete_unrelated_files(tmp_path, db_session):
    unrelated_file = tmp_path / "main.untracked.faiss"
    unrelated_map = _map_path(unrelated_file)
    unrelated_file.write_bytes(b"not-managed")
    unrelated_map.write_bytes(b"not-managed-map")

    other_settings = _settings_for(tmp_path, filename="other.faiss", retention=0)
    other_manager = IndexManager(other_settings)
    other_snapshots = [_save_snapshot(other_manager, db_session, seq) for seq in range(2)]

    main_settings = _settings_for(tmp_path, filename="main.faiss", retention=1)
    main_manager = IndexManager(main_settings)
    _save_snapshot(main_manager, db_session, 1)
    _save_snapshot(main_manager, db_session, 2)

    assert unrelated_file.exists()
    assert unrelated_map.exists()
    for other_path in other_snapshots:
        assert other_path.exists()
        assert _map_path(other_path).exists()


def test_index_snapshot_save_failure_does_not_prune_existing_snapshots(
    tmp_path,
    db_session,
    monkeypatch,
):
    keep_settings = _settings_for(tmp_path, filename="failure.faiss", retention=0)
    keep_manager = IndexManager(keep_settings)
    snapshots = [_save_snapshot(keep_manager, db_session, seq) for seq in range(2)]

    prune_settings = _settings_for(tmp_path, filename="failure.faiss", retention=1)
    failing_manager = IndexManager(prune_settings)

    def fail_save_index_files() -> Path:
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr(failing_manager, "_save_index_files", fail_save_index_files)

    with pytest.raises(RuntimeError, match="simulated save failure"):
        failing_manager.save_snapshot(db_session)

    remaining_rows = IndexSnapshotRepo(db_session).list_latest_for_path(
        str(failing_manager.index_path)
    )
    assert len(remaining_rows) == 2
    for snapshot_path in snapshots:
        assert snapshot_path.exists()
        assert _map_path(snapshot_path).exists()
