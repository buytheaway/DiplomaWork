from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from app.core.config import get_settings
from app.db.models import AuditLog
from app.security.auth import classify_api_key
from app.security.crypto import (
    decrypt_embedding_payload,
    decrypt_snapshot_payload,
    encrypt_embedding_payload,
)
from app.services.index.index_manager import IndexManager
from app.services.storage.repositories import AuditLogRepo


def test_classify_api_key_operator_and_admin():
    settings = get_settings().model_copy(
        update={
            "api_key": "operator-key",
            "admin_api_key": "admin-key",
        }
    )
    assert classify_api_key(settings, "operator-key") == "operator"
    assert classify_api_key(settings, "admin-key") == "admin"
    assert classify_api_key(settings, "wrong") is None


def test_embedding_payload_roundtrip():
    raw = b"\x01\x02\x03\x04"
    encrypted = encrypt_embedding_payload(raw)
    assert encrypted != raw
    assert decrypt_embedding_payload(encrypted) == raw


def test_legacy_embedding_payload_passthrough():
    raw = b"\x10\x11legacy"
    assert decrypt_embedding_payload(raw) == raw


def test_audit_log_retention_prunes_old_rows(db_session):
    old_entry = AuditLog(
        event_type="search",
        actor_role="operator",
        route="/v1/search",
        status_code=200,
        details={"old": True},
        created_at=datetime.now(timezone.utc) - timedelta(days=90),
    )
    fresh_entry = AuditLog(
        event_type="search",
        actor_role="operator",
        route="/v1/search",
        status_code=200,
        details={"old": False},
    )
    db_session.add_all([old_entry, fresh_entry])
    db_session.commit()

    removed = AuditLogRepo(db_session).prune_older_than(30)
    db_session.commit()

    assert removed == 1
    remaining = db_session.query(AuditLog).all()
    assert len(remaining) == 1
    assert remaining[0].details["old"] is False


def test_index_manager_saves_encrypted_snapshot_files(tmp_path, db_session):
    settings = get_settings().model_copy(
        update={
            "index_path": str(tmp_path / "secure.faiss"),
            "embedding_dim": 3,
        }
    )
    manager = IndexManager(settings)
    manager.add_embedding("vec-1", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    manager.save_snapshot(db_session)
    db_session.commit()

    snapshot_path = tmp_path / "secure.faiss"
    map_path = tmp_path / "secure.faiss.map.json"
    assert snapshot_path.read_bytes().startswith(b"ENC1")
    assert map_path.read_bytes().startswith(b"ENC1")
    assert decrypt_snapshot_payload(snapshot_path.read_bytes())

    reloaded = IndexManager(settings)
    assert reloaded.load_latest_snapshot(db_session) is True
    results = reloaded.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert results[0].embedding_id == "vec-1"
