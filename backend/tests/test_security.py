from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.config import get_settings
from app.db.models import AuditLog
from app.main import create_app
from app.security import auth as auth_module
from app.security.auth import classify_api_key
from app.security.crypto import (
    decrypt_embedding_payload,
    decrypt_snapshot_payload,
    encrypt_embedding_payload,
)
from app.security.rate_limit import rate_limit_rule_for_request
from app.services.index.index_manager import IndexManager
from app.services.storage.repositories import AuditLogRepo


def _make_rate_limited_client(
    db_session: Session,
    monkeypatch,
    *,
    enabled: bool = True,
    search_limit: int = 100,
    enroll_limit: int = 100,
    admin_limit: int = 100,
    api_key: str = "operator-key",
    admin_api_key: str = "admin-key",
) -> TestClient:
    settings = get_settings()
    monkeypatch.setattr(settings, "rate_limit_enabled", enabled)
    monkeypatch.setattr(settings, "rate_limit_search_per_min", search_limit)
    monkeypatch.setattr(settings, "rate_limit_enroll_per_min", enroll_limit)
    monkeypatch.setattr(settings, "rate_limit_admin_per_min", admin_limit)
    monkeypatch.setattr(settings, "api_key", api_key)
    monkeypatch.setattr(settings, "admin_api_key", admin_api_key)

    app = create_app()

    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    return TestClient(app)


def _post_search(client: TestClient, api_key: str = "operator-key", payload: bytes = b"query"):
    return client.post(
        "/v1/search",
        headers={"X-API-Key": api_key},
        params={"k": 1},
        files={"file": ("q.jpg", b"\x89PNG_" + payload, "image/jpeg")},
    )


def _post_enroll(client: TestClient, api_key: str = "operator-key", payload: bytes = b"enroll"):
    return client.post(
        "/v1/enroll",
        headers={"X-API-Key": api_key},
        files={"file": ("face.jpg", b"\x89PNG_" + payload, "image/jpeg")},
        data={"label": "RateLimit"},
    )


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


def test_classify_api_key_missing_key_fails():
    settings = get_settings().model_copy(
        update={
            "api_key": "operator-key",
            "admin_api_key": "admin-key",
        }
    )

    assert classify_api_key(settings, "") is None


def test_classify_api_key_uses_timing_safe_compare(monkeypatch):
    settings = get_settings().model_copy(
        update={
            "api_key": "operator-key",
            "admin_api_key": "admin-key",
        }
    )
    calls: list[tuple[str, str]] = []

    def fake_compare_digest(candidate: str, expected: str) -> bool:
        calls.append((candidate, expected))
        return candidate == expected

    monkeypatch.setattr(auth_module.secrets, "compare_digest", fake_compare_digest)

    assert classify_api_key(settings, "operator-key") == "operator"
    assert ("operator-key", "admin-key") in calls
    assert ("operator-key", "operator-key") in calls


def test_search_rate_limit_returns_429_after_configured_limit(db_session, monkeypatch):
    client = _make_rate_limited_client(db_session, monkeypatch, search_limit=1)
    with client:
        assert _post_search(client, payload=b"first").status_code == 200
        limited = _post_search(client, payload=b"second")

    assert limited.status_code == 429
    assert limited.json()["detail"] == "Rate limit exceeded. Try again later."
    assert "Retry-After" in limited.headers


def test_enroll_rate_limit_returns_429_after_configured_limit(db_session, monkeypatch):
    client = _make_rate_limited_client(db_session, monkeypatch, enroll_limit=1)
    with client:
        assert _post_enroll(client, payload=b"first").status_code == 200
        limited = _post_enroll(client, payload=b"second")

    assert limited.status_code == 429
    assert limited.json()["detail"] == "Rate limit exceeded. Try again later."


def test_disabled_rate_limiter_does_not_affect_requests(db_session, monkeypatch):
    client = _make_rate_limited_client(
        db_session,
        monkeypatch,
        enabled=False,
        search_limit=1,
    )
    with client:
        assert _post_search(client, payload=b"first").status_code == 200
        assert _post_search(client, payload=b"second").status_code == 200


def test_rate_limit_buckets_are_separate_for_valid_api_keys(db_session, monkeypatch):
    client = _make_rate_limited_client(db_session, monkeypatch, search_limit=1)
    with client:
        assert _post_search(client, api_key="operator-key", payload=b"operator").status_code == 200
        assert _post_search(client, api_key="operator-key", payload=b"operator-2").status_code == 429
        assert _post_search(client, api_key="admin-key", payload=b"admin").status_code == 200


def test_rate_limit_does_not_log_api_key(db_session, monkeypatch, caplog):
    api_key = "operator-secret-not-for-logs"
    client = _make_rate_limited_client(
        db_session,
        monkeypatch,
        search_limit=1,
        api_key=api_key,
    )
    with client:
        assert _post_search(client, api_key=api_key, payload=b"first").status_code == 200
        limited = _post_search(client, api_key=api_key, payload=b"second")

    assert limited.status_code == 429
    assert api_key not in limited.text
    assert api_key not in caplog.text


def test_rate_limit_rules_cover_sensitive_routes():
    settings = get_settings()

    assert rate_limit_rule_for_request(settings, "POST", "/v1/search").category == "search"
    assert rate_limit_rule_for_request(settings, "POST", "/v1/enroll").category == "enroll"
    assert rate_limit_rule_for_request(settings, "POST", "/v1/index/rebuild").category == "admin"
    assert rate_limit_rule_for_request(settings, "DELETE", "/v1/persons/person-1").category == "admin"
    assert rate_limit_rule_for_request(settings, "GET", "/v1/health") is None


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
        created_at=datetime.now(UTC) - timedelta(days=90),
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

    snapshot_path = manager.current_snapshot_path
    assert snapshot_path is not None
    map_path = snapshot_path.with_suffix(snapshot_path.suffix + ".map.json")
    assert snapshot_path.read_bytes().startswith(b"ENC1")
    assert map_path.read_bytes().startswith(b"ENC1")
    assert decrypt_snapshot_payload(snapshot_path.read_bytes())

    reloaded = IndexManager(settings)
    assert reloaded.load_latest_snapshot(db_session) is True
    results = reloaded.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert results[0].embedding_id == "vec-1"


def test_index_manager_loads_latest_valid_snapshot(tmp_path, db_session):
    settings = get_settings().model_copy(
        update={
            "index_path": str(tmp_path / "rolling.faiss"),
            "embedding_dim": 3,
        }
    )
    manager = IndexManager(settings)
    manager.add_embedding("vec-1", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    manager.save_snapshot(db_session)
    db_session.commit()

    manager.add_embedding("vec-2", np.array([0.0, 1.0, 0.0], dtype=np.float32))
    manager.save_snapshot(db_session)
    db_session.commit()

    reloaded = IndexManager(settings)
    assert reloaded.load_latest_snapshot(db_session) is True
    latest_results = reloaded.search(np.array([0.0, 1.0, 0.0], dtype=np.float32), k=1)
    assert latest_results[0].embedding_id == "vec-2"

    latest_snapshot_path = manager.current_snapshot_path
    assert latest_snapshot_path is not None
    latest_map_path = latest_snapshot_path.with_suffix(latest_snapshot_path.suffix + ".map.json")
    latest_map_path.unlink()

    fallback = IndexManager(settings)
    assert fallback.load_latest_snapshot(db_session) is True
    fallback_results = fallback.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert fallback_results[0].embedding_id == "vec-1"
