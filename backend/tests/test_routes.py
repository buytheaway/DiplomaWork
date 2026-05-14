"""Integration tests for all API v1 routes.

Uses the ``client`` fixture from conftest which injects an in-memory SQLite
session and wires up the full FastAPI app with ``DummyEmbeddingExtractor``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from app.core.config import settings
from app.db.models import AuditLog, Embedding, Person
from app.services.embeddings.interface import (
    DummyEmbeddingExtractor,
    FaceEmbedding,
    NoFaceDetectedError,
)
from app.services.runtime.pipeline_registry import PipelineRuntime

# ── health ───────────────────────────────────────────────────────────────────


def test_health(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "embedding_backend" in body
    assert "available_pipelines" in body


# ── enroll ───────────────────────────────────────────────────────────────────


def test_enroll_success(client):
    resp = client.post(
        "/v1/enroll",
        files={"file": ("face.jpg", b"\x89PNG_fake_image_bytes", "image/jpeg")},
        data={"label": "Alice"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "dummy"
    assert body["dim"] == 512
    assert body["faces_detected"] == 1
    assert body["pipeline"] == "both"
    assert len(body["enrollments"]) == 2
    assert "person_id" in body
    assert "embedding_id" in body


def test_enroll_without_label(client):
    resp = client.post(
        "/v1/enroll",
        files={"file": ("face.jpg", b"\x89PNG_some_bytes", "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "dummy"
    assert resp.json()["pipeline"] == "both"


def test_enroll_custom_pipeline(client):
    resp = client.post(
        "/v1/enroll",
        files={"file": ("face.jpg", b"\x89PNG_custom_image_bytes", "image/jpeg")},
        data={"label": "CustomAlice", "pipeline": "custom"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["pipeline"] == "custom"
    assert body["model"] == "dummy_custom"


def test_enroll_both_pipelines(client):
    resp = client.post(
        "/v1/enroll",
        files={"file": ("face.jpg", b"\x89PNG_both_image_bytes", "image/jpeg")},
        data={"label": "DualAlice", "pipeline": "both"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["pipeline"] == "both"
    assert len(body["enrollments"]) == 2
    assert {item["pipeline"] for item in body["enrollments"]} == {"pretrained", "custom"}


def test_enroll_same_label_reuses_person_and_rebuilds_pipelines(client, db_session):
    first_resp = client.post(
        "/v1/enroll",
        files={"file": ("face.jpg", b"\x89PNG_duplicate_one", "image/jpeg")},
        data={"label": "Duplicate Alice"},
    )
    second_resp = client.post(
        "/v1/enroll",
        files={"file": ("face.jpg", b"\x89PNG_duplicate_two", "image/jpeg")},
        data={"label": "Duplicate Alice"},
    )

    assert first_resp.status_code == 200
    assert second_resp.status_code == 200
    assert second_resp.json()["person_id"] == first_resp.json()["person_id"]

    active_people = db_session.execute(
        select(Person).where(Person.label == "Duplicate Alice", Person.status == "active")
    ).scalars().all()
    assert len(active_people) == 1

    active_embeddings = db_session.execute(
        select(Embedding).where(
            Embedding.person_id == active_people[0].id,
            Embedding.is_active.is_(True),
        )
    ).scalars().all()
    inactive_embeddings = db_session.execute(
        select(Embedding).where(
            Embedding.person_id == active_people[0].id,
            Embedding.is_active.is_(False),
        )
    ).scalars().all()
    assert {item.pipeline for item in active_embeddings} == {"pretrained", "custom"}
    assert len(inactive_embeddings) == 2

    pretrained_stats = client.get("/v1/index/stats", params={"pipeline": "pretrained"})
    custom_stats = client.get("/v1/index/stats", params={"pipeline": "custom"})
    assert pretrained_stats.json()["embeddings_count"] == 1
    assert custom_stats.json()["embeddings_count"] == 1


def test_enroll_empty_file_returns_400(client):
    """DummyExtractor rejects empty bytes → 400."""
    resp = client.post(
        "/v1/enroll",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400


# ── search ───────────────────────────────────────────────────────────────────


def test_search_empty_index(client):
    resp = client.post(
        "/v1/search",
        params={"k": 3},
        files={"file": ("face.jpg", b"\x89PNG_query", "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"] == []
    assert body["k"] == 3
    assert body["decision"] == "unknown"
    assert body["best_score"] is None
    assert body["threshold_used"] > 0


def test_search_after_enroll(client):
    # Enroll first
    client.post(
        "/v1/enroll",
        files={"file": ("a.jpg", b"\x89PNG_enroll", "image/jpeg")},
        data={"label": "Bob"},
    )
    # Search
    resp = client.post(
        "/v1/search",
        params={"k": 1},
        files={"file": ("q.jpg", b"\x89PNG_query", "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    results = body["results"]
    assert len(results) >= 1
    assert results[0]["label"] == "Bob"
    # threshold / decision fields present
    assert body["best_score"] is not None
    assert isinstance(body["threshold_used"], float)
    assert body["decision"] == "match"
    assert isinstance(body["best_match_above_threshold"], bool)


def test_search_one_face_unknown(client, monkeypatch):
    client.post(
        "/v1/enroll",
        files={"file": ("a.jpg", b"\x89PNG_unknown_case", "image/jpeg")},
        data={"label": "ThresholdBob"},
    )
    monkeypatch.setattr(settings, "match_threshold", 1.1)

    resp = client.post(
        "/v1/search",
        params={"k": 1},
        files={"file": ("q.jpg", b"\x89PNG_query_unknown", "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["faces_detected"] == 1
    assert body["decision"] == "unknown"
    assert body["best_match_above_threshold"] is False
    assert body["best_score"] is not None
    assert body["results"] == []


def test_search_custom_pipeline_after_custom_enroll(client):
    client.post(
        "/v1/enroll",
        files={"file": ("a.jpg", b"\x89PNG_custom_enroll", "image/jpeg")},
        data={"label": "BobCustom", "pipeline": "custom"},
    )
    resp = client.post(
        "/v1/search",
        params={"k": 1, "pipeline": "custom"},
        files={"file": ("q.jpg", b"\x89PNG_query_custom", "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["pipeline"] == "custom"
    assert body["model"] == "dummy_custom"
    assert len(body["results"]) >= 1
    assert body["results"][0]["pipeline"] == "custom"


def test_rebuild_filters_same_model_by_pipeline(client):
    registry = client.app.state.pipeline_registry
    registry.get("custom").extractor.model_name = registry.get("pretrained").extractor.model_name

    client.post(
        "/v1/enroll",
        files={"file": ("a.jpg", b"\x89PNG_same_model_custom", "image/jpeg")},
        data={"label": "SameModelCustom", "pipeline": "custom"},
    )

    rebuild_resp = client.post(
        "/v1/index/rebuild",
        json={"index_type": "hnsw", "params": {}, "pipeline": "pretrained"},
    )
    assert rebuild_resp.status_code == 200
    assert rebuild_resp.json()["embeddings_count"] == 0

    pretrained_resp = client.post(
        "/v1/search",
        params={"k": 1, "pipeline": "pretrained"},
        files={"file": ("q.jpg", b"\x89PNG_same_model_query", "image/jpeg")},
    )
    assert pretrained_resp.status_code == 200
    assert pretrained_resp.json()["results"] == []

    custom_resp = client.post(
        "/v1/search",
        params={"k": 1, "pipeline": "custom"},
        files={"file": ("q.jpg", b"\x89PNG_same_model_query", "image/jpeg")},
    )
    assert custom_resp.status_code == 200
    assert custom_resp.json()["results"][0]["label"] == "SameModelCustom"


def _strip_search_latency(payload: dict) -> dict:
    normalized = dict(payload)
    normalized["latency_ms"] = None
    breakdown = normalized.get("latency_breakdown")
    if isinstance(breakdown, dict):
        normalized["latency_breakdown"] = {
            "detect_ms": None,
            "embed_ms": breakdown.get("embed_ms"),
            "search_ms": None,
            "total_ms": None,
        }
    return normalized


def test_search_audit_can_be_disabled_without_changing_response(client, db_session, monkeypatch):
    enabled_resp = client.post(
        "/v1/search",
        params={"k": 1},
        files={"file": ("q.jpg", b"\x89PNG_query_audit_on", "image/jpeg")},
    )
    assert enabled_resp.status_code == 200

    monkeypatch.setattr(settings, "enable_search_audit", False)

    disabled_resp = client.post(
        "/v1/search",
        params={"k": 1},
        files={"file": ("q.jpg", b"\x89PNG_query_no_audit", "image/jpeg")},
    )

    assert disabled_resp.status_code == 200
    assert _strip_search_latency(enabled_resp.json()) == _strip_search_latency(disabled_resp.json())
    logs = db_session.execute(
        select(AuditLog).where(AuditLog.event_type == "search")
    ).scalars().all()
    assert len(logs) == 1


class _MultiFaceDummyExtractor(DummyEmbeddingExtractor):
    def __init__(self, dim: int = 512) -> None:
        super().__init__(dim=dim)
        self.model_name = "dummy_multiface"

    def extract_embeddings(self, image_bytes: bytes) -> list[FaceEmbedding]:
        base = self.extract_embedding(image_bytes)
        second = base.copy()
        second[0] = 0.6
        second[1] = 0.8
        return [
            FaceEmbedding(embedding=base, detection_score=0.99, bbox=(0.0, 0.0, 16.0, 16.0)),
            FaceEmbedding(embedding=second, detection_score=0.95, bbox=(24.0, 8.0, 40.0, 24.0)),
        ]


class _NoFaceDummyExtractor(DummyEmbeddingExtractor):
    def __init__(self, dim: int = 512) -> None:
        super().__init__(dim=dim)
        self.model_name = "dummy_noface"

    def extract_embeddings(self, image_bytes: bytes) -> list[FaceEmbedding]:
        raise NoFaceDetectedError("No face detected in image")


def test_search_multiple_faces_returns_face_indexes(client):
    client.post(
        "/v1/enroll",
        files={"file": ("a.jpg", b"\x89PNG_multi_enroll", "image/jpeg")},
        data={"label": "MultiBob"},
    )

    registry = client.app.state.pipeline_registry
    runtime = registry.get("pretrained")
    registry._pipelines["pretrained"] = PipelineRuntime(
        key="pretrained",
        backend=runtime.backend,
        extractor=_MultiFaceDummyExtractor(),
        index_manager=runtime.index_manager,
    )
    client.app.state.extractor = registry.get("pretrained").extractor

    resp = client.post(
        "/v1/search",
        params={"k": 1, "pipeline": "pretrained"},
        files={"file": ("q.jpg", b"\x89PNG_multi_query", "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["faces_detected"] == 2
    assert {item["face_index"] for item in body["results"]} == {0, 1}
    assert all("detection_score" in item for item in body["results"])


def test_search_no_faces_returns_422(client):
    registry = client.app.state.pipeline_registry
    runtime = registry.get("pretrained")
    registry._pipelines["pretrained"] = PipelineRuntime(
        key="pretrained",
        backend=runtime.backend,
        extractor=_NoFaceDummyExtractor(),
        index_manager=runtime.index_manager,
    )
    client.app.state.extractor = registry.get("pretrained").extractor

    resp = client.post(
        "/v1/search",
        params={"k": 1, "pipeline": "pretrained"},
        files={"file": ("q.jpg", b"\x89PNG_no_face", "image/jpeg")},
    )
    assert resp.status_code == 422
    assert "no face" in resp.json()["detail"].lower()


def test_search_empty_file_returns_400(client):
    resp = client.post(
        "/v1/search",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_search_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_bytes", 4)
    resp = client.post(
        "/v1/search",
        files={"file": ("big.jpg", b"12345", "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "too large" in resp.json()["detail"].lower()


def test_enroll_rejects_unsupported_content_type(client):
    resp = client.post(
        "/v1/enroll",
        files={"file": ("note.txt", b"not-an-image", "text/plain")},
        data={"label": "WrongType"},
    )
    assert resp.status_code == 400
    assert "unsupported file type" in resp.json()["detail"].lower()


# ── persons ──────────────────────────────────────────────────────────────────


def test_list_persons_empty(client):
    resp = client.get("/v1/persons")
    assert resp.status_code == 200
    assert resp.json() == {
        "items": [],
        "total": 0,
        "limit": 200,
        "offset": 0,
    }


def test_list_persons_after_enroll(client):
    client.post(
        "/v1/enroll",
        files={"file": ("f1.jpg", b"\x89PNG_list_test", "image/jpeg")},
        data={"label": "ListTest"},
    )
    resp = client.get("/v1/persons")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    assert body["limit"] == 200
    assert body["offset"] == 0
    assert len(body["items"]) >= 1
    assert body["items"][0]["label"] == "ListTest"
    # Должен быть без embeddings (это list-endpoint)
    assert "embeddings" not in body["items"][0]


def test_list_persons_total_uses_all_active_rows_not_page_size(client, db_session):
    now = datetime.now(UTC)
    for idx in range(3):
        db_session.add(
            Person(
                label=f"PagePerson{idx}",
                created_at=now + timedelta(seconds=idx),
                updated_at=now + timedelta(seconds=idx),
            )
        )
    db_session.add(
        Person(
            label="DeletedPagePerson",
            status="deleted",
            created_at=now + timedelta(seconds=10),
            updated_at=now + timedelta(seconds=10),
        )
    )
    db_session.commit()

    resp = client.get("/v1/persons", params={"limit": 2, "offset": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 0
    assert len(body["items"]) == 2
    assert all(item["status"] == "active" for item in body["items"])


def test_list_persons_limit_offset_pages_active_rows(client, db_session):
    now = datetime.now(UTC)
    for idx in range(3):
        db_session.add(
            Person(
                label=f"OffsetPerson{idx}",
                created_at=now + timedelta(seconds=idx),
                updated_at=now + timedelta(seconds=idx),
            )
        )
    db_session.commit()

    resp = client.get("/v1/persons", params={"limit": 1, "offset": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert body["limit"] == 1
    assert body["offset"] == 1
    assert len(body["items"]) == 1


def test_list_persons_search_filters_total_and_items(client, db_session):
    now = datetime.now(UTC)
    db_session.add_all(
        [
            Person(label="Aidar Sarsenov #SCALE-000001", created_at=now),
            Person(label="Aidar Omarov #SCALE-000002", created_at=now + timedelta(seconds=1)),
            Person(label="Miras Tulegenov #SCALE-000003", created_at=now + timedelta(seconds=2)),
            Person(
                label="Aidar Deleted #SCALE-000004",
                status="deleted",
                created_at=now + timedelta(seconds=3),
            ),
        ]
    )
    db_session.commit()

    resp = client.get("/v1/persons", params={"q": "Aidar", "limit": 1, "offset": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert body["limit"] == 1
    assert body["offset"] == 0
    assert len(body["items"]) == 1
    assert "Aidar" in body["items"][0]["label"]


def test_list_persons_search_paginates_matching_rows(client, db_session):
    now = datetime.now(UTC)
    for idx in range(3):
        db_session.add(
            Person(
                label=f"Scale Match {idx}",
                created_at=now + timedelta(seconds=idx),
                updated_at=now + timedelta(seconds=idx),
            )
        )
    db_session.add(Person(label="Other Person", created_at=now + timedelta(seconds=10)))
    db_session.commit()

    resp = client.get("/v1/persons", params={"q": "Scale Match", "limit": 1, "offset": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert body["limit"] == 1
    assert body["offset"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["label"].startswith("Scale Match")


def test_list_persons_search_by_exact_person_id(client, db_session):
    person = Person(label="Exact Id Search")
    db_session.add(person)
    db_session.commit()

    resp = client.get("/v1/persons", params={"q": str(person.id)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == str(person.id)


def test_list_persons_rejects_limit_above_max(client):
    resp = client.get("/v1/persons", params={"limit": 501})
    assert resp.status_code == 422


def test_database_stats_empty(client):
    resp = client.get("/v1/database/stats")
    assert resp.status_code == 200
    assert resp.json() == {
        "active_persons": 0,
        "active_embeddings": 0,
        "embeddings_by_pipeline_model": [],
    }


def test_database_stats_counts_active_records_and_groups(client, db_session):
    active_person = Person(label="Active Identity")
    deleted_person = Person(label="Deleted Identity", status="deleted")
    db_session.add_all([active_person, deleted_person])
    db_session.flush()
    db_session.add_all(
        [
            Embedding(
                person_id=active_person.id,
                pipeline="pretrained",
                model="model_a",
                dim=4,
                vector=b"\0" * 16,
                is_active=True,
            ),
            Embedding(
                person_id=active_person.id,
                pipeline="custom",
                model="model_b",
                dim=4,
                vector=b"\0" * 16,
                is_active=True,
            ),
            Embedding(
                person_id=active_person.id,
                pipeline="pretrained",
                model="model_a",
                dim=4,
                vector=b"\0" * 16,
                is_active=False,
            ),
            Embedding(
                person_id=deleted_person.id,
                pipeline="pretrained",
                model="model_a",
                dim=4,
                vector=b"\0" * 16,
                is_active=True,
            ),
        ]
    )
    db_session.commit()

    resp = client.get("/v1/database/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active_persons"] == 1
    assert body["active_embeddings"] == 2
    assert body["embeddings_by_pipeline_model"] == [
        {"pipeline": "custom", "model_name": "model_b", "count": 1},
        {"pipeline": "pretrained", "model_name": "model_a", "count": 1},
    ]


def test_get_person_after_enroll(client):
    enroll_resp = client.post(
        "/v1/enroll",
        files={"file": ("c.jpg", b"\x89PNG_charlie", "image/jpeg")},
        data={"label": "Charlie"},
    )
    person_id = enroll_resp.json()["person_id"]

    resp = client.get(f"/v1/persons/{person_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] == "Charlie"
    assert body["status"] == "active"
    assert len(body["embeddings"]) == 2
    assert {item["pipeline"] for item in body["embeddings"]} == {"pretrained", "custom"}


def test_get_person_not_found(client):
    resp = client.get("/v1/persons/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


def test_delete_person(client):
    enroll_resp = client.post(
        "/v1/enroll",
        files={"file": ("d.jpg", b"\x89PNG_dave", "image/jpeg")},
        data={"label": "Dave"},
    )
    person_id = enroll_resp.json()["person_id"]

    resp = client.delete(f"/v1/persons/{person_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # After delete, GET still returns the person but status is "deleted"
    get_resp = client.get(f"/v1/persons/{person_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == "deleted"


def test_delete_person_removes_results_from_index(client):
    enroll_resp = client.post(
        "/v1/enroll",
        files={"file": ("d2.jpg", b"\x89PNG_delete_index", "image/jpeg")},
        data={"label": "DeleteIndex"},
    )
    person_id = enroll_resp.json()["person_id"]

    delete_resp = client.delete(f"/v1/persons/{person_id}")
    assert delete_resp.status_code == 200

    search_resp = client.post(
        "/v1/search",
        params={"k": 3},
        files={"file": ("q.jpg", b"\x89PNG_query_after_delete", "image/jpeg")},
    )
    assert search_resp.status_code == 200
    assert search_resp.json()["results"] == []


def test_delete_person_rebuilds_only_affected_pipeline(client, db_session):
    pre_resp = client.post(
        "/v1/enroll",
        files={"file": ("pre.jpg", b"\x89PNG_delete_pre", "image/jpeg")},
        data={"label": "KeepPretrained", "pipeline": "pretrained"},
    )
    assert pre_resp.status_code == 200

    custom_resp = client.post(
        "/v1/enroll",
        files={"file": ("custom.jpg", b"\x89PNG_delete_custom", "image/jpeg")},
        data={"label": "DeleteCustom", "pipeline": "custom"},
    )
    assert custom_resp.status_code == 200
    custom_person_id = custom_resp.json()["person_id"]

    delete_resp = client.delete(f"/v1/persons/{custom_person_id}")
    assert delete_resp.status_code == 200

    log = db_session.execute(
        select(AuditLog)
        .where(AuditLog.event_type == "delete_person")
        .order_by(AuditLog.created_at.desc())
    ).scalars().first()
    assert log is not None
    assert log.details["rebuilt_pipelines"] == ["custom"]

    search_resp = client.post(
        "/v1/search",
        params={"k": 1, "pipeline": "pretrained"},
        files={"file": ("q.jpg", b"\x89PNG_query_after_custom_delete", "image/jpeg")},
    )
    assert search_resp.status_code == 200
    assert search_resp.json()["results"][0]["label"] == "KeepPretrained"


def test_delete_person_not_found(client):
    resp = client.delete("/v1/persons/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


# ── index ────────────────────────────────────────────────────────────────────


def test_index_stats(client):
    resp = client.get("/v1/index/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "index_type" in body
    assert "embeddings_count" in body
    assert "is_trained" in body
    assert "embedding_backend" in body
    assert "pipeline" in body


def test_index_stats_after_enroll(client):
    client.post(
        "/v1/enroll",
        files={"file": ("e.jpg", b"\x89PNG_eve", "image/jpeg")},
        data={"label": "Eve"},
    )
    resp = client.get("/v1/index/stats")
    assert resp.status_code == 200
    assert resp.json()["embeddings_count"] >= 1
