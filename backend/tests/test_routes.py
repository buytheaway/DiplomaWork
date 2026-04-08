"""Integration tests for all API v1 routes.

Uses the ``client`` fixture from conftest which injects an in-memory SQLite
session and wires up the full FastAPI app with ``DummyEmbeddingExtractor``.
"""

from __future__ import annotations

from sqlalchemy import select

from app.core.config import settings
from app.db.models import AuditLog
from app.services.embeddings.interface import DummyEmbeddingExtractor, FaceEmbedding, NoFaceDetectedError
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
    assert "person_id" in body
    assert "embedding_id" in body


def test_enroll_without_label(client):
    resp = client.post(
        "/v1/enroll",
        files={"file": ("face.jpg", b"\x89PNG_some_bytes", "image/jpeg")},
    )
    assert resp.status_code == 200
    assert resp.json()["model"] == "dummy"


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
    assert body["decision"] in ("match", "unknown")
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
    assert len(body["results"]) >= 1


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


def test_search_compare(client):
    client.post(
        "/v1/enroll",
        files={"file": ("a.jpg", b"\x89PNG_dual_compare", "image/jpeg")},
        data={"label": "CompareBob", "pipeline": "both"},
    )
    resp = client.post(
        "/v1/search/compare",
        params={"k": 2},
        files={"file": ("q.jpg", b"\x89PNG_query_compare", "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "comparisons" in body
    assert len(body["comparisons"]) == 2
    assert {item["pipeline"] for item in body["comparisons"]} == {"pretrained", "custom"}


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


def _strip_compare_latency(payload: dict) -> dict:
    normalized = dict(payload)
    comparisons = []
    for item in normalized.get("comparisons", []):
        row = dict(item)
        row["latency_ms"] = None
        row["latency_breakdown"] = None
        comparisons.append(row)
    normalized["comparisons"] = comparisons
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


def test_compare_audit_can_be_disabled_without_changing_response(client, db_session, monkeypatch):
    enabled_resp = client.post(
        "/v1/search/compare",
        params={"k": 1},
        files={"file": ("q.jpg", b"\x89PNG_compare_audit_on", "image/jpeg")},
    )
    assert enabled_resp.status_code == 200

    monkeypatch.setattr(settings, "enable_compare_audit", False)

    disabled_resp = client.post(
        "/v1/search/compare",
        params={"k": 1},
        files={"file": ("q.jpg", b"\x89PNG_compare_no_audit", "image/jpeg")},
    )

    assert disabled_resp.status_code == 200
    assert _strip_compare_latency(enabled_resp.json()) == _strip_compare_latency(disabled_resp.json())
    logs = db_session.execute(
        select(AuditLog).where(AuditLog.event_type == "search_compare")
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
    assert resp.json() == []


def test_list_persons_after_enroll(client):
    client.post(
        "/v1/enroll",
        files={"file": ("f1.jpg", b"\x89PNG_list_test", "image/jpeg")},
        data={"label": "ListTest"},
    )
    resp = client.get("/v1/persons")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) >= 1
    assert body[0]["label"] == "ListTest"
    # Должен быть без embeddings (это list-endpoint)
    assert "embeddings" not in body[0]


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
    assert len(body["embeddings"]) == 1


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
