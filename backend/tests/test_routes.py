"""Integration tests for all API v1 routes.

Uses the ``client`` fixture from conftest which injects an in-memory SQLite
session and wires up the full FastAPI app with ``DummyEmbeddingExtractor``.
"""

from __future__ import annotations

import io

import pytest


# ── health ───────────────────────────────────────────────────────────────────


def test_health(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "embedding_backend" in body


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


def test_search_empty_file_returns_400(client):
    resp = client.post(
        "/v1/search",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400


# ── persons ──────────────────────────────────────────────────────────────────


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


def test_index_stats_after_enroll(client):
    client.post(
        "/v1/enroll",
        files={"file": ("e.jpg", b"\x89PNG_eve", "image/jpeg")},
        data={"label": "Eve"},
    )
    resp = client.get("/v1/index/stats")
    assert resp.status_code == 200
    assert resp.json()["embeddings_count"] >= 1
