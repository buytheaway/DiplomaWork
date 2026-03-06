"""Smoke tests — health endpoint + dummy extractor wiring."""

import os

os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_endpoint():
    with TestClient(create_app()) as client:
        response = client.get("/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "embedding_backend" in body


def test_testing_uses_dummy_extractor():
    with TestClient(create_app()) as client:
        assert client.app.state.extractor.model_name == "dummy"


def test_dummy_extractor_dim_matches_config():
    with TestClient(create_app()) as client:
        ext = client.app.state.extractor
        assert ext.dim == 512  # default EMBEDDING_DIM
