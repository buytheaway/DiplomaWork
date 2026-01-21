import os

from fastapi.testclient import TestClient

os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite+pysqlite:///:memory:"

from app.main import create_app


def test_health_endpoint():
    client = TestClient(create_app())
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
