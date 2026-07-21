from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "cashgap-lab"}


def test_missing_dataset_returns_404() -> None:
    with TestClient(app) as client:
        response = client.get("/api/datasets/not-found")
    assert response.status_code == 404

