from fastapi.testclient import TestClient

from src.clv_app.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200
    assert "ready" in response.json()


def test_model_info_endpoint_returns_metrics():
    response = client.get("/model/info")
    assert response.status_code == 200

    payload = response.json()
    assert "classifier" in payload
    assert "regressor" in payload
    assert "data" in payload


def test_pipeline_summary_endpoint_returns_output_stages():
    response = client.get("/pipeline/summary")
    assert response.status_code == 200

    payload = response.json()
    assert payload["throughput"]["records_processed"] > 0
    assert len(payload["pipeline_stages"]) > 0
