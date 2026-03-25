"""Tests for the FastAPI MLOps Pipeline API.

Covers all endpoints: health, pipeline execution/history, model registry
(register, list, get, 404), and experiment tracking (log, list, best).
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app, _run_history, _registry, _tracker

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_state():
    """Clear shared state between tests to avoid cross-contamination."""
    _run_history.clear()
    _registry._models.clear()
    _tracker._experiments.clear()
    yield


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
        assert "timestamp" in resp.json()

    def test_pipeline_status_idle(self):
        resp = client.get("/pipeline/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"
        assert data["total_runs"] == 0
        assert data["last_run"] is None


# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_run_pipeline_default(self):
        resp = client.post("/pipeline/run", json={"name": "test-pipe"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["pipeline"] == "test-pipe"
        assert len(data["steps"]) == 4

    def test_run_pipeline_custom_rows(self):
        resp = client.post("/pipeline/run", json={"name": "big", "n_rows": 5000})
        assert resp.status_code == 200
        steps = resp.json()["steps"]
        assert all(s["status"] == "success" for s in steps)

    def test_pipeline_history_empty(self):
        resp = client.get("/pipeline/history")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_pipeline_history_after_runs(self):
        client.post("/pipeline/run", json={"name": "run-1"})
        client.post("/pipeline/run", json={"name": "run-2"})
        resp = client.get("/pipeline/history?limit=5")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_pipeline_status_after_run(self):
        client.post("/pipeline/run", json={"name": "status-test"})
        resp = client.get("/pipeline/status")
        data = resp.json()
        assert data["total_runs"] == 1
        assert data["last_run"]["pipeline"] == "status-test"


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_register_and_list(self):
        body = {
            "name": "churn-clf",
            "version": "1.0",
            "metrics": {"accuracy": 0.93, "f1": 0.91},
            "metadata": {"author": "test"},
        }
        resp = client.post("/models/register", json=body)
        assert resp.status_code == 200
        assert resp.json()["registered"]["name"] == "churn-clf"

        resp = client.get("/models")
        assert len(resp.json()["models"]) == 1

    def test_get_model_found(self):
        client.post("/models/register", json={
            "name": "fraud", "version": "2.0", "metrics": {"auc": 0.98},
        })
        resp = client.get("/models/fraud/2.0")
        assert resp.status_code == 200
        assert resp.json()["version"] == "2.0"

    def test_get_model_not_found(self):
        resp = client.get("/models/nonexistent/0.0")
        assert resp.status_code == 404

    def test_list_empty(self):
        resp = client.get("/models")
        assert resp.json()["models"] == []


# ---------------------------------------------------------------------------
# Experiment Tracking
# ---------------------------------------------------------------------------

class TestExperiments:
    def test_log_experiment(self):
        body = {
            "name": "lr-sweep",
            "params": {"lr": 0.01, "epochs": 50},
            "metrics": {"accuracy": 0.88},
        }
        resp = client.post("/experiments/log", json=body)
        assert resp.status_code == 200
        assert resp.json()["logged"]["name"] == "lr-sweep"

    def test_list_experiments(self):
        client.post("/experiments/log", json={
            "name": "exp-a", "params": {"lr": 0.1}, "metrics": {"accuracy": 0.8},
        })
        client.post("/experiments/log", json={
            "name": "exp-b", "params": {"lr": 0.01}, "metrics": {"accuracy": 0.9},
        })
        resp = client.get("/experiments")
        assert len(resp.json()["experiments"]) == 2

    def test_best_experiment(self):
        client.post("/experiments/log", json={
            "name": "low", "params": {}, "metrics": {"accuracy": 0.7},
        })
        client.post("/experiments/log", json={
            "name": "high", "params": {}, "metrics": {"accuracy": 0.95},
        })
        resp = client.get("/experiments/best?metric=accuracy&mode=max")
        assert resp.status_code == 200
        assert resp.json()["best"]["name"] == "high"

    def test_best_experiment_min_mode(self):
        client.post("/experiments/log", json={
            "name": "low-loss", "params": {}, "metrics": {"loss": 0.1},
        })
        client.post("/experiments/log", json={
            "name": "high-loss", "params": {}, "metrics": {"loss": 0.9},
        })
        resp = client.get("/experiments/best?metric=loss&mode=min")
        assert resp.json()["best"]["name"] == "low-loss"

    def test_best_experiment_none_logged(self):
        resp = client.get("/experiments/best?metric=accuracy")
        assert resp.status_code == 404
