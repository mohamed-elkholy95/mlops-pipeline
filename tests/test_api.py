import pytest
from fastapi.testclient import TestClient
from src.api.main import app
client = TestClient(app)

class TestAPI:
    def test_health(self): assert client.get("/health").status_code == 200
    def test_status(self): assert client.get("/pipeline/status").status_code == 200
    def test_models(self): assert client.get("/models").status_code == 200
