"""Tests for the FastAPI service (uses the small fixture model, not models/model.joblib)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import STATE, app

RESULT_KEYS = {"prediction", "label", "probability_disease", "confidence", "model_name"}


@pytest.fixture
def client(trained_pipeline):
    with TestClient(app) as c:  # runs the startup/shutdown lifespan
        STATE["model"] = trained_pipeline
        STATE["meta"] = {"model_name": "test_model", "threshold": 0.5}
        yield c


@pytest.fixture
def client_without_model(client):
    STATE["model"] = None
    return client


# --------------------------------------------------------------------------- #
# Service endpoints
# --------------------------------------------------------------------------- #
def test_root(client):
    body = client.get("/").json()
    assert body["docs"] == "/docs"
    assert "/predict" in body["endpoints"]


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model_name": "test_model"}


def test_health_reports_missing_model(client_without_model):
    response = client_without_model.get("/health")
    assert response.status_code == 503
    assert response.json()["model_loaded"] is False


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["model_name"] == "test_model"


def test_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]


# --------------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------------- #
def test_predict_valid_patient(client, patient):
    response = client.post("/predict", json=patient)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == RESULT_KEYS
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability_disease"] <= 1
    assert body["model_name"] == "test_model"


def test_predict_without_optional_fields(client, patient):
    payload = {k: v for k, v in patient.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize(
    "change",
    [
        {"age": 200},          # out of range
        {"chol": -5},          # out of range
        {"cp": 9},             # not an allowed category
        {"sex": 3},            # not an allowed category
        {"age": "old"},        # wrong type
        {"weight": 80},        # unknown field
    ],
)
def test_predict_rejects_invalid_input(client, patient, change):
    response = client.post("/predict", json={**patient, **change})
    assert response.status_code == 422


def test_predict_rejects_missing_required_field(client, patient):
    payload = {k: v for k, v in patient.items() if k != "age"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_returns_503_without_model(client_without_model, patient):
    assert client_without_model.post("/predict", json=patient).status_code == 503


# --------------------------------------------------------------------------- #
# /predict/batch
# --------------------------------------------------------------------------- #
def test_batch_prediction(client, patients):
    response = client.post("/predict/batch", json={"patients": patients})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == len(patients)
    assert all(set(p) == RESULT_KEYS for p in body["predictions"])


def test_batch_matches_single_predictions(client, patients):
    batch = client.post("/predict/batch", json={"patients": patients}).json()["predictions"]
    singles = [client.post("/predict", json=p).json() for p in patients]
    assert batch == singles


def test_batch_rejects_empty_list(client):
    assert client.post("/predict/batch", json={"patients": []}).status_code == 422


def test_batch_rejects_more_than_100(client, patient):
    response = client.post("/predict/batch", json={"patients": [patient] * 101})
    assert response.status_code == 422
