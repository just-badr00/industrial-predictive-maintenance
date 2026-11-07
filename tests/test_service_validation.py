"""Remaining API validation and logging checks, without loading MLflow artifacts."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.api.main as api


EXAMPLES = Path(__file__).resolve().parents[1] / "docs" / "examples"


@pytest.fixture
def payload():
    return json.loads((EXAMPLES / "predict_engine_1.json").read_text(encoding="utf-8"))


@pytest.fixture
def client(monkeypatch):
    def fake_loader():
        return {
            "model": object(),
            "model_name": "test-model",
            "model_version": "test",
            "source_run_id": "test-source",
            "gold_run_id": "test-gold",
            "feature_columns": ["time_in_cycles", "sensor_measurement_7_%change"],
        }

    monkeypatch.setattr(api, "load_registered_model", fake_loader)
    with TestClient(api.app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_model_info_reports_loaded_identity(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json() == {
        "model_name": "test-model",
        "model_version": "test",
        "source_run_id": "test-source",
        "gold_run_id": "test-gold",
    }


@pytest.mark.parametrize(
    "case",
    ["nonpositive_id", "string_id", "short_history", "missing_cycle", "extra_field"],
)
def test_invalid_requests_stop_before_feature_preparation(client, monkeypatch, payload, case):
    calls = []

    def forbidden_preparation(*args, **kwargs):
        calls.append(True)
        raise AssertionError("Invalid request reached feature preparation")

    monkeypatch.setattr(api, "prepare_model_input", forbidden_preparation)
    if case == "nonpositive_id":
        payload["engine_id"] = 0
    elif case == "string_id":
        payload["engine_id"] = "1"
    elif case == "short_history":
        payload["history"] = payload["history"][:19]
    elif case == "missing_cycle":
        del payload["history"][5]
    elif case == "extra_field":
        payload["history"][0]["unknown_sensor"] = 1.0

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]
    assert calls == []


@pytest.mark.parametrize("path", ["/predict", "/predict/batch"])
def test_undefined_feature_returns_422(client, payload, path):
    payload["history"][-2]["sensor_measurement_7"] = 0.0
    if path.endswith("/batch"):
        # The first engine is valid; the second must reject the entire batch.
        valid = json.loads((EXAMPLES / "predict_engine_1.json").read_text(encoding="utf-8"))
        payload["engine_id"] = 2
        body = {"engines": [valid, payload]}
    else:
        body = payload

    response = client.post(path, json=body)
    assert response.status_code == 422
    assert "undefined percentage changes" in response.json()["detail"]


def test_empty_batch_is_rejected(client):
    response = client.post("/predict/batch", json={"engines": []})
    assert response.status_code == 422
    assert any(error["loc"] == ["body", "engines"] for error in response.json()["detail"])


@pytest.mark.parametrize("path", ["/model-info", "/predict", "/predict/batch"])
def test_model_unavailable_returns_503(client, monkeypatch, payload, path):
    monkeypatch.setattr(api.app.state, "model_data", None)
    if path == "/model-info":
        response = client.get(path)
    else:
        body = {"engines": [payload]} if path.endswith("/batch") else payload
        response = client.post(path, json=body)
    assert response.status_code == 503
    assert response.json() == {"detail": "The prediction model is unavailable."}


@pytest.mark.parametrize("status", [200, 422, 503, 500])
def test_request_logging_and_error_propagation(client, monkeypatch, caplog, payload, status):
    # This app logger does not propagate, so attach pytest's capture handler directly.
    api.logger.addHandler(caplog.handler)
    try:
        if status == 503:
            monkeypatch.setattr(api.app.state, "model_data", None)
        if status == 500:
            def broken_preparation(*args, **kwargs):
                raise RuntimeError("Controlled test failure")

            monkeypatch.setattr(api, "prepare_model_input", broken_preparation)

        if status in (200, 503):
            response = client.get("/health")
        elif status == 422:
            response = client.post("/predict", json={})
        else:
            response = client.post("/predict", json=payload)

        assert response.status_code == status
        message = "Request failed unexpectedly" if status == 500 else "Request completed"
        records = [record for record in caplog.records if record.getMessage() == message]
        assert len(records) == 1
        record = records[0]
        assert record.details["status_code"] == status
        assert record.details["duration_ms"] >= 0
        assert len(record.details["request_id"]) == 32
        assert record.details["model_version"] == (None if status == 503 else "test")
        if status == 500:
            assert record.levelname == "ERROR"
            assert record.exc_info
            assert response.text == "Internal Server Error"
            assert "Controlled test failure" not in response.text
        else:
            assert response.headers["X-Request-ID"] == record.details["request_id"]
    finally:
        api.logger.removeHandler(caplog.handler)
