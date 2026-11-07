import json
from pathlib import Path
from fastapi.testclient import TestClient
import src.api.main as api

class FakeModel:
    def predict(self, model_input):
        return [17.4] * len(model_input)


def test_predict_returns_expected_response(monkeypatch):
    # A small test feature list; the real registered model uses 137 features.
    feature_columns = ["time_in_cycles", "current_sensor_measurement_7"]
    expected_factors = [
        {"feature": "time_in_cycles", "value": 31.0, "shap_value": 3.0},
    ]
    fake_explainer = object()

    def fake_model_loader():
        return {
            "model": FakeModel(),
            "model_name": "test-model",
            "model_version": "test",
            "feature_columns": feature_columns,
            "explainer": fake_explainer,
        }

    def fake_explain_predictions(explainer, model_input):
        # Check the endpoint passes the configured explainer and prepared inputs.
        assert explainer is fake_explainer
        assert model_input.columns.tolist() == feature_columns
        assert model_input.shape == (1, 2)
        assert model_input.iloc[0]["time_in_cycles"] == 31
        return [expected_factors]

    monkeypatch.setattr(api, "load_registered_model", fake_model_loader)
    monkeypatch.setattr(api, "explain_predictions", fake_explain_predictions)

    project_root = Path(__file__).resolve().parents[1]
    example_path = project_root / "docs" / "examples" / "predict_engine_1.json"
    data = json.loads(example_path.read_text(encoding="utf-8"))

    with TestClient(api.app) as client:
        response = client.post("/predict", json=data)

        assert response.status_code == 200
        assert response.json() == {
            "engine_id": 1,
            "cycle": 31,
            "predicted_rul": 17.4,
            "status": "warning",
            "top_factors": expected_factors,
            "model_version": "test",
        }

def test_batch_preserves_engine_prediction_and_factor_order(monkeypatch):
    feature_columns = ["time_in_cycles", "current_sensor_measurement_7"]
    expected_factors = [
        [{"feature": "time_in_cycles", "value": 126.0, "shap_value": -4.0}],
        [{"feature": "time_in_cycles", "value": 31.0, "shap_value": 3.0}],
        [{"feature": "time_in_cycles", "value": 49.0, "shap_value": -2.0}],
    ]
    calls = {"predict": 0, "explain": 0}

    class FakeBatchModel:
        def predict(self, model_input):
            calls["predict"] += 1
            assert model_input.columns.tolist() == feature_columns
            assert model_input["time_in_cycles"].tolist() == [126, 31, 49]
            return [45.0, 17.4, 5.0]

    fake_explainer = object()

    def fake_model_loader():
        return {
            "model": FakeBatchModel(),
            "model_name": "test-model",
            "model_version": "test",
            "feature_columns": feature_columns,
            "explainer": fake_explainer,
        }

    def fake_explain_predictions(explainer, model_input):
        calls["explain"] += 1
        assert explainer is fake_explainer
        assert model_input["time_in_cycles"].tolist() == [126, 31, 49]
        return expected_factors

    monkeypatch.setattr(api, "load_registered_model", fake_model_loader)
    monkeypatch.setattr(api, "explain_predictions", fake_explain_predictions)

    project_root = Path(__file__).resolve().parents[1]
    example_path = project_root / "docs" / "examples" / "predict_batch_engines_1_2_3.json"
    data = json.loads(example_path.read_text(encoding="utf-8"))
    # Deliberately use engine order 3, 1, 2, rather than sorted engine IDs.
    data["engines"] = [data["engines"][2], data["engines"][0], data["engines"][1]]

    with TestClient(api.app) as client:
        response = client.post("/predict/batch", json=data)

        assert response.status_code == 200
        results = response.json()
        assert len(results) == 3
        assert [result["engine_id"] for result in results] == [3, 1, 2]
        assert [result["cycle"] for result in results] == [126, 31, 49]
        assert [result["predicted_rul"] for result in results] == [45.0, 17.4, 5.0]
        assert [result["status"] for result in results] == ["healthy", "warning", "critical"]
        assert [result["top_factors"] for result in results] == expected_factors
        assert all(result["model_version"] == "test" for result in results)
        assert calls == {"predict": 1, "explain": 1}


def test_health_returns_ok(monkeypatch):
    # /health needs model data to exist, but it never makes a prediction.
    def fake_model_loader():
        return {
            "model": object(),
            "model_name": "test-model",
            "model_version": "test",
        }

    # Replace the loader only in this test; pytest restores it afterward.
    monkeypatch.setattr(api, "load_registered_model", fake_model_loader)

    # Entering this block runs startup; leaving it runs shutdown.
    with TestClient(api.app) as client:
        response = client.get("/health")

        # These are the results we expect from the real health endpoint.
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "message": "Predictive maintenance API is running",
            "model_loaded": True,
        }


def test_health_returns_503_when_model_unavailable(monkeypatch):
    def fake_model_loader():
        return {
            "model": object(),
            "model_name": "test-model",
            "model_version": "test",
        }

    monkeypatch.setattr(api, "load_registered_model", fake_model_loader)

    with TestClient(api.app) as client:
        # Startup has finished. Simulate model unavailability in this test app.
        monkeypatch.setattr(api.app.state, "model_data", None)

        response = client.get("/health")

        # A handled error is a response we can inspect, not a failed test.
        assert response.status_code == 503
        assert response.json() == {
            "detail": "The prediction model is unavailable."
        }


def test_predict_rejects_duplicate_cycles(monkeypatch):
    def fake_model_loader():
        return {
            "model": object(),
            "model_name": "test-model",
            "model_version": "test",
        }

    def unexpected_feature_preparation(*args, **kwargs):
        raise AssertionError("Invalid requests must be rejected before feature preparation.")

    monkeypatch.setattr(api, "load_registered_model", fake_model_loader)
    monkeypatch.setattr(api, "prepare_model_input", unexpected_feature_preparation)

    project_root = Path(__file__).resolve().parents[1]
    example_path = project_root / "docs" / "examples" / "predict_engine_1.json"
    data = json.loads(example_path.read_text(encoding="utf-8"))

    # Replace cycle 6 with 5, introducing a duplicate in an otherwise valid request.
    data["history"][5]["time_in_cycles"] = 5

    with TestClient(api.app) as client:
        response = client.post("/predict", json=data)

        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any(
            error["loc"] == ["body", "history"]
            and "Cycles must start at 1 and be ordered and consecutive." in error["msg"]
            for error in errors
        )


def test_predict_rejects_missing_sensor(monkeypatch):
    def fake_model_loader():
        return {
            "model": object(),
            "model_name": "test-model",
            "model_version": "test",
        }

    def unexpected_feature_preparation(*args, **kwargs):
        raise AssertionError("Invalid requests must be rejected before feature preparation.")

    monkeypatch.setattr(api, "load_registered_model", fake_model_loader)
    monkeypatch.setattr(api, "prepare_model_input", unexpected_feature_preparation)

    project_root = Path(__file__).resolve().parents[1]
    example_path = project_root / "docs" / "examples" / "predict_engine_1.json"
    data = json.loads(example_path.read_text(encoding="utf-8"))

    del data["history"][0]["sensor_measurement_7"]

    with TestClient(api.app) as client:
        response = client.post("/predict", json=data)

        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any(
            error["loc"] == ["body", "history", 0, "sensor_measurement_7"]
            and error["type"] == "missing"
            for error in errors
        )
