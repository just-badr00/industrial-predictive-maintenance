"""Offline checks: object storage is simulated in memory, never real MinIO."""

from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ClientError

from src.ingestion import raw_ingestion as raw
from src.validation import raw_validation as validation
from src.transformations import raw_to_silver as silver
from src.transformations import silver_to_features as gold
from src.validation import feature_validation as feature_checks
from src.modeling import batch_inference as inference
from src.transformations import publish_health as publication


@pytest.fixture
def frames():
    rows = []
    for engine in (1, 2):
        for cycle in (1, 2, 3):
            rows.append({
                "unit_number": engine, "time_in_cycles": cycle,
                **{f"operational_setting_{i}": float(i) for i in range(1, 4)},
                **{f"sensor_measurement_{i}": float(i + cycle) for i in range(1, 22)},
            })
    return {
        "train_data": pd.DataFrame(rows),
        "test_data": pd.DataFrame(rows),
        "RUL_data": pd.DataFrame({"unit_number": [1, 2], "remaining_useful_life": [0, 5]}),
    }


class MemoryStorage:
    def __init__(self):
        self.objects = {}
        self.writes = []
        self.fail_key = None

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, ContentType):
        if self.fail_key and self.fail_key in Key:
            raise RuntimeError("simulated upload failure")
        self.objects[Key] = Body
        self.writes.append(Key)


@pytest.fixture
def storage(monkeypatch, frames):
    client = MemoryStorage()
    monkeypatch.setenv("MINIO_BUCKET", "offline-test")
    for module in (raw, validation, silver):
        monkeypatch.setattr(module, "create_minio_client", lambda: client)
    monkeypatch.setattr(raw, "read_source_data", lambda paths: frames)
    return client


def test_cleaning_removes_only_exact_duplicates_and_preserves_raw(frames):
    frames["train_data"] = pd.concat(
        [frames["train_data"], frames["train_data"].iloc[[0]]], ignore_index=True
    )
    original = {name: frame.copy(deep=True) for name, frame in frames.items()}
    summary = validation.validate_raw_frames(frames)
    result = silver.prepare_silver_data(frames)
    assert summary["train_data"]["exact_duplicates"] == 1
    assert len(result["train_data"]) == 6
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(frame, original[name])
        assert result[name]["data_quality_status"].eq("passed_silver_validation").all()
        assert "processing_timestamp" in result[name]


@pytest.mark.parametrize("problem", [
    "schema", "empty", "missing", "infinity", "text", "fractional_id",
    "negative_id", "gap", "conflict", "negative_rul", "rul_alignment",
])
def test_invalid_raw_rejected(frames, problem):
    frame = frames["train_data"]
    if problem == "schema":
        frames["train_data"] = frame.drop(columns="sensor_measurement_1")
    elif problem == "empty":
        frames["train_data"] = frame.iloc[:0]
    elif problem == "missing":
        frame.loc[0, "sensor_measurement_1"] = np.nan
    elif problem == "infinity":
        frame.loc[0, "sensor_measurement_1"] = np.inf
    elif problem == "text":
        frame["sensor_measurement_1"] = "not numeric"
    elif problem == "fractional_id":
        frame["unit_number"] = frame["unit_number"].astype(float) + 0.5
    elif problem == "negative_id":
        frame.loc[0, "unit_number"] = -1
    elif problem == "gap":
        frames["train_data"] = frame.drop(index=1)
    elif problem == "conflict":
        conflicting = frame.iloc[[0]].copy()
        conflicting["sensor_measurement_1"] += 10
        frames["train_data"] = pd.concat([frame, conflicting], ignore_index=True)
    elif problem == "negative_rul":
        frames["RUL_data"].loc[0, "remaining_useful_life"] = -1
    else:
        frames["RUL_data"] = frames["RUL_data"].iloc[::-1]
    with pytest.raises((ValueError, TypeError)):
        validation.validate_raw_frames(frames)


def test_offline_pipeline_and_reruns_preserve_other_layers(storage):
    metadata = raw.ingest_raw_data({}, "20261025T120000000000Z", "2026-10-25")
    assert validation.validate_raw_data(metadata) == metadata
    assert silver.transform_to_silver(metadata) == metadata
    manifest = raw.read_manifest(storage, "offline-test", metadata)
    gold_record = manifest.iloc[[0]].copy()
    gold_record["layer"] = "gold"
    gold_record["dataset_name"] = "rul_training_features"
    raw.update_manifest(storage, "offline-test", metadata, gold_record.to_dict("records"), manifest)
    raw_bytes = {key: value for key, value in storage.objects.items() if key.startswith("raw/")}
    silver.transform_to_silver(metadata)
    assert all(storage.objects[key] == value for key, value in raw_bytes.items())
    raw.ingest_raw_data({}, metadata["run_id"], metadata["load_date"])
    manifest = raw.read_manifest(storage, "offline-test", metadata)
    assert manifest["layer"].value_counts().to_dict() == {"raw": 3, "silver": 3, "gold": 1}
    assert not manifest.duplicated(["run_id", "layer", "dataset_name"]).any()


def test_partial_raw_upload_does_not_publish_manifest(storage):
    storage.fail_key = "raw/RUL_data/"
    with pytest.raises(RuntimeError, match="simulated upload failure"):
        raw.ingest_raw_data({}, "test-run", "2026-10-25")
    assert not any(key.endswith("manifest.parquet") for key in storage.objects)


def test_partial_silver_upload_preserves_raw_manifest(storage):
    metadata = raw.ingest_raw_data({}, "test-run", "2026-10-25")
    before = storage.objects[metadata["metadata_key"]]
    storage.fail_key = "silver/RUL_data/"
    with pytest.raises(RuntimeError, match="simulated upload failure"):
        silver.transform_to_silver(metadata)
    assert storage.objects[metadata["metadata_key"]] == before


def test_changed_raw_rejected_after_silver_without_writes(storage, frames):
    metadata = raw.ingest_raw_data({}, "test-run", "2026-10-25")
    silver.transform_to_silver(metadata)
    writes_before = len(storage.writes)
    frames["train_data"].loc[0, "sensor_measurement_1"] += 1
    with pytest.raises(ValueError, match="Start a new DAG run"):
        raw.ingest_raw_data({}, metadata["run_id"], metadata["load_date"])
    assert len(storage.writes) == writes_before


def test_manifest_wrong_run_row_count_and_ambiguity_rejected(storage):
    metadata = raw.ingest_raw_data({}, "test-run", "2026-10-25")
    with pytest.raises(ValueError, match="different pipeline run ID"):
        validation.validate_raw_data({**metadata, "run_id": "other-run"})
    manifest = raw.read_manifest(storage, "offline-test", metadata)
    manifest.loc[0, "row_count"] += 1
    raw.upload_parquet(storage, "offline-test", metadata["metadata_key"], manifest)
    with pytest.raises(ValueError, match="row count"):
        validation.validate_raw_data(metadata)
    manifest.loc[0, "row_count"] -= 1
    duplicate = pd.concat([manifest, manifest.iloc[[0]]], ignore_index=True)
    raw.upload_parquet(storage, "offline-test", metadata["metadata_key"], duplicate)
    with pytest.raises(ValueError, match="Expected one successful"):
        validation.validate_raw_data(metadata)


# NEW: Gold equivalence, causality, model-input checks and manifest integration.
@pytest.fixture
def feature_frames(frames):
    for name in ("train_data", "test_data"):
        rows = []
        for engine in (1, 2):
            for cycle in range(1, 26):
                row = frames[name].iloc[0].to_dict()
                row.update({"unit_number": engine, "time_in_cycles": cycle})
                row.update({
                    f"sensor_measurement_{i}": float(i + engine * 10 + cycle)
                    for i in range(1, 22)
                })
                rows.append(row)
        frames[name] = pd.DataFrame(rows).astype({"unit_number": int, "time_in_cycles": int})
    return frames


@pytest.fixture
def feature_columns():
    return ["time_in_cycles", "current_sensor_measurement_2",
            "sensor_measurement_3_ema_20", "sensor_measurement_4_mean_20",
            "sensor_measurement_2_%change"]


def test_gold_matches_notebook05_and_does_not_mutate_silver(feature_frames):
    import ast
    import json
    from pathlib import Path

    silver_frames = silver.prepare_silver_data(feature_frames)
    original = {name: frame.copy(deep=True) for name, frame in silver_frames.items()}
    result = gold.prepare_gold_features(silver_frames)
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks" /
                           "05_Feature engineering_Final.ipynb").read_text(encoding="utf-8"))
    namespace = {"pd": pd, "np": np}
    for cell_number in (4, 5):
        tree = ast.parse("".join(notebook["cells"][cell_number]["source"]))
        functions = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)], type_ignores=[])
        exec(compile(functions, "notebook05-functions", "exec"), namespace)
    for source_name, dataset_name in (("train_data", "train_features"), ("test_data", "test_features")):
        expected = namespace["prepare_feature_input"](silver_frames[source_name])
        sensors = [column for column in expected if column.startswith("current_sensor_")]
        expected = namespace["add_time_series_features"](expected, sensors)
        if source_name == "train_data":
            last = expected.groupby("unit_number").agg(last_cycle=("time_in_cycles", "max"))
            expected = expected.merge(last, on="unit_number", how="left")
            expected["remaining_useful_life"] = expected["last_cycle"] - expected["time_in_cycles"]
            expected = expected.drop(columns="last_cycle")
        pd.testing.assert_frame_equal(result[dataset_name], expected)
    for name in original:
        pd.testing.assert_frame_equal(silver_frames[name], original[name])


def test_gold_features_are_causal_and_grouped(feature_frames):
    baseline = gold.prepare_gold_features(silver.prepare_silver_data(feature_frames))
    changed = {name: frame.copy(deep=True) for name, frame in feature_frames.items()}
    mask = changed["test_data"]["unit_number"].eq(1) & changed["test_data"]["time_in_cycles"].eq(25)
    changed["test_data"].loc[mask, "sensor_measurement_3"] += 1000
    updated = gold.prepare_gold_features(silver.prepare_silver_data(changed))
    unaffected = ~mask
    pd.testing.assert_frame_equal(baseline["test_features"].loc[unaffected], updated["test_features"].loc[unaffected])


@pytest.mark.parametrize("problem", ["missing_feature", "eligible_nan", "infinity", "wrong_rul", "lost_row", "changed_sensor", "label_in_test"])
def test_gold_validation_rejects_corruption(feature_frames, feature_columns, problem):
    silver_frames = silver.prepare_silver_data(feature_frames)
    result = gold.prepare_gold_features(silver_frames)
    dataframe = result["train_features"]
    if problem == "missing_feature":
        result["train_features"] = dataframe.drop(columns="sensor_measurement_3_ema_20")
    elif problem == "eligible_nan":
        dataframe.loc[19, "sensor_measurement_3_ema_20"] = np.nan
    elif problem == "infinity":
        dataframe.loc[0, "sensor_measurement_3_ema_20"] = np.inf
    elif problem == "wrong_rul":
        dataframe.loc[0, "remaining_useful_life"] += 1
    elif problem == "lost_row":
        result["train_features"] = dataframe.iloc[1:]
    elif problem == "changed_sensor":
        dataframe.loc[0, "current_sensor_measurement_2"] += 1
    else:
        result["test_features"]["remaining_useful_life"] = 1
    with pytest.raises(ValueError):
        feature_checks.validate_feature_frames(result, silver_frames, feature_columns)


def test_gold_validation_accepts_early_window_nans(feature_frames, feature_columns):
    silver_frames = silver.prepare_silver_data(feature_frames)
    result = gold.prepare_gold_features(silver_frames)
    assert result["train_features"].iloc[:19]["sensor_measurement_4_mean_20"].isna().all()
    summary = feature_checks.validate_feature_frames(result, silver_frames, feature_columns)
    assert summary["train_features"]["eligible_rows"] == 12
    with pytest.raises(ValueError, match="must not be model inputs"):
        feature_checks.validate_feature_frames(result, silver_frames, ["unit_number"])


def test_gold_manifest_rerun_and_validation(storage, feature_frames, feature_columns, monkeypatch):
    monkeypatch.setattr(gold, "create_minio_client", lambda: storage)
    metadata = raw.ingest_raw_data({}, "gold-run", "2026-10-26")
    silver.transform_to_silver(metadata)
    original_manifest = raw.read_manifest(storage, "offline-test", metadata)
    extra = original_manifest.iloc[[0]].copy()
    extra["layer"] = "gold"
    extra["dataset_name"] = "rul_training_features"
    raw.update_manifest(storage, "offline-test", metadata, extra.to_dict("records"), original_manifest)
    assert gold.generate_features(metadata) == metadata
    gold.generate_features(metadata)
    manifest = raw.read_manifest(storage, "offline-test", metadata)
    assert manifest["layer"].value_counts().to_dict() == {"raw": 3, "silver": 3, "gold": 3}
    assert not manifest.duplicated(["run_id", "layer", "dataset_name"]).any()
    loaded_gold = raw.read_layer_data(storage, "offline-test", metadata, "gold",
                                     dataset_names=("train_features", "test_features"))
    loaded_silver = raw.read_layer_data(storage, "offline-test", metadata, "silver")
    feature_checks.validate_feature_frames(loaded_gold, loaded_silver, feature_columns)


def test_partial_gold_upload_does_not_publish_success(storage, feature_frames, monkeypatch):
    monkeypatch.setattr(gold, "create_minio_client", lambda: storage)
    metadata = raw.ingest_raw_data({}, "gold-run", "2026-10-26")
    silver.transform_to_silver(metadata)
    before = storage.objects[metadata["metadata_key"]]
    storage.fail_key = "gold/test_features/"
    with pytest.raises(RuntimeError, match="simulated upload failure"):
        gold.generate_features(metadata)
    assert storage.objects[metadata["metadata_key"]] == before


# NEW: Prediction API boundary and durable dashboard publication, fully offline.
class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


@pytest.fixture
def fake_api(monkeypatch):
    import json
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    model_info = json.loads((project_root / "output/serving_model/model_info.json").read_text())
    calls = []

    def post(url, json, timeout):
        calls.append(json)
        return FakeResponse([
            {"engine_id": engine["engine_id"], "cycle": engine["history"][-1]["time_in_cycles"],
             "predicted_rul": 9.0 if engine["engine_id"] == 1 else 40.0,
             "status": "critical" if engine["engine_id"] == 1 else "healthy",
             "model_version": model_info["model_version"],
             "top_factors": [{"feature": "time_in_cycles", "value": 25.0, "shap_value": -2.0}]}
            for engine in json["engines"]
        ])

    monkeypatch.setattr(inference.requests, "get", lambda *args, **kwargs: FakeResponse(model_info))
    monkeypatch.setattr(inference.requests, "post", post)
    return project_root, model_info, calls, post


def test_prediction_requests_include_full_history_but_no_metadata_or_labels(feature_frames):
    silver_frames = silver.prepare_silver_data(feature_frames)
    engines = inference.build_engine_requests(silver_frames["test_data"])
    assert [engine["engine_id"] for engine in engines] == [1, 2]
    for engine in engines:
        assert [row["time_in_cycles"] for row in engine["history"]] == list(range(1, 26))
        assert len(engine["history"][0]) == 25
        assert not {"unit_number", "processing_timestamp", "remaining_useful_life"}.intersection(engine["history"][0])


@pytest.mark.parametrize("problem", ["incomplete", "wrong_engine", "wrong_cycle", "wrong_version", "wrong_status", "nonfinite", "malformed"])
def test_invalid_api_predictions_rejected(feature_frames, fake_api, monkeypatch, problem):
    _, model_info, _, original_post = fake_api
    engines = inference.build_engine_requests(feature_frames["test_data"])

    def bad_post(*args, **kwargs):
        result = original_post(*args, **kwargs).json()
        if problem == "incomplete":
            return FakeResponse(result[:-1])
        if problem == "wrong_engine":
            result[0]["engine_id"] = 99
        elif problem == "wrong_cycle":
            result[0]["cycle"] = 20
        elif problem == "wrong_version":
            result[0]["model_version"] = "different"
        elif problem == "wrong_status":
            result[0]["status"] = "healthy"
        elif problem == "nonfinite":
            result[0]["predicted_rul"] = float("nan")
        else:
            del result[0]["top_factors"]
        return FakeResponse(result)

    monkeypatch.setattr(inference.requests, "post", bad_post)
    with pytest.raises(ValueError):
        inference.request_predictions(engines, "http://fake-api", model_info)


def test_model_identity_mismatch_rejected_before_prediction(feature_frames, fake_api, monkeypatch):
    _, model_info, calls, _ = fake_api
    monkeypatch.setattr(inference.requests, "get", lambda *a, **kw: FakeResponse({**model_info, "source_run_id": "other"}))
    with pytest.raises(ValueError, match="model identity"):
        inference.request_predictions(inference.build_engine_requests(feature_frames["test_data"]), "http://fake-api", model_info)
    assert not calls


def test_prediction_and_publication_preserve_manifest_and_rerun(storage, feature_frames, fake_api, monkeypatch):
    project_root, _, calls, _ = fake_api
    for module in (gold, inference, publication):
        monkeypatch.setattr(module, "create_minio_client", lambda: storage)
    metadata = raw.ingest_raw_data({}, "prediction-run", "2026-10-27")
    silver.transform_to_silver(metadata)
    gold.generate_features(metadata)
    inference.predict_current_health(metadata, project_root, "http://fake-api", batch_size=1)
    assert len(calls) == 2
    publication.publish_dashboard_dataset(metadata)
    publication.publish_dashboard_dataset(metadata)
    manifest = raw.read_manifest(storage, "offline-test", metadata)
    assert len(manifest) == 10
    assert not manifest.duplicated(["run_id", "layer", "dataset_name"]).any()
    health = raw.read_layer_data(storage, "offline-test", metadata, "gold", dataset_names=("current_engine_health",))["current_engine_health"]
    assert health["engine_id"].tolist() == [1, 2]
    assert health["priority"].tolist() == ["High", "Low"]
    assert health["pipeline_run_id"].eq(metadata["run_id"]).all()


def test_failed_prediction_batch_publishes_nothing(storage, feature_frames, fake_api, monkeypatch):
    project_root, _, _, original_post = fake_api
    for module in (gold, inference):
        monkeypatch.setattr(module, "create_minio_client", lambda: storage)
    metadata = raw.ingest_raw_data({}, "failed-prediction", "2026-10-27")
    silver.transform_to_silver(metadata)
    gold.generate_features(metadata)
    before = storage.objects[metadata["metadata_key"]]
    batch_count = 0

    def fail_second_batch(*args, **kwargs):
        nonlocal batch_count
        batch_count += 1
        if batch_count == 2:
            raise inference.requests.Timeout("simulated timeout")
        return original_post(*args, **kwargs)

    monkeypatch.setattr(inference.requests, "post", fail_second_batch)
    with pytest.raises(inference.requests.Timeout):
        inference.predict_current_health(metadata, project_root, "http://fake-api", batch_size=1)
    assert storage.objects[metadata["metadata_key"]] == before
    assert not any(key.startswith("gold/maintenance_predictions/") for key in storage.objects)


@pytest.mark.parametrize("problem", ["missing_engine", "duplicate", "wrong_run", "wrong_cycle", "bad_factors"])
def test_publication_rejects_invalid_snapshots(feature_frames, fake_api, problem):
    _, model_info, _, _ = fake_api
    engines = inference.build_engine_requests(feature_frames["test_data"])
    predictions = inference.request_predictions(engines, "http://fake-api", model_info)
    metadata = {"run_id": "snapshot-run"}
    dataframe = inference.prediction_table(predictions, metadata, model_info)
    if problem == "missing_engine":
        dataframe = dataframe.iloc[:1]
    elif problem == "duplicate":
        dataframe = pd.concat([dataframe, dataframe.iloc[:1]], ignore_index=True)
    elif problem == "wrong_run":
        dataframe.loc[0, "pipeline_run_id"] = "other"
    elif problem == "wrong_cycle":
        dataframe.loc[0, "cycle"] = 22
    else:
        dataframe.loc[0, "top_factors_json"] = "invalid json"
    with pytest.raises(ValueError):
        publication.prepare_dashboard_health(dataframe, feature_frames["test_data"], metadata)
