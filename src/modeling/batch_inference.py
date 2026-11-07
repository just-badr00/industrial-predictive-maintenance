"""Request current test-engine health from the existing prediction API."""

# NEW: Airflow remains an API client; model serving stays in maintenance-api.
from datetime import datetime, timezone
import time
import json
import os
from pathlib import Path

import pandas as pd
import requests

from src.api.schemas import BatchPredictionRequest, PredictionResponse
from src.ingestion.raw_ingestion import (
    create_minio_client, read_layer_data, read_manifest, update_manifest, upload_parquet,
)
from src.modeling.predict import health_status


def build_engine_requests(test_dataframe):
    observation_columns = (
        ["time_in_cycles"]
        + [f"operational_setting_{i}" for i in range(1, 4)]
        + [f"sensor_measurement_{i}" for i in range(1, 22)]
    )
    engines = []
    for engine_id, history in test_dataframe.groupby("unit_number", sort=True):
        history = history.sort_values("time_in_cycles")
        engines.append({
            "engine_id": int(engine_id),
            "history": history.loc[:, observation_columns].to_dict("records"),
        })
    # Reuse exactly the API's validation, including all cycles starting at 1.
    return BatchPredictionRequest.model_validate({"engines": engines}).model_dump()["engines"]


def check_api_identity(api_url, expected_model_info):
    response = requests.get(f"{api_url}/model-info", timeout=(10, 30))
    response.raise_for_status()
    actual = response.json()
    if not isinstance(actual, dict):
        raise ValueError("The API returned invalid model identity data.")
    for key, expected in expected_model_info.items():
        if actual.get(key) != expected:
            raise ValueError(f"API model identity does not match the serving bundle: {key}.")


def request_predictions(engines, api_url, expected_model_info, batch_size=10,time_per_batch=None):
    
    if batch_size < 1 or not engines:
        raise ValueError("Prediction requires a positive batch_size and at least one engine.")
    check_api_identity(api_url, expected_model_info)
    predictions = []
    for start in range(0, len(engines), batch_size):
        batch = engines[start:start + batch_size]
        started_at = time.perf_counter()
        response = requests.post(
            f"{api_url}/predict/batch", json={"engines": batch}, timeout=(10, 180)
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        response.raise_for_status()
        results = response.json()
        if not isinstance(results, list) or len(results) != len(batch):
            raise ValueError("The API returned an incomplete prediction batch.")
        for engine, result in zip(batch, results):
            prediction = PredictionResponse.model_validate(result)
            if prediction.engine_id != engine["engine_id"]:
                raise ValueError("API engine/result order does not match the request.")
            if prediction.cycle != engine["history"][-1]["time_in_cycles"]:
                raise ValueError("The API predicted a different engine cycle.")
            if prediction.model_version != expected_model_info["model_version"]:
                raise ValueError("The API model version changed during prediction.")
            if prediction.status != health_status(prediction.predicted_rul):
                raise ValueError("API health status does not match the RUL thresholds.")
            predictions.append(prediction.model_dump())
        if time_per_batch is not None:
            time_per_batch.append(elapsed_ms)
        print(f"Predicted {len(predictions)}/{len(engines)} test engines.", flush=True)
    check_api_identity(api_url, expected_model_info)
    return predictions


def prediction_table(predictions, metadata, model_info):
    recorded_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for prediction in predictions:
        row = {key: value for key, value in prediction.items() if key != "top_factors"}
        # A string column keeps nested SHAP factors portable in Parquet/Streamlit.
        row["top_factors_json"] = json.dumps(prediction["top_factors"], allow_nan=False)
        row.update({
            "pipeline_run_id": metadata["run_id"], 
            "prediction_time": recorded_at,
            "model_name": model_info["model_name"],
            "source_run_id": model_info["source_run_id"],
            "model_gold_run_id": model_info["gold_run_id"],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def save_health_dataset(s3_client, bucket_name, metadata, dataset_name, dataframe):
    object_key = (
        f"gold/{dataset_name}/load_date={metadata['load_date']}/"
        f"run_id={metadata['run_id']}/{dataset_name}.parquet"
    )
    print(upload_parquet(s3_client, bucket_name, object_key, dataframe))
    record = {
        "run_id": metadata["run_id"], "load_date": metadata["load_date"],
        "layer": "gold", "dataset_name": dataset_name, "object_key": object_key,
        "status": "success", "row_count": len(dataframe),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest = read_manifest(s3_client, bucket_name, metadata)
    update_manifest(s3_client, bucket_name, metadata, [record], manifest)


def predict_current_health(metadata, project_root="/opt/cmapss", api_url=None, batch_size=10):
    api_url = (api_url or os.environ.get("PREDICTION_API_URL", "http://host.docker.internal:8001")).rstrip("/")
    with open(Path(project_root) / "output/serving_model/model_info.json", encoding="utf-8") as file:
        model_info = json.load(file)
    s3_client = create_minio_client()
    bucket_name = os.environ["MINIO_BUCKET"]
    test_dataframe = read_layer_data(
        s3_client, bucket_name, metadata, "silver", dataset_names=("test_data",)
    )["test_data"]
    test_features = read_layer_data(
        s3_client, bucket_name, metadata, "gold", dataset_names=("test_features",)
    )["test_features"]
    # Enforce the same fleet and current cycles as the Gold data just validated.
    keys = ["unit_number", "time_in_cycles"]
    expected_keys = test_dataframe[keys].sort_values(keys).reset_index(drop=True)
    actual_keys = test_features[keys].sort_values(keys).reset_index(drop=True)
    if not expected_keys.equals(actual_keys):
        raise ValueError("Silver prediction histories and Gold engine/cycle rows differ.")
    engines = build_engine_requests(test_dataframe)
    time_per_batch = []
    predictions = request_predictions(engines, api_url, model_info, batch_size,time_per_batch=time_per_batch)
    dataframe = prediction_table(predictions, metadata, model_info)
    # No persistent predictions are published until every batch passes validation.
    save_health_dataset(s3_client, bucket_name, metadata, "maintenance_predictions", dataframe)
    print(f"Saved {len(dataframe)} current-engine predictions for run {metadata['run_id']}.")
    result_metadata = metadata.copy()
    result_metadata["prediction_timing"] = {
        "batch_request_count": len(time_per_batch),
        "mean_batch_request_ms": sum(time_per_batch) / len(time_per_batch),
        "max_batch_request_ms": max(time_per_batch),
    }
    return result_metadata