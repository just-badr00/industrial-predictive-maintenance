"""Publish this run's complete health snapshot for later dashboard consumption."""

# NEW: Durable, run-scoped dashboard data; no mutable latest pointer or drift claim.
import json
import os

import pandas as pd

from src.api.schemas import PredictionResponse
from src.ingestion.raw_ingestion import create_minio_client, read_layer_data
from src.modeling.batch_inference import save_health_dataset
from src.modeling.predict import health_status


def prepare_dashboard_health(predictions, test_dataframe, metadata):
    required = {
        "engine_id", "cycle", "predicted_rul", "status", "model_version",
        "top_factors_json", "pipeline_run_id", "prediction_time", "model_name",
        "source_run_id", "model_gold_run_id",
    }
    if set(predictions.columns) != required or predictions.empty:
        raise ValueError("The stored prediction table has an invalid schema or is empty.")
    if predictions.isna().any().any() or predictions.duplicated("engine_id").any():
        raise ValueError("Stored predictions contain missing values or duplicate engines.")
    if not predictions["pipeline_run_id"].eq(metadata["run_id"]).all():
        raise ValueError("Stored predictions belong to a different pipeline run.")
    for column in ("model_version", "model_name", "source_run_id", "model_gold_run_id"):
        if predictions[column].nunique() != 1:
            raise ValueError("A health snapshot must use one consistent model identity.")
    pd.to_datetime(predictions["prediction_time"], utc=True, errors="raise")

    for row in predictions.to_dict("records"):
        
        prediction = PredictionResponse.model_validate({
            key: row[key] for key in ("engine_id", "cycle", "predicted_rul", "status", "model_version")
        } | {"top_factors": json.loads(row["top_factors_json"])})

        if prediction.status != health_status(prediction.predicted_rul):
            raise ValueError("Stored prediction status does not match RUL thresholds.")
    expected = test_dataframe.groupby("unit_number")["time_in_cycles"].max().to_dict()
    actual = predictions.set_index("engine_id")["cycle"].to_dict()
    if actual != expected:
        raise ValueError("Prediction coverage does not match all current test engines/cycles.")
    health = predictions.copy()
    health["priority"] = health["status"].map({"critical": "High", "warning": "Medium", "healthy": "Low"})
    return health.sort_values(["predicted_rul", "engine_id"]).reset_index(drop=True)


def publish_dashboard_dataset(metadata):
    s3_client = create_minio_client()
    bucket_name = os.environ["MINIO_BUCKET"]
    predictions = read_layer_data(
        s3_client, bucket_name, metadata, "gold", dataset_names=("maintenance_predictions",)
    )["maintenance_predictions"]
    test_dataframe = read_layer_data(
        s3_client, bucket_name, metadata, "silver", dataset_names=("test_data",)
    )["test_data"]
    health = prepare_dashboard_health(predictions, test_dataframe, metadata)
    save_health_dataset(s3_client, bucket_name, metadata, "current_engine_health", health)
    print(f"Published {len(health)} engines: {health['status'].value_counts().to_dict()}")
    return metadata.copy()
