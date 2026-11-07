"""Check Gold lineage/structure and the saved model's input contract."""

# NEW: Validate complete Gold history, then eligible rows in trained feature order.
import json
import os
from pathlib import Path

import numpy as np
from pandas.api.types import is_numeric_dtype, is_bool_dtype

from src.ingestion.raw_ingestion import create_minio_client, read_layer_data


def validate_feature_frames(gold_dataframes, silver_dataframes, feature_columns):
    if not feature_columns or len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Saved model features must be nonempty and unique.")
    if {"unit_number", "remaining_useful_life"}.intersection(feature_columns):
        raise ValueError("Engine ID and RUL target must not be model inputs.")
    base_columns = (
        ["unit_number", "time_in_cycles"]
        + [f"operational_setting_{i}" for i in range(1, 4)]
        + [f"current_sensor_measurement_{i}" for i in range(1, 22)]
    )
    engineered_columns = [
        f"sensor_measurement_{i}{suffix}"
        for suffix in ("_mean_20", "_std_20", "_min_20", "_max_20", "_difference",
                       "_%change", "_sensor_slope_20", "_ema_20")
        for i in range(1, 22)
    ]
    summary = {}
    for dataset_name, source_name in (
        ("train_features", "train_data"), ("test_features", "test_data")
    ):
        dataframe = gold_dataframes[dataset_name]
        expected_columns = base_columns + engineered_columns
        if dataset_name == "train_features":
            expected_columns = expected_columns + ["remaining_useful_life"]
        if list(dataframe.columns) != expected_columns:
            raise ValueError(f"Wrong Gold schema: {dataset_name}.")
        if dataframe.empty or any(
            not is_numeric_dtype(dataframe[column]) or is_bool_dtype(dataframe[column])
            for column in expected_columns
        ):
            raise ValueError(f"{dataset_name} must contain nonempty numeric data.")
        if np.isinf(dataframe.to_numpy(dtype=float)).any():
            raise ValueError(f"{dataset_name} contains infinite values.")
        source = silver_dataframes[source_name].sort_values(
            ["unit_number", "time_in_cycles"]
        ).reset_index(drop=True)
        keys = ["unit_number", "time_in_cycles"]
        if dataframe.duplicated(keys).any() or not dataframe[keys].reset_index(drop=True).equals(source[keys]):
            raise ValueError(f"{dataset_name} engine/cycle rows do not match Silver.")
        renamed_source = source.rename(columns={
            f"sensor_measurement_{i}": f"current_sensor_measurement_{i}" for i in range(1, 22)
        })
        if not dataframe[base_columns].reset_index(drop=True).equals(renamed_source[base_columns]):
            raise ValueError(f"{dataset_name} changed original measurements.")
        if dataset_name == "train_features":
            expected_rul = source.groupby("unit_number")["time_in_cycles"].transform("max") - source["time_in_cycles"]
            if not np.array_equal(dataframe["remaining_useful_life"].to_numpy(), expected_rul.to_numpy()):
                raise ValueError("Training RUL does not match final cycle minus current cycle.")
        missing_features = set(feature_columns) - set(dataframe.columns)
        if missing_features:
            raise ValueError(f"Missing trained model features: {sorted(missing_features)}")
        eligible = dataframe.loc[dataframe["time_in_cycles"].ge(20)]
        if set(eligible["unit_number"]) != set(dataframe["unit_number"]):
            raise ValueError(f"{dataset_name} has an engine with fewer than 20 cycles.")
        # Extra/removed constant features can contain undefined percentage changes.
        # Only the saved model inputs must be finite on every eligible row.
        model_input = eligible.loc[:, feature_columns]
        if not np.isfinite(model_input.to_numpy(dtype=float)).all():
            raise ValueError(f"{dataset_name} has missing/non-finite eligible model features.")
        summary[dataset_name] = {
            "row_count": len(dataframe), "eligible_rows": len(eligible),
            "model_feature_count": len(feature_columns),
        }
    return summary


def validate_features(metadata, project_root="/opt/cmapss"):
    feature_metadata_path = Path(project_root) / "output" / "serving_model" / "final_training_features.json"
    with open(feature_metadata_path, "r", encoding="utf-8") as file:
        feature_columns = json.load(file)["feature_columns"]
    s3_client = create_minio_client()
    bucket_name = os.environ["MINIO_BUCKET"]
    gold_dataframes = read_layer_data(
        s3_client, bucket_name, metadata, "gold",
        dataset_names=("train_features", "test_features"),
    )
    silver_dataframes = read_layer_data(s3_client, bucket_name, metadata, "silver")
    summary = validate_feature_frames(gold_dataframes, silver_dataframes, feature_columns)
    print(f"Gold validation passed for run {metadata['run_id']}: {summary}")
    return metadata.copy()
