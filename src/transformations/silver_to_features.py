"""Generate Notebook 05's full Gold feature tables from this run's Silver data."""

# NEW: Reuse the established feature mathematics without changing API inference.
from datetime import datetime, timezone
import os

from src.ingestion.raw_ingestion import (
    create_minio_client, read_layer_data, read_manifest, update_manifest, upload_parquet,
)
from src.transformations.feature_engineering import (
    prepare_feature_input, add_time_series_features,
)
from src.validation.raw_validation import validate_raw_frames


def prepare_gold_features(silver_dataframes):
    raw_columns = {}
    for dataset_name in ("train_data", "test_data", "RUL_data"):
        dataframe = silver_dataframes[dataset_name]
        if not dataframe["data_quality_status"].eq("passed_silver_validation").all():
            raise ValueError(f"{dataset_name} has not passed Silver validation.")
        raw_columns[dataset_name] = dataframe.drop(
            columns=["processing_timestamp", "data_quality_status"]
        ).copy()
    validate_raw_frames(raw_columns)
    gold_dataframes = {}
    for source_name, dataset_name in (
        ("train_data", "train_features"), ("test_data", "test_features")
    ):
        dataframe = prepare_feature_input(raw_columns[source_name])
        if dataframe.duplicated(["unit_number", "time_in_cycles"]).any():
            raise ValueError(f"Silver contains duplicate engine/cycle keys: {source_name}.")
        sensors = [
            column for column in dataframe.columns
            if column.startswith("current_sensor_measurement_")
        ]
        # Keep complete history during calculation: EMA must start at cycle 1.
        dataframe = add_time_series_features(dataframe, sensors)
        if dataset_name == "train_features":
            last_cycle = dataframe.groupby("unit_number")["time_in_cycles"].transform("max")
            dataframe["remaining_useful_life"] = last_cycle - dataframe["time_in_cycles"]
        gold_dataframes[dataset_name] = dataframe
    return gold_dataframes


def generate_features(metadata):
    s3_client = create_minio_client()
    bucket_name = os.environ["MINIO_BUCKET"]
    silver_dataframes = read_layer_data(s3_client, bucket_name, metadata, "silver")
    gold_dataframes = prepare_gold_features(silver_dataframes)
    records = []
    for dataset_name, dataframe in gold_dataframes.items():
        object_key = (
            f"gold/{dataset_name}/load_date={metadata['load_date']}/"
            f"run_id={metadata['run_id']}/{dataset_name}.parquet"
        )
        print(upload_parquet(s3_client, bucket_name, object_key, dataframe))
        print(f"{dataset_name}: {dataframe.shape}; early-window NaNs retained as in Notebook 05.")
        records.append({
            "run_id": metadata["run_id"], "load_date": metadata["load_date"],
            "layer": "gold", "dataset_name": dataset_name, "object_key": object_key,
            "status": "success", "row_count": len(dataframe),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })
    manifest = read_manifest(s3_client, bucket_name, metadata)
    update_manifest(s3_client, bucket_name, metadata, records, manifest)
    return metadata.copy()
