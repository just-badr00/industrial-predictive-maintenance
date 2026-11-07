"""Create run-scoped Silver objects without changing stored Raw objects."""

# NEW: Extracted Silver cleaning, uploads and manifest update from Notebook 02.
from datetime import datetime, timezone
import os

from src.ingestion.raw_ingestion import (
    create_minio_client, read_layer_data, read_manifest,
    update_manifest, upload_parquet,
)
from src.validation.raw_validation import validate_raw_frames


def prepare_silver_data(raw_dataframes):
    # Also protects direct/manual calls that bypass the validation task.
    validate_raw_frames(raw_dataframes)
    silver_dataframes = {}
    processing_timestamp = datetime.now(timezone.utc).isoformat()
    for dataset_name, dataframe in raw_dataframes.items():
        silver_dataframe = dataframe.drop_duplicates().copy()
        sort_columns = ["unit_number"]
        if dataset_name != "RUL_data":
            sort_columns.append("time_in_cycles")
        silver_dataframe = silver_dataframe.sort_values(sort_columns).reset_index(drop=True)
        silver_dataframe["processing_timestamp"] = processing_timestamp
        silver_dataframe["data_quality_status"] = "passed_silver_validation"
        silver_dataframes[dataset_name] = silver_dataframe
    return silver_dataframes


def transform_to_silver(metadata):
    s3_client = create_minio_client()
    bucket_name = os.environ["MINIO_BUCKET"]
    raw_dataframes = read_layer_data(s3_client, bucket_name, metadata, "raw")
    silver_dataframes = prepare_silver_data(raw_dataframes)
    silver_metadata_records = []
    for dataset_name, dataframe in silver_dataframes.items():
        object_key = (
            f"silver/{dataset_name}/load_date={metadata['load_date']}/"
            f"run_id={metadata['run_id']}/{dataset_name}.parquet"
        )
        print(upload_parquet(s3_client, bucket_name, object_key, dataframe))
        silver_metadata_records.append({
            "run_id": metadata["run_id"],
            "load_date": metadata["load_date"],
            "layer": "silver",
            "dataset_name": dataset_name,
            "object_key": object_key,
            "status": "success",
            "row_count": len(dataframe),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })

    # Read immediately before publishing to preserve records from other stages.
    manifest = read_manifest(s3_client, bucket_name, metadata)
    update_manifest(s3_client, bucket_name, metadata, silver_metadata_records, manifest)
    return metadata.copy()
