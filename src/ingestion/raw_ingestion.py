from datetime import datetime, timezone
import pandas as pd
import os
import boto3
from botocore.client import Config
from io import BytesIO
from botocore.exceptions import ClientError

def read_source_data(source_files):

    columns = [
    "unit_number" if i == 0
    else "time_in_cycles" if i == 1
    else f"operational_setting_{i-1}" if i <= 4
    else f"sensor_measurement_{i-4}"
    for i in range(26)
    ]

    columns_RUL = ["remaining_useful_life"]

    raw_df_train = pd.read_csv(source_files["train_FD001.txt"],sep=r'\s+',header=None,names=columns,index_col=False)
    raw_df_test = pd.read_csv(source_files["test_FD001.txt"],sep=r'\s+',header=None,names=columns,index_col=False)
    raw_df_RUL = pd.read_csv(source_files["RUL_FD001.txt"],sep=r'\s+',header=None,names=columns_RUL,index_col=False)

    a = raw_df_RUL.index + 1
    raw_df_RUL["unit_number"] = a
    raw_df_RUL = raw_df_RUL[["unit_number","remaining_useful_life"]]

    int_columns = ["unit_number","time_in_cycles"]
    numeric_columns = columns.copy()
    numeric_columns.remove("unit_number")
    numeric_columns.remove("time_in_cycles")
    for column in int_columns:
        raw_df_train[column] = raw_df_train[column].astype(int)
        raw_df_test[column] = raw_df_test[column].astype(int)
    for column in numeric_columns:
        raw_df_train[column] = raw_df_train[column].astype(float)
        raw_df_test[column] = raw_df_test[column].astype(float)

    return {
        "train_data": raw_df_train,
        "test_data": raw_df_test,
        "RUL_data": raw_df_RUL,
    }

def create_minio_client():

    s3_client = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )

    return s3_client

def upload_parquet(s3_client, bucket_name, object_key, dataframe):
    parquet_buffer = BytesIO()
    
    dataframe.to_parquet(
        parquet_buffer,
        index=False,
        engine="fastparquet",
    )

    parquet_buffer.seek(0)

    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=parquet_buffer.getvalue(),
        ContentType="application/octet-stream",
    )

    return f"Successfully uploaded {object_key} to bucket {bucket_name}."


# Read exact objects and preserve other stages when updating a run manifest.
def read_parquet_object(s3_client, bucket_name, object_key):
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    return pd.read_parquet(BytesIO(response["Body"].read()), engine="fastparquet")


def read_manifest(s3_client, bucket_name, metadata):
    manifest = read_parquet_object(s3_client, bucket_name, metadata["metadata_key"])
    required_columns = {
        "run_id", "load_date", "layer", "dataset_name", "object_key",
        "status", "row_count", "recorded_at",
    }
    if not required_columns.issubset(manifest.columns) or manifest.empty:
        raise ValueError("The run manifest is empty or missing required columns.")
    if not manifest["run_id"].eq(metadata["run_id"]).all():
        raise ValueError("The manifest contains a different pipeline run ID.")
    if not manifest["load_date"].eq(metadata["load_date"]).all():
        raise ValueError("The manifest contains a different load date.")
    return manifest


# NEW: Gold uses feature-table names; Raw/Silver keep their existing defaults.
def read_layer_data(s3_client, bucket_name, metadata, layer, dataset_names=None):
    manifest = read_manifest(s3_client, bucket_name, metadata)
    dataframes = {}
    if dataset_names is None:
        dataset_names = ("train_data", "test_data", "RUL_data")
    for dataset_name in dataset_names:
        matches = manifest[
            manifest["layer"].eq(layer)
            & manifest["dataset_name"].eq(dataset_name)
        ]
        if len(matches) != 1 or matches.iloc[0]["status"] != "success":
            raise ValueError(f"Expected one successful {layer} record for {dataset_name}.")
        record = matches.iloc[0]
        expected_prefix = (
            f"{layer}/{dataset_name}/load_date={metadata['load_date']}/"
            f"run_id={metadata['run_id']}/"
        )
        if not record["object_key"].startswith(expected_prefix):
            raise ValueError(f"Object key does not match the run: {dataset_name}.")
        dataframe = read_parquet_object(s3_client, bucket_name, record["object_key"])
        if len(dataframe) != record["row_count"]:
            raise ValueError(f"Manifest row count does not match {dataset_name}.")
        dataframes[dataset_name] = dataframe
    return dataframes


def update_manifest(s3_client, bucket_name, metadata, records, existing_manifest):
    # Replace only records owned by this stage; keep Raw/Silver/Gold neighbours.
    stage_records = pd.DataFrame(records)
    if existing_manifest.empty:
        manifest = stage_records
    else:
        replaced = (
            existing_manifest["run_id"].eq(metadata["run_id"])
            & existing_manifest["layer"].eq(stage_records["layer"].iloc[0])
            & existing_manifest["dataset_name"].isin(stage_records["dataset_name"])
        )
        manifest = pd.concat(
            [existing_manifest.loc[~replaced], stage_records], ignore_index=True
        )
    print(upload_parquet(s3_client, bucket_name, metadata["metadata_key"], manifest))

def ingest_raw_data(source_files, pipeline_run_id, pipeline_load_date):

    raw_dataframes = read_source_data(source_files)
    s3_client = create_minio_client()
    bucket_name = os.environ["MINIO_BUCKET"]

    # A Raw rerun must preserve this run's downstream metadata and inputs.
    metadata = {
        "run_id": pipeline_run_id,
        "load_date": pipeline_load_date,
        "metadata_key": (
            f"metadata/etl_metadata/load_date={pipeline_load_date}/"
            f"run_id={pipeline_run_id}/manifest.parquet"
        ),
    }
    try:
        existing_manifest = read_manifest(s3_client, bucket_name, metadata)
    except ClientError as error:
        if error.response["Error"]["Code"] != "NoSuchKey":
            raise
        existing_manifest = pd.DataFrame()

    if not existing_manifest.empty and existing_manifest["layer"].ne("raw").any():
        previous_raw = read_layer_data(s3_client, bucket_name, metadata, "raw")
        for dataset_name, dataframe in raw_dataframes.items():
            if not dataframe.equals(previous_raw[dataset_name]):
                raise ValueError(
                    "Raw inputs changed after downstream processing. Start a new DAG run."
                )

    filenames = {
        "train_data": "train.parquet",
        "test_data": "test.parquet",
        "RUL_data": "RUL.parquet",
    }

    raw_metadata_records = []

    for dataset_name, dataframe in raw_dataframes.items():
        object_key = (
            f"raw/{dataset_name}/"
            f"load_date={pipeline_load_date}/"
            f"run_id={pipeline_run_id}/"
            f"{filenames[dataset_name]}"
        )

        print(upload_parquet(s3_client, bucket_name, object_key, dataframe))
        raw_metadata_records.append({
            "run_id": pipeline_run_id,
            "load_date": pipeline_load_date,
            "layer": "raw",
            "dataset_name": dataset_name,
            "object_key": object_key,
            "status": "success",
            "row_count": len(dataframe),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })

    # Publish only after all uploads, preserving unrelated stage records.
    update_manifest(s3_client, bucket_name, metadata, raw_metadata_records, existing_manifest)
    return metadata
