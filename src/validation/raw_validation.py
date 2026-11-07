"""Validate this run's stored Raw objects using Notebook 02's rules."""

# NEW: Extracted Raw validation; no Airflow imports or storage writes.
import os

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype, is_bool_dtype

from src.ingestion.raw_ingestion import create_minio_client, read_layer_data


def validate_raw_frames(raw_dataframes):
    columns = (
        ["unit_number", "time_in_cycles"]
        + [f"operational_setting_{i}" for i in range(1, 4)]
        + [f"sensor_measurement_{i}" for i in range(1, 22)]
    )
    cleaned = {}
    summary = {}
    for dataset_name in ("train_data", "test_data", "RUL_data"):
        dataframe = raw_dataframes[dataset_name]
        expected_columns = (
            ["unit_number", "remaining_useful_life"]
            if dataset_name == "RUL_data" else columns
        )
        if list(dataframe.columns) != expected_columns:
            raise ValueError(f"Wrong Raw schema for {dataset_name}.")
        if dataframe.empty:
            raise ValueError(f"{dataset_name} is empty.")
        if any(
            not is_numeric_dtype(dataframe[column]) or is_bool_dtype(dataframe[column])
            for column in expected_columns
        ):
            raise TypeError(f"{dataset_name} must contain numeric columns.")
        if dataframe.isna().any().any():
            raise ValueError(f"{dataset_name} contains missing values.")
        if not np.isfinite(dataframe.to_numpy(dtype=float)).all():
            raise ValueError(f"{dataset_name} contains non-finite numeric values.")

        id_columns = ["unit_number"]
        if dataset_name != "RUL_data":
            id_columns.append("time_in_cycles")
        for column in id_columns:
            if not is_integer_dtype(dataframe[column]):
                raise TypeError(f"{dataset_name}.{column} must have an integer dtype.")
            if dataframe[column].le(0).any():
                raise ValueError(f"{dataset_name}.{column} must be positive.")

        # Exact duplicates are cleaned in Silver; conflicting key duplicates fail.
        unique_rows = dataframe.drop_duplicates().copy()
        if unique_rows.duplicated(subset=id_columns, keep=False).any():
            raise ValueError(f"{dataset_name} contains conflicting duplicate records.")
        if dataset_name == "RUL_data":
            if unique_rows["remaining_useful_life"].lt(0).any():
                raise ValueError("RUL_data contains negative remaining useful life.")
        else:
            unique_rows = unique_rows.sort_values(id_columns).reset_index(drop=True)
            expected_cycles = unique_rows.groupby("unit_number").cumcount() + 1
            if not unique_rows["time_in_cycles"].eq(expected_cycles).all():
                raise ValueError(f"{dataset_name} has missing cycles or does not start at 1.")
        cleaned[dataset_name] = unique_rows
        summary[dataset_name] = {
            "row_count": len(dataframe),
            "exact_duplicates": int(dataframe.duplicated().sum()),
        }

    test_engines = cleaned["test_data"]["unit_number"].unique().tolist()
    if test_engines != cleaned["RUL_data"]["unit_number"].tolist():
        raise ValueError("Test engines and RUL labels do not match in number/order.")
    return summary


def validate_raw_data(metadata):
    s3_client = create_minio_client()
    raw_dataframes = read_layer_data(
        s3_client, os.environ["MINIO_BUCKET"], metadata, "raw"
    )
    summary = validate_raw_frames(raw_dataframes)
    print(f"Raw validation passed for run {metadata['run_id']}: {summary}")
    # Pass only lineage to the next task, not DataFrames.
    return metadata.copy()
