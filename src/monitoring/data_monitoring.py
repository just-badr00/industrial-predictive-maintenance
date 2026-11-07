import os
import json
import pandas as pd 
import numpy as np
from pandas.api.types import is_integer_dtype, is_numeric_dtype, is_bool_dtype
from datetime import datetime, timezone
from src.ingestion.raw_ingestion import create_minio_client, read_parquet_object, read_manifest

def summarize_data_quality(dataframe):

    numeric_data = dataframe.select_dtypes(include="number")
    non_finite_rows = (~np.isfinite(numeric_data).all(axis=1)).sum()
    
    return {
        "row_count": len(dataframe),
        "missing_cells": dataframe.isnull().sum().sum(),
        "duplicate_rows": dataframe.duplicated().sum(),
        "missing_rows": dataframe.isnull().any(axis=1).sum(),
        "missing_cell_rate" : dataframe.isnull().sum().sum() / (dataframe.shape[0] * dataframe.shape[1]) if dataframe.size > 0 else None,
        "column_count" : dataframe.shape[1],
        "missing_cells_by_column" : dataframe.isnull().sum().to_dict(),
        "non_finite_rows" : non_finite_rows if dataframe.shape[0] > 0 else 0,
    }

def check_schema(dataframe, expected_columns):

    # NEW: Preserve Raw validation's column order, numeric types and integer keys.
    # Inspect dtypes positionally so duplicate column names can be reported safely.
    non_numeric_columns = sorted({
        column for column, dtype in zip(dataframe.columns, dataframe.dtypes)
        if column in expected_columns
        and (not is_numeric_dtype(dtype) or is_bool_dtype(dtype))
    })
    non_integer_id_columns = sorted({
        column for column, dtype in zip(dataframe.columns, dataframe.dtypes)
        if column in expected_columns
        and column in ("unit_number", "time_in_cycles")
        and (not is_integer_dtype(dtype) or is_bool_dtype(dtype))
    })

    return {
        "missing_columns": sorted(set(expected_columns) - set(dataframe.columns)),
        "unexpected_columns": sorted(set(dataframe.columns) - set(expected_columns)),
        "schema_matches": set(dataframe.columns) == set(expected_columns),
        "non_numeric_columns": non_numeric_columns,
        "column_order_matches": list(dataframe.columns) == list(expected_columns),
        "duplicate_columns": sorted(set(dataframe.columns[dataframe.columns.duplicated()])),
        "non_integer_id_columns": non_integer_id_columns,
    }


# Combine the measurements for one Raw dataset without changing its data.
def build_raw_quality_report(dataframe, expected_columns, expected_row_count):
    summary = summarize_data_quality(dataframe)
    schema = check_schema(dataframe, expected_columns)

    # The manifest supplies the expected count for this exact stored object.
    row_count_matches = summary["row_count"] == expected_row_count
    basic_checks_passed = bool(
        summary["row_count"] > 0
        and schema["schema_matches"]
        # NEW: Match the existing Raw validator before running record checks.
        and schema["column_order_matches"]
        and not schema["duplicate_columns"]
        and not schema["non_numeric_columns"]
        and not schema["non_integer_id_columns"]
        and summary["missing_cells"] == 0
        and summary["non_finite_rows"] == 0
        and row_count_matches
    )

    # Exact duplicates are reported; Silver is responsible for removing them.
    # Engine IDs, conflicting keys and cycle histories need separate checks.
    return {
        "summary": summary,
        "schema": schema,
        "expected_row_count": expected_row_count,
        "row_count_matches": row_count_matches,
        "basic_checks_passed": basic_checks_passed,
    }

def check_engine_records(dataframe):

    invalid_engine_id_rows = dataframe[(dataframe["unit_number"].isna()) | (~np.isfinite(dataframe["unit_number"])) | (dataframe["unit_number"] <= 0) | (dataframe["unit_number"] % 1 != 0)] 
    invalid_cycle_rows = dataframe[(dataframe["time_in_cycles"].isna()) | (~np.isfinite(dataframe["time_in_cycles"])) | (dataframe["time_in_cycles"] <= 0) | (dataframe["time_in_cycles"] % 1 != 0)]
    dataframe_non_dup = dataframe.drop_duplicates()
    conflicting_key_rows = dataframe_non_dup[dataframe_non_dup.duplicated(subset=["unit_number", "time_in_cycles"], keep=False)]
    return {
        "invalid_engine_id_rows": len(invalid_engine_id_rows),
        "invalid_cycle_rows": len(invalid_cycle_rows),
        "conflicting_key_rows": len(conflicting_key_rows),
    }


# Check observed cycle coverage after engine IDs and cycles are validated.
def check_cycle_history(dataframe):
    # One engine-cycle pair is enough for this check; conflicts are checked separately.
    history = (
        dataframe[["unit_number", "time_in_cycles"]]
        .drop_duplicates()
        .sort_values(["unit_number", "time_in_cycles"])
        .reset_index(drop=True)
    )

    # Each engine must have cycles 1, 2, 3, ... through its last observation.
    expected_cycles = history.groupby("unit_number").cumcount() + 1
    mismatched_cycles = history["time_in_cycles"] != expected_cycles
    affected_engine_ids = sorted(
        history.loc[mismatched_cycles, "unit_number"].unique().tolist()
    )

    # Count affected engines once, even when several later cycles mismatch.
    return {
        "engines_with_cycle_gaps": len(affected_engine_ids),
        "affected_engine_ids": affected_engine_ids,
    }

def build_engine_quality_report(dataframe, expected_columns, expected_row_count):
    report = build_raw_quality_report(dataframe, expected_columns, expected_row_count)
    report["engine_records"] = None
    report["cycle_history"] = None
    report["engine_checks_passed"] = False
    if not report["basic_checks_passed"] :
        return report

    report["engine_records"] = check_engine_records(dataframe)    
    if any(count > 0 for count in report["engine_records"].values()):
        return report

    report["cycle_history"] = check_cycle_history(dataframe)
    if report["cycle_history"]["engines_with_cycle_gaps"] == 0:
        report["engine_checks_passed"] = True

    return report


# Check numeric RUL records after the required columns and types are checked.
def check_rul_records(dataframe):
    engine_ids = dataframe["unit_number"]
    invalid_engine_id_rows = (
        engine_ids.isna()
        | ~np.isfinite(engine_ids)
        | (engine_ids <= 0)
        | (engine_ids % 1 != 0)
    )

    # Zero and fractional RUL values are valid; negative/non-finite values are not.
    rul_values = dataframe["remaining_useful_life"]
    invalid_rul_rows = (
        rul_values.isna()
        | ~np.isfinite(rul_values)
        | (rul_values < 0)
    )

    # Identical copies can be removed in Silver; different labels must be reported.
    dataframe_non_dup = dataframe.drop_duplicates()
    conflicting_engine_rows = dataframe_non_dup.duplicated(
        subset=["unit_number"], keep=False
    )

    return {
        "invalid_engine_id_rows": int(invalid_engine_id_rows.sum()),
        "invalid_rul_rows": int(invalid_rul_rows.sum()),
        "conflicting_engine_rows": int(conflicting_engine_rows.sum()),
    }


# Combine basic quality measurements and RUL-specific record checks.
def build_rul_quality_report(dataframe, expected_columns, expected_row_count):
    report = build_raw_quality_report(dataframe, expected_columns, expected_row_count)
    report["rul_records"] = None
    report["rul_checks_passed"] = False

    # Missing columns or non-numeric values must not reach the numeric RUL checks.
    if not report["basic_checks_passed"]:
        return report

    report["rul_records"] = check_rul_records(dataframe)
    report["rul_checks_passed"] = all(
        count == 0 for count in report["rul_records"].values()
    )
    return report


# Compare test-engine coverage with the engines in the RUL label table.
def check_test_rul_alignment(test_dataframe, rul_dataframe):
    # Run after each table's schema and engine IDs have passed their checks.
    # Sets ignore row order and repeated observations of the same engine.
    test_engine_ids = set(test_dataframe["unit_number"])
    rul_engine_ids = set(rul_dataframe["unit_number"])

    missing_rul_engine_ids = sorted(test_engine_ids - rul_engine_ids)
    unexpected_rul_engine_ids = sorted(rul_engine_ids - test_engine_ids)

    # NEW: Preserve the validator's label order after removing identical copies.
    # Test histories are sorted by engine before comparison in validate_raw_frames.
    rul_engine_order_matches = (
        sorted(test_engine_ids)
        == rul_dataframe.drop_duplicates()["unit_number"].tolist()
    )
    engine_ids_match = not missing_rul_engine_ids and not unexpected_rul_engine_ids

    # Coverage does not check label values or conflicts; the RUL report does that.
    return {
        "test_engine_count": len(test_engine_ids),
        "rul_engine_count": len(rul_engine_ids),
        "missing_rul_engine_ids": missing_rul_engine_ids,
        "unexpected_rul_engine_ids": unexpected_rul_engine_ids,
        "engine_ids_match": engine_ids_match,
        "rul_engine_order_matches": rul_engine_order_matches,
        "alignment_checks_passed": engine_ids_match and rul_engine_order_matches,
    }

def build_run_quality_report(raw_dataframes, expected_row_counts):
    engine_columns = (
        ["unit_number", "time_in_cycles"]
        + [f"operational_setting_{i}" for i in range(1, 4)]
        + [f"sensor_measurement_{i}" for i in range(1, 22)]
    )

    rul_columns = ["unit_number", "remaining_useful_life"]

    train_report = build_engine_quality_report(raw_dataframes["train_data"], engine_columns, expected_row_counts["train_data"])
    test_report = build_engine_quality_report(raw_dataframes["test_data"], engine_columns, expected_row_counts["test_data"])
    rul_report = build_rul_quality_report(raw_dataframes["RUL_data"], rul_columns, expected_row_counts["RUL_data"])

    alignment_report = None
    if test_report["engine_checks_passed"] and rul_report["rul_checks_passed"]:
        alignment_report = check_test_rul_alignment(raw_dataframes["test_data"], raw_dataframes["RUL_data"])

    return{
        "datasets": {
            "train_data": train_report,
            "test_data": test_report,
            "RUL_data": rul_report,
        },
        "test_rul_alignment": alignment_report,
        "raw_checks_passed": bool(
            train_report["engine_checks_passed"]
            and test_report["engine_checks_passed"]
            and rul_report["rul_checks_passed"]
            and alignment_report["alignment_checks_passed"]
        ),
        }

def monitor_raw_data(metadata):
    s3_client = create_minio_client()
    bucket_name = os.environ["MINIO_BUCKET"]
    raw_dataframes = {}
    manifest = read_manifest(s3_client, bucket_name, metadata)
    dataset_names = ("train_data", "test_data", "RUL_data")

    expected_row_counts = {}
    for dataset_name in dataset_names:
        matches = manifest[
            manifest["layer"].eq("raw")
            & manifest["dataset_name"].eq(dataset_name)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one Raw record for {dataset_name}.")
        expected_prefix = (
                f"raw/{dataset_name}/load_date={metadata['load_date']}/"
                f"run_id={metadata['run_id']}/"
            )
        record = matches.iloc[0]
        if not record["object_key"].startswith(expected_prefix):
            raise ValueError(f"Object key does not match the run: {dataset_name}.")

        if record["status"] != "success":
            raise ValueError(f"Raw dataset is not successful: {dataset_name}.")

        dataframe = read_parquet_object(s3_client, bucket_name, record["object_key"])
        expected_row_counts[dataset_name] = record["row_count"]
        raw_dataframes[dataset_name] = dataframe

    full_report = build_run_quality_report(raw_dataframes, expected_row_counts)
    full_report["run_id"] = metadata["run_id"]
    full_report["load_date"] = metadata["load_date"]
    full_report["metadata_key"] = metadata["metadata_key"]
    full_report["monitored_at"] = datetime.now(timezone.utc).isoformat()

    save_quality_report(s3_client, bucket_name, metadata, full_report)
    
    return full_report

def convert_numpy_value(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")

def save_quality_report(s3_client, bucket_name, metadata, report):
    object_key = (
        f"monitoring/data_quality/load_date={metadata['load_date']}/"
        f"run_id={metadata['run_id']}/raw_quality_report.json"
    )

    report_json = json.dumps(
        report,
        default=convert_numpy_value,
        allow_nan=False,
        indent=2,
    )

    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=report_json.encode("utf-8"),
        ContentType="application/json",
    )

    return object_key
