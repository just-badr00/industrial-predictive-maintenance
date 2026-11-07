import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from pathlib import Path
import json
from datetime import datetime, timezone
from src.ingestion.raw_ingestion import create_minio_client,read_layer_data
import os

def summarize_feature_distributions(dataframe, feature_columns):
    # Reject unusable inputs before describe() can silently skip missing values.
    feature_columns = list(feature_columns)
    if not feature_columns or len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Model feature names must be nonempty and unique.")
    if not dataframe.columns.is_unique:
        raise ValueError("The feature table contains duplicate column names.")

    missing_columns = sorted(
        (set(feature_columns) | {"time_in_cycles"}) - set(dataframe.columns)
    )
    if missing_columns:
        raise ValueError(f"Missing drift input columns: {missing_columns}")
    cycle_dtype = dataframe["time_in_cycles"].dtype
    if not is_numeric_dtype(cycle_dtype) or is_bool_dtype(cycle_dtype):
        raise ValueError("time_in_cycles must be numeric.")

    final_df = dataframe[dataframe["time_in_cycles"]>=20]
    final_df = final_df.loc[:,feature_columns]
    if len(final_df) < 2:
        raise ValueError("Drift summaries require at least two rows at cycle 20 or later.")
    if any(
        not is_numeric_dtype(dtype) or is_bool_dtype(dtype)
        for dtype in final_df.dtypes
    ):
        raise ValueError("Selected model features must be numeric.")
    if final_df.isna().any().any() or not np.isfinite(final_df.to_numpy(dtype=float)).all():
        raise ValueError("Eligible model features contain missing or infinite values.")
    # Early-window NaNs are excluded above; constant eligible features remain valid.
    statistics = final_df.describe(percentiles=[0.05, 0.5, 0.95]).T
    return statistics

def compare_feature_distributions(reference_dataframe, current_dataframe, feature_columns):
    reference_dataframe_statistics = summarize_feature_distributions(reference_dataframe,feature_columns)
    current_dataframe_statistics = summarize_feature_distributions(current_dataframe,feature_columns)

    reference_dataframe_statistics = reference_dataframe_statistics.add_prefix("reference_")
    current_dataframe_statistics = current_dataframe_statistics.add_prefix("current_")
    comparison = current_dataframe_statistics.join(reference_dataframe_statistics)
    comparison["mean_change"] = comparison["current_mean"] - comparison["reference_mean"]

    reference_std = comparison["reference_std"].replace(0, np.nan)

    comparison["mean_shift_in_reference_std"] = (
        comparison["mean_change"].abs() / reference_std
    )

    # Compare spread separately from changes in the centre and percentiles.
    # A ratio of 1 means equal standard deviations; zero reference std stays undefined.
    comparison["std_ratio"] = comparison["current_std"] / reference_std
    comparison["median_change"] = comparison["current_50%"] - comparison["reference_50%"]
    comparison["p05_change"] = comparison["current_5%"] - comparison["reference_5%"]
    comparison["p95_change"] = comparison["current_95%"] - comparison["reference_95%"]
    return comparison

def load_drift_data(metadata, project_root):

    files_path = Path(project_root) / "output" / "serving_model"


    with open(files_path/"model_info.json","r",encoding="utf-8") as file:
        model_info = json.load(file)

    reference_run_id = model_info["gold_run_id"]

    reference_load_date = datetime.strptime(
        reference_run_id[:8], "%Y%m%d"
    ).strftime("%Y-%m-%d")

    reference_metadata = {
        "run_id": reference_run_id,
        "load_date": reference_load_date,
        "metadata_key": (
            f"metadata/etl_metadata/load_date={reference_load_date}/"
            f"run_id={reference_run_id}/manifest.parquet"
        ),
    }
    s3_client = create_minio_client()
    bucket_name = os.environ["MINIO_BUCKET"]

    reference_dataframes = read_layer_data(
        s3_client, bucket_name, reference_metadata, "gold",
        dataset_names=("train_features",),
    )

    current_dataframes = read_layer_data(
        s3_client, bucket_name, metadata, "gold",
        dataset_names=("test_features",),
    )

    with open(files_path/"final_training_features.json","r",encoding="utf-8") as file:
        final_training_features = json.load(file)

    feature_columns = final_training_features["feature_columns"]

    return{
        "reference_dataframe": reference_dataframes["train_features"],
        "current_dataframe": current_dataframes["test_features"],
        "feature_columns": feature_columns,
        "model_info": model_info,
        "reference_metadata": reference_metadata,
    }

def flag_mean_shifts(comparison, threshold):
    comparison_copy = comparison.copy()
    comparison_copy["mean_shift_flag"] = comparison_copy["mean_shift_in_reference_std"].apply(lambda x: True if x>=threshold else ( False if x <threshold else np.nan))
    return comparison_copy


# Flag spread ratios outside the bounds without changing the comparison input.
def flag_spread_changes(comparison, lower_ratio, upper_ratio):
    if (
        not np.isfinite(lower_ratio)
        or not np.isfinite(upper_ratio)
        or not 0 <= lower_ratio < upper_ratio
    ):
        raise ValueError("Spread bounds must be finite and satisfy 0 <= lower_ratio < upper_ratio.")

    comparison_copy = comparison.copy()
    std_ratio = comparison_copy["std_ratio"]
    # Strict comparisons leave values exactly on either boundary unflagged.
    comparison_copy["spread_change_flag"] = (
        (std_ratio < lower_ratio) | (std_ratio > upper_ratio)
    ).astype("boolean")
    # Missing ratios mean the feature could not be assessed, rather than passing.
    comparison_copy.loc[std_ratio.isna(), "spread_change_flag"] = pd.NA
    return comparison_copy

def build_drift_report(metadata,project_root,mean_shift_threshold,lower_ratio,upper_ratio):
    drift_data = load_drift_data(metadata,project_root)
    comparison = compare_feature_distributions(drift_data["reference_dataframe"],drift_data["current_dataframe"],drift_data["feature_columns"])
    comparison = flag_mean_shifts(comparison, mean_shift_threshold)
    comparison = flag_spread_changes(comparison, lower_ratio, upper_ratio)

    feature_records = comparison.rename_axis("feature").reset_index()

    feature_records = feature_records.astype(object).where(
        feature_records.notna(), None
    )

    return { 
        # Record when this comparison was calculated, independently of load dates.
        "monitored_at": datetime.now(timezone.utc).isoformat(),
        "current_metadata": metadata.copy(),
        "reference_metadata": drift_data["reference_metadata"],
        "model_info": drift_data["model_info"],
        "thresholds": {
            "mean_shift_threshold": mean_shift_threshold,
            "lower_ratio": lower_ratio,
            "upper_ratio": upper_ratio,
        },
        "summary": {
            "feature_count": int(len(comparison)),
            "mean_shift_flag_count": int(comparison["mean_shift_flag"].sum()),
            "spread_change_flag_count": int(comparison["spread_change_flag"].sum()),
            "unavailable_mean_shift_count": int(comparison["mean_shift_flag"].isna().sum()),
            "unavailable_spread_count": int(comparison["spread_change_flag"].isna().sum()),
        },
        "feature_comparison": feature_records.to_dict(orient="records"),
   }


# Save one drift report under the current input run, preserving reference lineage inside it.
def save_drift_report(s3_client, bucket_name, metadata, report):
    object_key = (
        f"monitoring/data_drift/load_date={metadata['load_date']}/"
        f"run_id={metadata['run_id']}/feature_drift_report.json"
    )
    # Unavailable measurements are already None; strict JSON rejects unexpected NaN/inf.
    report_json = json.dumps(report, allow_nan=False, indent=2)
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=report_json.encode("utf-8"),
        ContentType="application/json",
    )
    # Upload errors propagate, so this key is returned only after a successful write.
    return object_key


# NEW: Build and save drift findings; flags are observations, not validation failures.
def monitor_feature_drift(
metadata, project_root, mean_shift_threshold, lower_ratio, upper_ratio
):
    report = build_drift_report(
        metadata, project_root, mean_shift_threshold, lower_ratio, upper_ratio
    )
    s3_client = create_minio_client()
    bucket_name = os.environ["MINIO_BUCKET"]
    report_key = save_drift_report(s3_client, bucket_name, metadata, report)

    # These lines run after a successful upload; input/upload errors propagate.
    print(f"Saved feature drift report: {report_key}", flush=True)
    print(f"Drift summary: {report['summary']}", flush=True)
    return report
# END NEW: Feature drift monitoring entrypoint.
