from datetime import datetime, timezone
import os
import numpy as np
from src.ingestion.raw_ingestion import read_layer_data,create_minio_client
import json

# Summarize a validated batch containing one prediction per engine.
def summarize_prediction_batch(predictions):
    if predictions.empty:
        raise ValueError("Cannot summarize an empty prediction batch.")

    engine_count = len(predictions)
    rul = predictions["predicted_rul"]
    status = predictions["status"]
    healthy_count = int(status.eq("healthy").sum())
    warning_count = int(status.eq("warning").sum())
    critical_count = int(status.eq("critical").sum())

    return {
        "engine_count": int(engine_count),
        "mean_predicted_rul": float(rul.mean()),
        "median_predicted_rul": float(rul.median()),
        "min_predicted_rul": float(rul.min()),
        "max_predicted_rul": float(rul.max()),
        "healthy_count": healthy_count,
        "warning_count": warning_count,
        "critical_count": critical_count,
        # This is a percentage of engines in the batch, not an accuracy metric.
        "critical_percentage": float(critical_count / engine_count * 100),
        "model_versions": sorted(predictions["model_version"].unique().tolist()),
    }


def evaluate_prediction_accuracy(predictions, test_data, rul_data=None):

    if predictions.empty:
        raise ValueError("The predictions table that is given is empty")

    unavailable_result = {
            "status": "awaiting_labels",
            "evaluated_engine_count": 0,
            "label_coverage_percentage": 0.0,
            "mae": None,
            "rmse": None,
        }
    if rul_data is None or rul_data.empty:
        return unavailable_result

    labels = rul_data[
        ["unit_number", "remaining_useful_life"]
    ].rename(columns={
        "unit_number": "engine_id",
        "remaining_useful_life": "actual_rul",
    })

    matched = labels.merge(predictions,on="engine_id",how="inner",validate="one_to_one")

    if matched.empty:
        return unavailable_result

    final_cycles = test_data.groupby("unit_number")["time_in_cycles"].max()
    matched["expected_cycles"] = matched["engine_id"].map(final_cycles)

    if not (matched["cycle"] == matched["expected_cycles"]).all():
        raise ValueError(
            "Predictions must match each engine's final observed test cycle."
        )

    values = matched[["actual_rul", "predicted_rul"]].to_numpy(dtype=float)

    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Actual and predicted RUL must be finite and nonnegative.")

    # Evaluate only matched endpoint labels and make their coverage explicit.
    errors = matched["actual_rul"] - matched["predicted_rul"]
    evaluated_engine_count = len(matched)
    return {
        "status": (
            "evaluated" if evaluated_engine_count == len(predictions)
            else "partially_labeled"
        ),
        "evaluated_engine_count": int(evaluated_engine_count),
        "label_coverage_percentage": float(evaluated_engine_count / len(predictions) * 100),
        "mae": float(errors.abs().mean()),
        "rmse": float((errors ** 2).mean() ** 0.5),
    }

def assess_accuracy_alert(accuracy, mae_limit):
    if accuracy["status"] != "evaluated":
        return {
            "status" : "unavailable",
            "mae_cycles" : accuracy["mae"],
            "limit_cycles" : mae_limit
        }
    if accuracy["mae"] <= mae_limit:
        return {
            "status" : "within_limit",
            "mae_cycles" : accuracy["mae"],
            "limit_cycles" : mae_limit
        }
    else :
        return {
            "status" : "above_limit",
            "mae_cycles" : accuracy["mae"],
            "limit_cycles" : mae_limit
        }


# NEW: Keep prediction behavior and optional outcome-based accuracy in one report.
def build_model_monitoring_report(
    predictions, test_data, metadata, rul_data=None, label_source=None, mae_limit=25.0
):
    prediction_summary = summarize_prediction_batch(predictions)
    accuracy = evaluate_prediction_accuracy(predictions, test_data, rul_data)

    # An evaluated score needs provenance; absent/unmatched labels remain unavailable.
    if accuracy["evaluated_engine_count"] > 0:
        if not isinstance(label_source, str) or not label_source.strip():
            raise ValueError("A nonempty label_source is required when accuracy is evaluated.")

    return {
        "monitored_at": datetime.now(timezone.utc).isoformat(),
        "current_metadata": metadata.copy(),
        "prediction_summary": prediction_summary,
        "accuracy": accuracy,
        "label_source": label_source if accuracy["evaluated_engine_count"] > 0 else None,
        "accuracy_scope": "final_observed_test_cycle",
        "accuracy_alert": assess_accuracy_alert(accuracy, mae_limit),
    }
# END NEW: Combined model-monitoring report.

def load_model_monitoring_data(metadata, include_benchmark_labels=False):

    s3_client = create_minio_client()

    bucket_name = os.environ["MINIO_BUCKET"]

    prediction_dataframes = read_layer_data(
        s3_client, bucket_name, metadata, "gold",
        dataset_names=("maintenance_predictions",),
    )

    silver_dataset_names = ["test_data"]

    if include_benchmark_labels:
        silver_dataset_names.append("RUL_data")

    silver_dataframes = read_layer_data(
        s3_client, bucket_name, metadata, "silver",
        dataset_names=silver_dataset_names,
    )

    return {
        "predictions": prediction_dataframes["maintenance_predictions"],
        "test_data": silver_dataframes["test_data"],
        "rul_data": silver_dataframes.get("RUL_data"),
        "label_source": ("fd001_benchmark" if include_benchmark_labels else None),
    }

def save_model_monitoring_report(s3_client, bucket_name, metadata, report):

    object_key = (
        f"monitoring/model/load_date={metadata['load_date']}/"
        f"run_id={metadata['run_id']}/model_monitoring_report.json"
    )
    report_json = json.dumps(report, allow_nan=False, indent=2)
    s3_client.put_object(
        Bucket=bucket_name,
        Key=object_key,
        Body=report_json.encode("utf-8"),
        ContentType="application/json",
    )
    # Upload errors propagate, so this key is returned only after a successful write.
    return object_key

def monitor_model(metadata, include_benchmark_labels=False):
    data = load_model_monitoring_data(metadata,include_benchmark_labels)
    report = build_model_monitoring_report(
        data["predictions"],
        data["test_data"],
        metadata,
        rul_data=data["rul_data"],
        label_source=data["label_source"],
    )
    s3_client = create_minio_client()
    bucket_name = os.environ["MINIO_BUCKET"]
    key = save_model_monitoring_report(s3_client, bucket_name, metadata, report)
    print(f"The object key returned is : {key}, Here are some important details about the model accuracy :\n{report["accuracy"]}")
    return report

