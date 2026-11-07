"""Check Raw monitoring against the existing validation contract, without MinIO."""

import ast
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.monitoring import data_monitoring as monitoring
from src.validation.raw_validation import validate_raw_frames


@pytest.fixture
def frames():
    rows = [
        {
            "unit_number": engine,
            "time_in_cycles": cycle,
            **{f"operational_setting_{i}": float(i) for i in range(1, 4)},
            **{f"sensor_measurement_{i}": float(i + cycle) for i in range(1, 22)},
        }
        for engine in (1, 2) for cycle in (1, 2, 3)
    ]
    return {
        "train_data": pd.DataFrame(rows),
        "test_data": pd.DataFrame(rows),
        "RUL_data": pd.DataFrame({"unit_number": [1, 2], "remaining_useful_life": [0, 5]}),
    }


def report_for(frames):
    return monitoring.build_run_quality_report(
        frames, {name: len(frame) for name, frame in frames.items()}
    )


@pytest.mark.parametrize("dataset", ["train_data", "test_data", "RUL_data"])
@pytest.mark.parametrize("problem", [
    "missing_column", "unexpected_column", "reordered_columns", "duplicate_column",
    "empty", "missing_value", "infinity", "text", "boolean",
    "float_id_dtype", "fractional_id", "zero_id", "negative_id", "conflict",
])
def test_monitoring_preserves_raw_rejection_rules(frames, dataset, problem):
    frame = frames[dataset]
    value_column = "remaining_useful_life" if dataset == "RUL_data" else "sensor_measurement_1"
    if problem == "missing_column":
        frames[dataset] = frame.drop(columns=value_column)
    elif problem == "unexpected_column":
        frames[dataset] = frame.assign(extra=1)
    elif problem == "reordered_columns":
        frames[dataset] = frame[frame.columns[::-1]]
    elif problem == "duplicate_column":
        frames[dataset] = pd.concat([frame, frame[[value_column]]], axis=1)
    elif problem == "empty":
        frames[dataset] = frame.iloc[:0]
    elif problem == "missing_value":
        frame[value_column] = frame[value_column].astype(float)
        frame.loc[0, value_column] = np.nan
    elif problem == "infinity":
        frame[value_column] = frame[value_column].astype(float)
        frame.loc[0, value_column] = np.inf
    elif problem == "text":
        frame[value_column] = "invalid"
    elif problem == "boolean":
        frame[value_column] = True
    elif problem in ("float_id_dtype", "fractional_id"):
        frame["unit_number"] = frame["unit_number"].astype(float)
        if problem == "fractional_id":
            frame["unit_number"] += 0.5
    elif problem in ("zero_id", "negative_id"):
        frame.loc[0, "unit_number"] = 0 if problem == "zero_id" else -1
    else:
        conflicting = frame.iloc[[0]].copy()
        conflicting[value_column] += 10
        frames[dataset] = pd.concat([frame, conflicting], ignore_index=True)

    originals = {name: frame.copy(deep=True) for name, frame in frames.items()}
    with pytest.raises((ValueError, TypeError)):
        validate_raw_frames(frames)
    report = report_for(frames)
    assert report["raw_checks_passed"] is False
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(frame, originals[name])
    # Even schema/type failures must produce a report that can be saved.
    json.dumps(report, default=monitoring.convert_numpy_value, allow_nan=False)


@pytest.mark.parametrize("dataset", ["train_data", "test_data"])
@pytest.mark.parametrize("problem", ["float_cycle_dtype", "fractional_cycle", "zero_cycle", "gap", "missing_start"])
def test_monitoring_preserves_cycle_rules(frames, dataset, problem):
    frame = frames[dataset]
    if problem in ("float_cycle_dtype", "fractional_cycle"):
        frame["time_in_cycles"] = frame["time_in_cycles"].astype(float)
        if problem == "fractional_cycle":
            frame.loc[0, "time_in_cycles"] = 1.5
    elif problem == "zero_cycle":
        frame.loc[0, "time_in_cycles"] = 0
    else:
        frames[dataset] = frame.drop(index=1 if problem == "gap" else 0)
    with pytest.raises((ValueError, TypeError)):
        validate_raw_frames(frames)
    assert report_for(frames)["raw_checks_passed"] is False


@pytest.mark.parametrize("problem", ["negative_rul", "reordered_labels", "missing_engine", "unexpected_engine"])
def test_monitoring_preserves_rul_rules(frames, problem):
    if problem == "negative_rul":
        frames["RUL_data"].loc[0, "remaining_useful_life"] = -1
    elif problem == "reordered_labels":
        frames["RUL_data"] = frames["RUL_data"].iloc[::-1]
    elif problem == "missing_engine":
        frames["RUL_data"] = frames["RUL_data"].iloc[:1]
    else:
        frames["RUL_data"].loc[1, "unit_number"] = 3
    with pytest.raises(ValueError):
        validate_raw_frames(frames)
    report = report_for(frames)
    assert report["raw_checks_passed"] is False
    if problem == "reordered_labels":
        assert report["test_rul_alignment"]["engine_ids_match"] is True
        assert report["test_rul_alignment"]["rul_engine_order_matches"] is False


@pytest.mark.parametrize("variant", ["valid", "exact_duplicates", "fractional_rul", "unsorted_histories"])
def test_monitoring_preserves_accepted_data(frames, variant):
    if variant == "exact_duplicates":
        frames = {name: pd.concat([frame, frame.iloc[[0]]], ignore_index=True) for name, frame in frames.items()}
    elif variant == "fractional_rul":
        frames["RUL_data"]["remaining_useful_life"] = [0.0, 5.5]
    elif variant == "unsorted_histories":
        for name in ("train_data", "test_data"):
            frames[name] = frames[name].iloc[::-1]
    validate_raw_frames(frames)
    assert report_for(frames)["raw_checks_passed"] is True


@pytest.mark.parametrize("problem", [None, "float_id_dtype", "reordered_labels", "count_mismatch"])
def test_monitoring_saves_report_before_gate_decision(frames, monkeypatch, problem):
    metadata = {"run_id": "offline_run", "load_date": "2026-10-31", "metadata_key": "offline_manifest"}
    if problem == "float_id_dtype":
        frames["train_data"]["unit_number"] = frames["train_data"]["unit_number"].astype(float)
    elif problem == "reordered_labels":
        frames["RUL_data"] = frames["RUL_data"].iloc[::-1]
    manifest = pd.DataFrame([
        {
            "layer": "raw", "dataset_name": name, "status": "success",
            "row_count": len(frame) + (1 if problem == "count_mismatch" and name == "test_data" else 0),
            "object_key": f"raw/{name}/load_date=2026-10-31/run_id=offline_run/data.parquet",
        }
        for name, frame in frames.items()
    ])
    client = Mock()
    monkeypatch.setenv("MINIO_BUCKET", "offline_bucket")
    monkeypatch.setattr(monitoring, "create_minio_client", lambda: client)
    monkeypatch.setattr(monitoring, "read_manifest", lambda *args: manifest)
    monkeypatch.setattr(monitoring, "read_parquet_object", lambda client, bucket, key: frames[key.split("/")[1]])
    report = monitoring.monitor_raw_data(metadata)
    client.put_object.assert_called_once()
    upload = client.put_object.call_args.kwargs
    saved = json.loads(upload["Body"])
    assert saved["raw_checks_passed"] is (problem is None)
    assert saved["monitored_at"] == report["monitored_at"]
    assert datetime.fromisoformat(saved["monitored_at"]).utcoffset() == timezone.utc.utcoffset(None)
    assert all(saved[key] == value for key, value in metadata.items())
    assert upload["Key"] == "monitoring/data_quality/load_date=2026-10-31/run_id=offline_run/raw_quality_report.json"
    assert upload["ContentType"] == "application/json"


def test_dag_has_one_raw_gate_and_stops_on_failed_report():
    path = Path(__file__).resolve().parents[1] / "airflow/dags/predictive_maintenance_pipeline.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dag = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    tasks = [node for node in dag.body if isinstance(node, ast.FunctionDef)]
    assert not any(node.name == "validate_raw" for node in tasks)
    gate = next(node for node in tasks if node.name == "monitor_raw_quality")
    gate.decorator_list = []
    loader = Mock(return_value={"raw_checks_passed": True})
    namespace = {"monitor_raw_data": loader}
    exec(compile(ast.Module(body=[gate], type_ignores=[]), str(path), "exec"), namespace)
    metadata = {"run_id": "offline_run"}
    assert namespace["monitor_raw_quality"](metadata) is metadata
    loader.return_value = {"raw_checks_passed": False}
    with pytest.raises(ValueError, match="offline_run"):
        namespace["monitor_raw_quality"](metadata)
    calls = {
        node.targets[0].id: node.value for node in dag.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
    }
    assert calls["monitored_metadata"].args[0].id == "raw_metadata"
    assert calls["silver_metadata"].args[0].id == "monitored_metadata"
