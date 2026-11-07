"""Numeric drift summaries use eligible rows and never hide invalid model inputs."""

import json
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.monitoring.drift_monitoring import (
    summarize_feature_distributions, compare_feature_distributions, flag_spread_changes,
    build_drift_report, save_drift_report, monitor_feature_drift,
)
from src.monitoring import drift_monitoring


@pytest.fixture
def frame():
    return pd.DataFrame({
        "time_in_cycles": [19, 20, 21, 22],
        "feature_a": [np.nan, 0.0, 10.0, 20.0],
        "feature_b": [np.nan, 5.0, 5.0, 5.0],
        "unused": [np.nan] * 4,
    }, index=[10, 30, 20, 40])


def test_summary_excludes_warmup_and_keeps_feature_order(frame):
    original = frame.copy(deep=True)
    result = summarize_feature_distributions(frame, ["feature_b", "feature_a"])
    assert result.index.tolist() == ["feature_b", "feature_a"]
    assert result.columns.tolist() == ["count", "mean", "std", "min", "5%", "50%", "95%", "max"]
    np.testing.assert_allclose(result.loc["feature_a"], [3, 10, 10, 0, 1, 10, 19, 20])
    assert result.loc["feature_b", "std"] == 0
    assert_frame_equal(frame, original)


@pytest.mark.parametrize("problem, message", [
    ("empty_features", "nonempty and unique"),
    ("repeated_features", "nonempty and unique"),
    ("missing_feature", "Missing drift input columns"),
    ("missing_cycle", "Missing drift input columns"),
    ("duplicate_columns", "duplicate column names"),
    ("text_cycle", "time_in_cycles must be numeric"),
    ("boolean_cycle", "time_in_cycles must be numeric"),
    ("empty", "at least two rows"),
    ("only_warmup", "at least two rows"),
    ("one_eligible", "at least two rows"),
    ("missing_value", "missing or infinite"),
    ("infinity", "missing or infinite"),
    ("negative_infinity", "missing or infinite"),
    ("text_feature", "features must be numeric"),
    ("boolean_feature", "features must be numeric"),
])
def test_invalid_drift_inputs_rejected(frame, problem, message):
    features = ["feature_a", "feature_b"]
    if problem == "empty_features":
        features = []
    elif problem == "repeated_features":
        features = ["feature_a", "feature_a"]
    elif problem == "missing_feature":
        features.append("absent")
    elif problem == "missing_cycle":
        frame = frame.drop(columns="time_in_cycles")
    elif problem == "duplicate_columns":
        frame = pd.concat([frame, frame[["feature_a"]]], axis=1)
    elif problem == "text_cycle":
        frame["time_in_cycles"] = "20"
    elif problem == "boolean_cycle":
        frame["time_in_cycles"] = True
    elif problem == "empty":
        frame = frame.iloc[:0]
    elif problem == "only_warmup":
        frame = frame.iloc[:1]
    elif problem == "one_eligible":
        frame = frame.iloc[:2]
    elif problem in ("missing_value", "infinity", "negative_infinity"):
        frame.loc[30, "feature_a"] = {
            "missing_value": np.nan, "infinity": np.inf, "negative_infinity": -np.inf,
        }[problem]
    elif problem == "text_feature":
        frame["feature_a"] = "10"
    elif problem == "boolean_feature":
        frame["feature_a"] = True
    original = frame.copy(deep=True)
    with pytest.raises(ValueError, match=message):
        summarize_feature_distributions(frame, features)
    assert_frame_equal(frame, original)


def test_exactly_two_rows_and_constant_features_are_valid(frame):
    result = summarize_feature_distributions(frame.iloc[1:3], ["feature_b"])
    assert result.loc["feature_b", "count"] == 2
    assert result.loc["feature_b", "std"] == 0


def test_constant_reference_ratios_are_undefined_but_changes_are_available(frame):
    current = frame.copy(deep=True)
    current.loc[current["time_in_cycles"] >= 20, "feature_b"] = [4.0, 5.0, 6.0]
    result = compare_feature_distributions(frame, current, ["feature_b"])
    assert np.isnan(result.loc["feature_b", "mean_shift_in_reference_std"])
    assert np.isnan(result.loc["feature_b", "std_ratio"])
    assert result.loc["feature_b", "median_change"] == 0
    np.testing.assert_allclose(result.loc["feature_b", ["p05_change", "p95_change"]], [-0.9, 0.9])


def test_comparison_rejects_invalid_current_inputs(frame):
    current = frame.copy(deep=True)
    current.loc[30, "feature_a"] = np.nan
    with pytest.raises(ValueError, match="missing or infinite"):
        compare_feature_distributions(frame, current, ["feature_a"])


@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_spread_flags_boundaries_missing_values_and_unchanged_input(dtype):
    comparison = pd.DataFrame({
        "std_ratio": pd.Series([0.0, 0.49, 0.5, 1.0, 2.0, 2.01, np.nan], dtype=dtype),
        "mean_change": [0.0] * 7,
    })
    comparison.index = ["constant", "below", "lower", "middle", "upper", "above", "missing"]
    original = comparison.copy(deep=True)
    result = flag_spread_changes(comparison, 0.5, 2.0)
    assert result["spread_change_flag"].iloc[:6].tolist() == [True, True, False, False, False, True]
    assert pd.isna(result.loc["missing", "spread_change_flag"])
    assert str(result["spread_change_flag"].dtype) == "boolean"
    assert_frame_equal(comparison, original)
    assert_frame_equal(result[comparison.columns], original)


@pytest.mark.parametrize("lower, upper", [
    (-0.5, 2.0), (2.0, 0.5), (1.0, 1.0), (np.nan, 2.0),
    (0.5, np.nan), (0.5, np.inf), (-np.inf, 2.0),
])
def test_spread_flags_reject_invalid_bounds(lower, upper):
    with pytest.raises(ValueError, match="Spread bounds"):
        flag_spread_changes(pd.DataFrame({"std_ratio": [1.0]}), lower, upper)


def test_spread_flags_empty_comparison():
    result = flag_spread_changes(pd.DataFrame({"std_ratio": pd.Series(dtype=float)}), 0.5, 2.0)
    assert result.empty
    assert str(result["spread_change_flag"].dtype) == "boolean"


def test_complete_drift_report_saves_utc_lineage_and_null_values(frame, monkeypatch):
    metadata = {"run_id": "current_run", "load_date": "2026-11-01", "metadata_key": "current_manifest"}
    reference_metadata = {"run_id": "reference_run", "load_date": "2026-09-02", "metadata_key": "reference_manifest"}
    current = frame.copy(deep=True)
    current["feature_a"] += 20
    monkeypatch.setattr(drift_monitoring, "load_drift_data", lambda *args: {
        "reference_dataframe": frame, "current_dataframe": current,
        "feature_columns": ["feature_a", "feature_b"],
        "reference_metadata": reference_metadata, "model_info": {"model_version": "1"},
    })
    report = build_drift_report(metadata, "unused_root", 1.0, 0.5, 2.0)
    original = deepcopy(report)
    client = Mock()
    key = save_drift_report(client, "offline_bucket", metadata, report)
    client.put_object.assert_called_once()
    upload = client.put_object.call_args.kwargs
    assert key == "monitoring/data_drift/load_date=2026-11-01/run_id=current_run/feature_drift_report.json"
    assert upload["Key"] == key
    assert upload["Bucket"] == "offline_bucket"
    assert upload["ContentType"] == "application/json"
    assert isinstance(upload["Body"], bytes)
    saved = json.loads(upload["Body"].decode("utf-8"))
    assert saved == report == original
    assert datetime.fromisoformat(saved["monitored_at"]).utcoffset() == timezone.utc.utcoffset(None)
    assert saved["current_metadata"] == metadata
    assert saved["reference_metadata"] == reference_metadata
    assert saved["summary"]["mean_shift_flag_count"] == 1
    assert saved["summary"]["unavailable_mean_shift_count"] == 1
    constant_feature = saved["feature_comparison"][1]
    assert constant_feature["feature"] == "feature_b"
    for field in ("mean_shift_in_reference_std", "std_ratio", "mean_shift_flag", "spread_change_flag"):
        assert constant_feature[field] is None


@pytest.mark.parametrize("value, error_type", [
    (float("nan"), ValueError), (float("inf"), ValueError), (object(), TypeError),
])
def test_invalid_json_report_is_rejected_before_upload(value, error_type):
    client = Mock()
    with pytest.raises(error_type):
        save_drift_report(client, "offline", {"run_id": "run", "load_date": "2026-11-01"}, {"value": value})
    client.put_object.assert_not_called()


def test_drift_report_upload_failure_propagates():
    client = Mock()
    client.put_object.side_effect = RuntimeError("simulated upload failure")
    with pytest.raises(RuntimeError, match="upload failure"):
        save_drift_report(client, "offline", {"run_id": "run", "load_date": "2026-11-01"}, {"feature_comparison": []})


@pytest.mark.parametrize("shift", [0.0, 20.0])
def test_monitor_entrypoint_saves_and_returns_flagged_or_unflagged_report(frame, monkeypatch, capsys, shift):
    metadata = {"run_id": "current_run", "load_date": "2026-11-01", "metadata_key": "current_manifest"}
    original = frame.copy(deep=True)
    current = frame.copy(deep=True)
    current["feature_a"] += shift
    current_original = current.copy(deep=True)
    monkeypatch.setattr(drift_monitoring, "load_drift_data", lambda *args: {
        "reference_dataframe": frame, "current_dataframe": current,
        "feature_columns": ["feature_a", "feature_b"],
        "reference_metadata": {"run_id": "reference_run"}, "model_info": {"model_version": "1"},
    })
    client = Mock()
    monkeypatch.setenv("MINIO_BUCKET", "offline_bucket")
    monkeypatch.setattr(drift_monitoring, "create_minio_client", lambda: client)
    report = monitor_feature_drift(metadata, "unused_root", 1.0, 0.5, 2.0)
    client.put_object.assert_called_once()
    saved = json.loads(client.put_object.call_args.kwargs["Body"])
    assert saved == report
    assert report["summary"]["mean_shift_flag_count"] == (1 if shift else 0)
    assert report["summary"]["unavailable_mean_shift_count"] == 1
    output = capsys.readouterr().out
    assert "Saved feature drift report: monitoring/data_drift/" in output
    assert "Drift summary:" in output
    assert_frame_equal(frame, original)
    assert_frame_equal(current, current_original)


def test_monitor_entrypoint_does_not_upload_when_inputs_fail(monkeypatch):
    builder = Mock(side_effect=ValueError("invalid drift inputs"))
    client_factory = Mock()
    monkeypatch.setattr(drift_monitoring, "build_drift_report", builder)
    monkeypatch.setattr(drift_monitoring, "create_minio_client", client_factory)
    with pytest.raises(ValueError, match="invalid drift inputs"):
        monitor_feature_drift({}, "unused_root", 1.0, 0.5, 2.0)
    client_factory.assert_not_called()


def test_monitor_entrypoint_upload_failure_does_not_log_success(monkeypatch, capsys):
    monkeypatch.setattr(drift_monitoring, "build_drift_report", lambda *args: {"summary": {}})
    client = Mock()
    client.put_object.side_effect = RuntimeError("simulated upload failure")
    monkeypatch.setattr(drift_monitoring, "create_minio_client", lambda: client)
    monkeypatch.setenv("MINIO_BUCKET", "offline")
    with pytest.raises(RuntimeError, match="upload failure"):
        monitor_feature_drift({"run_id": "run", "load_date": "2026-11-01"}, "unused_root", 1.0, 0.5, 2.0)
    assert "Saved feature drift report" not in capsys.readouterr().out
