"""Model monitoring describes one validated current prediction per engine."""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.monitoring.model_monitoring import (
    summarize_prediction_batch, evaluate_prediction_accuracy, build_model_monitoring_report,
)


def test_prediction_summary_statistics_counts_and_json_types():
    predictions = pd.DataFrame({
        "engine_id": [1, 2, 3, 4, 5],
        "predicted_rul": [5.0, 10.0, 20.0, 30.0, 40.0],
        "status": ["critical", "critical", "warning", "warning", "healthy"],
        "model_version": ["2", "1", "2", "1", "1"],
    }, index=[9, 3, 8, 2, 1])
    original = predictions.copy(deep=True)
    report = summarize_prediction_batch(predictions)
    assert report == {
        "engine_count": 5,
        "mean_predicted_rul": 21.0,
        "median_predicted_rul": 20.0,
        "min_predicted_rul": 5.0,
        "max_predicted_rul": 40.0,
        "healthy_count": 1,
        "warning_count": 2,
        "critical_count": 2,
        "critical_percentage": 40.0,
        "model_versions": ["1", "2"],
    }
    for key in ("engine_count", "healthy_count", "warning_count", "critical_count"):
        assert type(report[key]) is int
    for key in ("mean_predicted_rul", "median_predicted_rul", "min_predicted_rul", "max_predicted_rul", "critical_percentage"):
        assert type(report[key]) is float
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert_frame_equal(predictions, original)


@pytest.mark.parametrize("status, rul, percentage", [
    ("healthy", 40.5, 0.0), ("warning", 20.5, 0.0), ("critical", 0.0, 100.0),
])
def test_single_engine_prediction_summary(status, rul, percentage):
    report = summarize_prediction_batch(pd.DataFrame({
        "engine_id": [1], "predicted_rul": [rul],
        "status": [status], "model_version": ["1"],
    }))
    assert report["engine_count"] == 1
    assert report[f"{status}_count"] == 1
    assert sum(report[f"{label}_count"] for label in ("healthy", "warning", "critical")) == 1
    for key in ("mean_predicted_rul", "median_predicted_rul", "min_predicted_rul", "max_predicted_rul"):
        assert report[key] == rul
    assert report["critical_percentage"] == percentage


def test_empty_prediction_batch_rejected():
    with pytest.raises(ValueError, match="empty prediction batch"):
        summarize_prediction_batch(pd.DataFrame())


@pytest.fixture
def evaluation_data():
    predictions = pd.DataFrame({
        "engine_id": [2, 1, 3], "cycle": [49, 31, 60],
        "predicted_rul": [16.0, 10.0, 30.0],
    }, index=[8, 2, 5])
    test_data = pd.DataFrame({
        "unit_number": [1, 2, 3, 1, 2, 3],
        "time_in_cycles": [1, 1, 1, 31, 49, 60],
    })
    labels = pd.DataFrame({
        "unit_number": [1, 3, 2], "remaining_useful_life": [13.0, 35.0, 20.0],
    }, index=[12, 4, 7])
    return predictions, test_data, labels


def test_complete_accuracy_matches_by_engine_and_preserves_inputs(evaluation_data):
    predictions, test_data, labels = evaluation_data
    originals = [frame.copy(deep=True) for frame in evaluation_data]
    report = evaluate_prediction_accuracy(predictions, test_data, labels)
    assert report["status"] == "evaluated"
    assert report["evaluated_engine_count"] == 3
    assert report["label_coverage_percentage"] == 100.0
    assert report["mae"] == 4.0
    assert report["rmse"] == pytest.approx((50 / 3) ** 0.5)
    assert type(report["evaluated_engine_count"]) is int
    for key in ("label_coverage_percentage", "mae", "rmse"):
        assert type(report[key]) is float
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    for frame, original in zip(evaluation_data, originals):
        assert_frame_equal(frame, original)


def test_partial_labels_report_coverage_and_only_matched_errors(evaluation_data):
    predictions, test_data, labels = evaluation_data
    labels = labels[labels["unit_number"].isin([1, 2])]
    report = evaluate_prediction_accuracy(predictions, test_data, labels)
    assert report["status"] == "partially_labeled"
    assert report["evaluated_engine_count"] == 2
    assert report["label_coverage_percentage"] == pytest.approx(200 / 3)
    assert report["mae"] == 3.5
    assert report["rmse"] == pytest.approx(12.5 ** 0.5)


@pytest.mark.parametrize("availability", ["none", "empty", "unmatched"])
def test_absent_labels_do_not_invent_accuracy(evaluation_data, availability):
    predictions, test_data, labels = evaluation_data
    if availability == "none":
        labels = None
    elif availability == "empty":
        labels = labels.iloc[:0]
    else:
        labels = pd.DataFrame({"unit_number": [99], "remaining_useful_life": [10.0]})
    assert evaluate_prediction_accuracy(predictions, test_data, labels) == {
        "status": "awaiting_labels", "evaluated_engine_count": 0,
        "label_coverage_percentage": 0.0, "mae": None, "rmse": None,
    }


def test_accuracy_rejects_empty_predictions(evaluation_data):
    predictions, test_data, labels = evaluation_data
    with pytest.raises(ValueError, match="empty"):
        evaluate_prediction_accuracy(predictions.iloc[:0], test_data, labels)


@pytest.mark.parametrize("problem", ["earlier_cycle", "missing_test_engine"])
def test_accuracy_rejects_wrong_label_cycle(evaluation_data, problem):
    predictions, test_data, labels = evaluation_data
    if problem == "earlier_cycle":
        predictions.loc[predictions["engine_id"].eq(1), "cycle"] = 30
    else:
        test_data = test_data[test_data["unit_number"].ne(1)]
    with pytest.raises(ValueError, match="final observed test cycle"):
        evaluate_prediction_accuracy(predictions, test_data, labels)


@pytest.mark.parametrize("side", ["predictions", "labels"])
def test_accuracy_rejects_ambiguous_engine_matches(evaluation_data, side):
    predictions, test_data, labels = evaluation_data
    if side == "predictions":
        predictions = pd.concat([predictions, predictions.iloc[:1]])
    else:
        labels = pd.concat([labels, labels.iloc[:1].assign(remaining_useful_life=100.0)])
    with pytest.raises(pd.errors.MergeError):
        evaluate_prediction_accuracy(predictions, test_data, labels)


@pytest.mark.parametrize("side", ["predictions", "labels"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1.0])
def test_accuracy_rejects_invalid_matched_rul_values(evaluation_data, side, value):
    predictions, test_data, labels = evaluation_data
    if side == "predictions":
        predictions.loc[predictions.index[0], "predicted_rul"] = value
    else:
        labels.loc[labels.index[0], "remaining_useful_life"] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        evaluate_prediction_accuracy(predictions, test_data, labels)


def test_zero_and_fractional_labels_are_valid(evaluation_data):
    predictions, test_data, labels = evaluation_data
    predictions["predicted_rul"] = [0.0, 0.5, 1.0]
    labels["remaining_useful_life"] = labels["unit_number"].map({1: 0.5, 2: 0.0, 3: 1.0})
    report = evaluate_prediction_accuracy(predictions, test_data, labels)
    assert report["mae"] == report["rmse"] == 0.0


@pytest.mark.parametrize("availability", ["complete", "partial", "none", "empty", "unmatched"])
def test_combined_report_keeps_summary_with_optional_accuracy(evaluation_data, availability):
    predictions, test_data, labels = evaluation_data
    predictions = predictions.assign(status=["warning", "critical", "warning"], model_version="1")
    metadata = {"run_id": "offline_run", "load_date": "2026-11-02", "metadata_key": "offline_manifest"}
    if availability == "partial":
        labels = labels[labels["unit_number"].isin([1, 2])]
    elif availability == "none":
        labels = None
    elif availability == "empty":
        labels = labels.iloc[:0]
    elif availability == "unmatched":
        labels = pd.DataFrame({"unit_number": [99], "remaining_useful_life": [10.0]})
    originals = [frame.copy(deep=True) for frame in (predictions, test_data)]
    labels_original = labels.copy(deep=True) if labels is not None else None
    label_source = "fd001_benchmark" if availability in ("complete", "partial") else None
    report = build_model_monitoring_report(predictions, test_data, metadata, labels, label_source)
    assert report["prediction_summary"] == summarize_prediction_batch(predictions)
    assert report["accuracy"] == evaluate_prediction_accuracy(predictions, test_data, labels)
    assert report["label_source"] == label_source
    assert report["accuracy_scope"] == "final_observed_test_cycle"
    assert report["current_metadata"] == metadata
    assert report["current_metadata"] is not metadata
    assert datetime.fromisoformat(report["monitored_at"]).utcoffset() == timezone.utc.utcoffset(None)
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    for frame, original in zip((predictions, test_data), originals):
        assert_frame_equal(frame, original)
    if labels is not None:
        assert_frame_equal(labels, labels_original)


@pytest.mark.parametrize("label_source", [None, "", "   ", 42])
def test_evaluated_model_report_requires_label_provenance(evaluation_data, label_source):
    predictions, test_data, labels = evaluation_data
    predictions = predictions.assign(status=["warning", "critical", "warning"], model_version="1")
    with pytest.raises(ValueError, match="nonempty label_source"):
        build_model_monitoring_report(predictions, test_data, {}, labels, label_source)


def test_unavailable_accuracy_does_not_claim_a_supplied_source(evaluation_data):
    predictions, test_data, _ = evaluation_data
    predictions = predictions.assign(status=["warning", "critical", "warning"], model_version="1")
    report = build_model_monitoring_report(
        predictions, test_data, {}, rul_data=None, label_source="fd001_benchmark"
    )
    assert report["label_source"] is None
    assert report["accuracy"]["status"] == "awaiting_labels"
    assert report["prediction_summary"]["engine_count"] == 3
