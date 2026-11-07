from src.ingestion.source_checks import check_source_data
from datetime import datetime, timezone, timedelta
from airflow.sdk import dag, task, get_current_context
from src.ingestion.raw_ingestion import ingest_raw_data
from src.transformations.raw_to_silver import transform_to_silver
from src.transformations.silver_to_features import generate_features
from src.validation.feature_validation import validate_features
from src.modeling.batch_inference import predict_current_health
from src.transformations.publish_health import publish_dashboard_dataset
from src.monitoring.data_monitoring import monitor_raw_data
from src.monitoring.drift_monitoring import monitor_feature_drift
from src.monitoring.model_monitoring import monitor_model

@dag(
    dag_id="cmapss_data_prediction",
    start_date=datetime(2026, 10, 22, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["cmapss"],
)
def predictive_maintenance_pipeline():

    @task(task_id="check_source_data")
    def check_source():
        source_files = check_source_data("/opt/cmapss")
        print(source_files)
        return source_files

    @task(task_id="ingest_raw_data",retries=1,retry_delay=timedelta(seconds=30),)
    def ingest_raw(source_files):
        context = get_current_context()
        pipeline_timestamp = context["dag_run"].start_date.astimezone(timezone.utc)

        pipeline_run_id = pipeline_timestamp.strftime("%Y%m%dT%H%M%S%fZ")
        pipeline_load_date = pipeline_timestamp.strftime("%Y-%m-%d")

        # Call ingest_raw_data with the three arguments and return its result.
        metadata = ingest_raw_data(source_files, pipeline_run_id, pipeline_load_date)
        return metadata

    @task(
        task_id="monitor_raw_quality",
        retries=1,
        retry_delay=timedelta(seconds=30),
    )
    def monitor_raw_quality(metadata):
        report = monitor_raw_data(metadata)

        print(
            f"Raw quality checks passed: {report['raw_checks_passed']}",
            flush=True,
        )

        if not report["raw_checks_passed"]:
            raise ValueError(
                f"Raw quality checks failed for run {metadata['run_id']}. "
                "See the saved monitoring report."
            )

        return metadata

    @task(
        task_id="transform_to_silver",
        retries=1,
        retry_delay=timedelta(seconds=30),
    )
    def make_silver(metadata):
        return transform_to_silver(metadata)

    @task(
        task_id="generate_features",
        retries=1,
        retry_delay=timedelta(seconds=30),
    )
    def make_features(metadata):
        return generate_features(metadata)

    @task(task_id="validate_features")
    def check_features(metadata):
        return validate_features(metadata)

    @task(
        task_id="monitor_feature_drift",
        retries=1,
        retry_delay=timedelta(seconds=30),
    )
    def monitor_drift(metadata):
        monitor_feature_drift(
            metadata,
            project_root="/opt/cmapss",
            mean_shift_threshold=1.0,
            lower_ratio=0.5,
            upper_ratio=2.0,
        )
        return metadata
    @task(
        task_id="predict_current_health",
        retries=1,
        retry_delay=timedelta(seconds=30),
    )
    def predict_health(metadata):
        return predict_current_health(metadata)

    @task(
        task_id="monitor_model",
        retries=1,
        retry_delay=timedelta(seconds=30),
    )
    def monitor_model_results(metadata):
        monitor_model(metadata, include_benchmark_labels=True)
        return metadata
    
    @task(
        task_id="publish_dashboard_dataset",
        retries=1,
        retry_delay=timedelta(seconds=30),
    )
    def publish_health(metadata):
        return publish_dashboard_dataset(metadata)
        
    source_files = check_source()
    raw_metadata = ingest_raw(source_files)
    # NEW: Monitoring saves the validation report and is the single Raw gate.
    monitored_metadata = monitor_raw_quality(raw_metadata)
    silver_metadata = make_silver(monitored_metadata)
    gold_metadata = make_features(silver_metadata)
    validated_gold = check_features(gold_metadata)
    drift_metadata = monitor_drift(validated_gold)
    prediction_metadata = predict_health(drift_metadata)
    model_monitoring_metadata = monitor_model_results(prediction_metadata)
    publish_health(model_monitoring_metadata)

predictive_maintenance_pipeline()
