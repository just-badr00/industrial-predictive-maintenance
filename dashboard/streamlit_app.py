import streamlit as st
import requests
import json
from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import os
from io import BytesIO
from datetime import datetime, timezone
import boto3
from botocore.client import Config
from botocore.exceptions import EndpointConnectionError, ClientError  # MinIO error responses.
from dotenv import load_dotenv

def read_parquet_prefix(s3_client, bucket_name, prefix):
    """
    Read every Parquet object under a MinIO prefix
    and return one pandas DataFrame.
    """
    #MinIO may contain many objects. Instead of returning all objects in one response, the S3 API may divide them into pages.
    paginator = s3_client.get_paginator("list_objects_v2")

    #This list will store the names of all Parquet objects found.
    parquet_keys = []

    #This asks MinIO: Show me all objects inside this bucket whose key starts with this prefix.
    for page in paginator.paginate(Bucket=bucket_name,Prefix=prefix):
        #Contents contains the objects found by MinIO.
        for obj in page.get("Contents", []):
            #The key is the complete MinIO path of the file.
            key = obj["Key"]

            if key.endswith(".parquet"):
                parquet_keys.append(key)

    if not parquet_keys:
        raise FileNotFoundError(
            f"No Parquet files found under prefix: {prefix}"
        )

    dataframes = []

    for key in parquet_keys:
        response = s3_client.get_object(
            Bucket=bucket_name,
            Key=key
        )

        parquet_bytes = response["Body"].read()

        dataframe = pd.read_parquet(
            BytesIO(parquet_bytes),
            engine="fastparquet"
        )

        dataframes.append(dataframe)

        print(f"Loaded: {key}")
        print(f"Shape: {dataframe.shape}")

    combined_dataframe = pd.concat(
        dataframes,
        ignore_index=True
    )

    return combined_dataframe

def read_parquet_object(s3_client, bucket_name, object_key):
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    return pd.read_parquet(
        BytesIO(response["Body"].read()),
        engine="fastparquet",
    )

def load_latest_airflow_health(s3_client, bucket_name):
    metadata_df = read_parquet_prefix(
        s3_client,
        bucket_name,
        "metadata/etl_metadata/",
    )

    published = metadata_df[
        (metadata_df["layer"] == "gold")
        & (metadata_df["dataset_name"] == "current_engine_health")
        & (metadata_df["status"] == "success")
    ]

    if published.empty:
        raise FileNotFoundError(
            "No published Airflow engine-health dataset was found."
        )

    latest_run_id = published["run_id"].max()
    selected = published[published["run_id"] == latest_run_id]

    if len(selected) != 1:
        raise ValueError(
            "Expected one published health dataset for the latest run."
        )

    record = selected.iloc[0]
    health_df = read_parquet_object(
        s3_client,
        bucket_name,
        record["object_key"],
    )

    if len(health_df) != record["row_count"]:
        raise ValueError("Health dataset row count differs from its manifest.")

    return health_df, record

def read_quality_report(s3_client, bucket_name, run_id, load_date):
    object_key = (
        f"monitoring/data_quality/load_date={load_date}/"
        f"run_id={run_id}/raw_quality_report.json"
    )

    response = s3_client.get_object(Bucket = bucket_name,Key = object_key)
    response_text = response["Body"].read().decode("utf-8")
    report_dict = json.loads(response_text)
    return report_dict

def read_drift_report(s3_client, bucket_name, run_id, load_date):
    object_key = (
        f"monitoring/data_drift/load_date={load_date}/"
        f"run_id={run_id}/feature_drift_report.json"
    )
    response = s3_client.get_object(Bucket = bucket_name,Key = object_key)
    response_text = response["Body"].read().decode("utf-8")
    report_dict = json.loads(response_text)
    return report_dict

def read_model_monitoring_report(s3_client, bucket_name, run_id, load_date):
    object_key = (
        f"monitoring/model/load_date={load_date}/"
        f"run_id={run_id}/model_monitoring_report.json"
    )
    response = s3_client.get_object(Bucket = bucket_name,Key = object_key)
    response_text = response["Body"].read().decode("utf-8")
    report_dict = json.loads(response_text)
    return report_dict

api_url = "http://127.0.0.1:8001"
project_root = Path(r"C:\Users\Badr\Desktop\Python_3.0")

# Initialize these values only once for this browser session.
if "health_data" not in st.session_state:
    st.session_state["health_data"] = None

if "model_info" not in st.session_state:
    st.session_state["model_info"] = None

if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

# Keep the batch prediction results between reruns.
if "fleet_predictions" not in st.session_state:
    st.session_state["fleet_predictions"] = None

try:

    example_path = project_root / "docs" / "examples" / "predict_batch_engines_1_2_3.json"
    with open(example_path, "r", encoding="utf-8") as file:
        example_data = json.load(file)

except OSError:
    st.error("Could not read the engine history file. Check docs/examples/predict_batch_engines_1_2_3.json.")
    st.stop()
except ValueError:
    st.error("The file contains invalid JSON.")
    st.stop()

page = st.sidebar.radio(
    "Choose a view",
    ["Overview", "Engine Explorer","Maintenance Queue","Model Performance","Data Quality"],
)

if page in ["Overview", "Maintenance Queue"]:
    prediction_source = st.sidebar.radio(
        "Prediction source",
        ["Demo engines", "Latest Airflow run"],
    )

st.title("Industrial Predictive Maintenance")
st.write("Explore engine sensor data and predicted remaining useful life.")
if page == "Engine Explorer" or page in ["Overview", "Maintenance Queue"]and prediction_source == "Demo engines":
    st.caption("Demo histories: engine 1 through cycle 31, ""engine 2 through cycle 49, and engine 3 through cycle 126.")
st.subheader("API connection")

# Only send requests on the rerun triggered by clicking this button.
if st.button("Check API connection"):
    # Clear the old result so a failed new check cannot show an old success.
    st.session_state["health_data"] = None
    st.session_state["model_info"] = None

    try:
        health_response = requests.get(f"{api_url}/health", timeout=10)
        # An HTTP error such as 503 should enter the error handling below.
        health_response.raise_for_status()
        health_data = health_response.json()

        model_response = requests.get(f"{api_url}/model-info", timeout=10)
        model_response.raise_for_status()
        model_info = model_response.json()
    except requests.exceptions.Timeout:
        st.error("The API took too long to respond. Try again shortly.")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to the API. Check that maintenance-api is running on port 8001.")
    except requests.exceptions.HTTPError as error:
        st.error(f"The API returned HTTP {error.response.status_code}. Check the API logs for details.")
    except requests.exceptions.RequestException:
        st.error("The API request failed. Try again shortly.")
    except ValueError:
        st.error("The API response could not be read as JSON.")
    else:
        # Save successful responses so later reruns can use them.
        st.session_state["health_data"] = health_data
        st.session_state["model_info"] = model_info

# Display saved results on every rerun, independently of the button.
if st.session_state["health_data"] is not None and st.session_state["model_info"] is not None:
    st.success("The last API check succeeded.")
    st.caption("Saved responses from the last check. Click the button to check again.")
    st.write("Health response")
    st.json(st.session_state["health_data"])
    st.write("Model information")
    st.json(st.session_state["model_info"])

# Engine Explorer - selection, sensor trends, prediction, and SHAP factors.
if page == "Engine Explorer":
    st.subheader("Engine Explorer")

    # The dropdown returns the selected value to our Python variable.
    # Remember the selected engine when switching views.
    if "selected_engine_id" not in st.session_state:
        st.session_state["selected_engine_id"] = 1

    engine_id = st.selectbox(
        "Select an engine",
        [1, 2, 3],
        index=[1, 2, 3].index(st.session_state["selected_engine_id"]),
    )
    st.session_state["selected_engine_id"] = engine_id

    # A saved prediction belongs to the engine whose history was submitted.
    if st.session_state["prediction"] is not None:
        if st.session_state["prediction"]["engine_id"] != engine_id:
            st.session_state["prediction"] = None

    # Changing the selection reruns this script with the new engine_id.
    st.write("Selected engine:", engine_id)

    st.subheader("Sensor trends")

    # Load the selected history outside the button so charts appear before prediction.

    prediction_request = None
    for engine in example_data["engines"]:
        if engine["engine_id"] == engine_id:
            prediction_request = engine
            break

    if prediction_request is None:
        st.error("The selected engine is missing from the example file.")
        st.stop()

    # Each dictionary in history becomes one row in the DataFrame.
    history_df = pd.DataFrame(prediction_request["history"])
    sensor_columns = [
        column for column in history_df.columns
        if column.startswith("sensor_measurement_")
    ]
    selected_sensor = st.selectbox("Select a sensor", sensor_columns, index=2)

    st.write("Observed cycles:", len(history_df), "| Latest cycle:", int(history_df["time_in_cycles"].max()))
    st.line_chart(
        history_df,
        x="time_in_cycles",
        y=selected_sensor,
        x_label="Operating cycle",
        y_label=selected_sensor,
    )
    st.caption("Observed sensor measurements from the selected demo history.")

    st.subheader("Engine prediction")

    if st.button(f"Predict RUL for engine {engine_id}"):
        st.session_state["prediction"] = None

        try:
            # POST sends the engine history in the request body as JSON.
            prediction_response = requests.post(
                f"{api_url}/predict",
                json=prediction_request,
                timeout=120,
            )
            prediction_response.raise_for_status()
            prediction = prediction_response.json()
        except requests.exceptions.Timeout:
            st.error("The prediction took too long. Try again shortly.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Check that maintenance-api is running on port 8001.")
        except requests.exceptions.HTTPError as error:
            st.error(f"Prediction failed with HTTP {error.response.status_code}. Check the API logs for details.")
        except requests.exceptions.RequestException:
            st.error("The prediction request failed. Try again shortly.")
        except ValueError:
            st.error("The API response could not be read as JSON.")
        else:
            st.session_state["prediction"] = prediction

    if st.session_state["prediction"] is not None:
        prediction = st.session_state["prediction"]
        st.write("Engine:", prediction["engine_id"], "| Cycle:", prediction["cycle"])
        st.metric("Predicted remaining useful life (cycles)", f"{prediction['predicted_rul']:.1f}")
        st.write("Health status:", prediction["status"])
        st.write("Model version:", prediction["model_version"])

        # Display SHAP factors in a readable table.
        st.write("Top prediction factors")

        factor_rows = []
        for factor in prediction["top_factors"]:
            if factor["shap_value"] > 0:
                effect = "Raises predicted RUL"
            elif factor["shap_value"] < 0:
                effect = "Lowers predicted RUL"
            else:
                effect = "No contribution"

            factor_rows.append({
                "Feature": factor["feature"],
                "Feature value": factor["value"],
                "SHAP contribution (cycles)": factor["shap_value"],
                "Effect": effect,
            })

        factors_df = pd.DataFrame(factor_rows)
        st.dataframe(factors_df, hide_index=True, width="stretch")
        st.caption(
            "The three strongest influences are shown, ranked by absolute SHAP contribution. "
            "Positive contributions raise predicted RUL; negative contributions lower it "
            "relative to the model baseline. These are model influences, not maintenance diagnoses."
        )

if page in ["Overview", "Maintenance Queue"] and prediction_source == "Latest Airflow run":
    st.subheader("Airflow engine-health snapshot")

    load_dotenv(project_root / ".env")

    try:
        bucket_name = os.environ["MINIO_BUCKET"]

        s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ["MINIO_ENDPOINT"],
            aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
            aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
            region_name="us-east-1",
            config=Config(signature_version="s3v4"),
        )

        airflow_health, publication_record = load_latest_airflow_health(
            s3_client,
            bucket_name,
        )

    except (
        EndpointConnectionError, ClientError,
        FileNotFoundError, ValueError, KeyError,
    ) as error:
        st.error(f"Could not load the Airflow snapshot: {error}")
        st.stop()

    st.write("Pipeline run:", publication_record["run_id"])
    published_at = pd.to_datetime(publication_record["recorded_at"],utc=True,)
    st.write("Published at (UTC):",published_at.strftime("%Y-%m-%d %H:%M:%S"),)
    st.write("Published engine count:", len(airflow_health))

    if page == "Overview":
        healthy_count = (airflow_health["status"] == "healthy").sum()
        warning_count = (airflow_health["status"] == "warning").sum()
        critical_count = (airflow_health["status"] == "critical").sum()
        average_rul = airflow_health["predicted_rul"].mean()
        model_versions = ", ".join(sorted(airflow_health["model_version"].unique()))

        # Each returned column is a container in which we can display a metric.
        engines_column, healthy_column, warning_column = st.columns(3)
        engines_column.metric("Published engines", len(airflow_health))
        healthy_column.metric("Healthy", int(healthy_count))
        warning_column.metric("Warning", int(warning_count))

        critical_column, rul_column, version_column = st.columns(3)
        critical_column.metric("Critical", int(critical_count))
        rul_column.metric("Average predicted RUL (cycles)", f"{average_rul:.1f}")
        version_column.metric("Model version(s)", model_versions)
        st.caption("Saved batch results for the latest observed cycle of each published engine.")

    display_health = airflow_health.copy()

    if page == "Maintenance Queue":
        health_status = st.selectbox("Filter by health status", ["all", "healthy", "warning","critical"])
        if health_status != "all":
            display_health = display_health[display_health["status"]==health_status]

    if display_health.empty:
        st.info("No engines match the selected health status.")
    else:
        st.dataframe(
            display_health[
                ["engine_id", "cycle", "predicted_rul",
                "status", "priority", "model_version"]
            ],
            hide_index=True,
            width="stretch",
        )

    st.caption("Saved results from the latest published Airflow run.")
    st.stop()

# Predict all three demo engines in one batch request.
if page == "Overview":
    st.subheader("Overview - demo engines")
    st.caption("Summary of the three saved engine histories. Click the button to update predictions.")

    if st.button("Predict all demo engines"):
        st.session_state["fleet_predictions"] = None

        try:
            # example_data already contains {"engines": [...]} for all three histories.
            batch_response = requests.post(
                f"{api_url}/predict/batch",
                json=example_data,
                timeout=120,
            )
            batch_response.raise_for_status()
            fleet_predictions = batch_response.json()
        except requests.exceptions.Timeout:
            st.error("Batch prediction took too long. Try again shortly.")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Check that maintenance-api is running on port 8001.")
        except requests.exceptions.HTTPError as error:
            st.error(f"Batch prediction failed with HTTP {error.response.status_code}. Check the API logs for details.")
        except requests.exceptions.RequestException:
            st.error("The batch prediction request failed. Try again shortly.")
        except ValueError:
            st.error("The batch API response could not be read as JSON.")
        else:
            st.session_state["fleet_predictions"] = fleet_predictions

    if st.session_state["fleet_predictions"] is not None:
        fleet_df = pd.DataFrame(st.session_state["fleet_predictions"])

        # Each comparison produces True/False values; sum counts the True values.
        healthy_count = (fleet_df["status"] == "healthy").sum()
        warning_count = (fleet_df["status"] == "warning").sum()
        critical_count = (fleet_df["status"] == "critical").sum()
        average_rul = fleet_df["predicted_rul"].mean()
        model_versions = ", ".join(sorted(fleet_df["model_version"].unique()))

        # Each returned column is a container in which we can display a metric.
        engines_column, healthy_column, warning_column = st.columns(3)
        engines_column.metric("Demo engines", len(fleet_df))
        healthy_column.metric("Healthy", int(healthy_count))
        warning_column.metric("Warning", int(warning_count))

        critical_column, rul_column, version_column = st.columns(3)
        critical_column.metric("Critical", int(critical_count))
        rul_column.metric("Average predicted RUL (cycles)", f"{average_rul:.1f}")
        version_column.metric("Model version(s)", model_versions)
        st.caption("Saved batch results for the latest observed cycle of each demo history; not continuous monitoring.")

if page == "Maintenance Queue":
    st.subheader("Maintenance Queue - demo engines")

    if st.session_state["fleet_predictions"] is None:
        st.info("Go to Overview and click Predict all demo engines to populate the maintenance queue.")
    else:
        fleet_df = pd.DataFrame(st.session_state["fleet_predictions"])
        maintenance_queue = fleet_df[["engine_id", "predicted_rul", "status"]].copy()

        # This is a simple display rule based on the API's existing health status.
        priority_by_status = {
            "critical": "High",
            "warning": "Medium",
            "healthy": "Low",
        }
        maintenance_queue["priority"] = maintenance_queue["status"].map(priority_by_status)

        # Lower predicted RUL appears first; engine ID breaks any ties.
        maintenance_queue = maintenance_queue.sort_values(
            ["predicted_rul", "engine_id"]
        ).reset_index(drop=True)

        health_status = st.selectbox("Filter by health status", ["all", "healthy", "warning","critical"])

        if health_status == "all":
            pass
        else:
            maintenance_queue = maintenance_queue[maintenance_queue["status"]==health_status]
        maintenance_queue = maintenance_queue.rename(columns={
            "engine_id": "Engine",
            "predicted_rul": "Predicted RUL (cycles)",
            "status": "Status",
            "priority": "Priority",
        })

        if maintenance_queue.empty:
            st.info("No engines match the selected health status.")
        else:
            st.dataframe(maintenance_queue, hide_index=True, width="stretch")
        st.caption(
            "Lowest predicted RUL first. Priority rule: critical = High, warning = Medium, healthy = Low. "
            "This table uses the same saved batch results as the Overview."
        )

if page == "Model Performance":
    # NEW: Show the report for the same published batch as the fleet views.
    st.subheader("Latest published batch monitoring")

    try:
        load_dotenv(project_root / ".env")
        BUCKET_NAME = os.environ["MINIO_BUCKET"]
        s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ["MINIO_ENDPOINT"],
            aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
            aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
            region_name="us-east-1",
            config=Config(signature_version="s3v4"),
        )
        _, publication_record = load_latest_airflow_health(
            s3_client, BUCKET_NAME
        )

        model_report = read_model_monitoring_report(
            s3_client,
            BUCKET_NAME,
            publication_record["run_id"],
            publication_record["load_date"],
        )
    except EndpointConnectionError:
        st.warning("Cannot connect to MinIO to read batch monitoring.")
    except FileNotFoundError:
        st.info("No published Airflow batch was found in MinIO.")
    except ClientError as error:
        if error.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
            st.info("The saved batch snapshot or model monitoring report is unavailable.")
        else:
            st.warning("MinIO rejected the batch monitoring request.")
    except ValueError:
        st.warning("The published batch or model monitoring report could not be read.")
    except KeyError:
        st.warning("Batch monitoring configuration or publication metadata is incomplete.")
    else:
        prediction_summary = model_report["prediction_summary"]
        accuracy = model_report["accuracy"]
        prediction_timing = model_report["current_metadata"].get("prediction_timing")
        accuracy_alert = model_report.get("accuracy_alert")
        st.write("Pipeline run", publication_record["run_id"])
        st.write("Monitored at (UTC)", model_report["monitored_at"])
        st.write("Monitored engines", prediction_summary["engine_count"])
        st.write("Model versions", ", ".join(prediction_summary["model_versions"]))
        st.write(
            f"{accuracy['evaluated_engine_count']} of {prediction_summary['engine_count']} "
            f"engines evaluated ({accuracy['label_coverage_percentage']:.1f}% label coverage)."
        )

        if prediction_timing is not None:
            count_col, average_col, slowest_col = st.columns(3)
            count_col.metric("Batch requests", prediction_timing["batch_request_count"])
            average_col.metric(
                "Average request time (ms)",
                f"{prediction_timing['mean_batch_request_ms']:.0f}",
            )
            slowest_col.metric(
                "Slowest request time (ms)",
                f"{prediction_timing['max_batch_request_ms']:.0f}",
            )

        if accuracy["status"] in ("evaluated", "partially_labeled"):
            mae_column, rmse_column = st.columns(2)
            mae_column.metric("Batch MAE (cycles)", f"{accuracy['mae']:.2f}")
            rmse_column.metric("Batch RMSE (cycles)", f"{accuracy['rmse']:.2f}")
            st.caption(
                f"Label source {model_report['label_source']}. Errors are calculated "
                "at the final observed test cycle of each evaluated engine."
            )
        else:
            st.info("Accuracy will appear when matching RUL labels are available.")

        if accuracy_alert is not None and accuracy_alert["status"] == "above_limit":
            st.warning(f"The observed MAE ({accuracy_alert["mae_cycles"]}) is above the MAE limit ({accuracy_alert["limit_cycles"]})")
    # END NEW: Latest batch monitoring.

    st.subheader("Historical MLflow evaluation")

    mlflow_database = (
        project_root / "mlflow.db"
    ).resolve().as_posix()

    if not Path(mlflow_database).is_file():
        st.error("The MLflow database file is missing. Check that mlflow.db exists in the project root.")
        st.stop()

    tracking_uri = f"sqlite:///{mlflow_database}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = MlflowClient()

    registered_model_name = "cmapss_fd001_rul_random_forest"
    
    model_version = 1

    try:
        model_version_details = client.get_model_version(
            name=registered_model_name,
            version=model_version,
        )
    except MlflowException :
        st.error(f"Could not retrieve model version {model_version} for registered model '{registered_model_name}'. Check that the model exists in MLflow.")
        st.stop()

    source_run_id = model_version_details.run_id

    if source_run_id is None:
        st.error("The registered model version has no source run.")
        st.stop()

    try :
        source_run = client.get_run(source_run_id)
    except MlflowException:
        st.error(f"Could not retrieve the source run with ID '{source_run_id}'. Check that the run exists in MLflow.")
        st.stop()

    metrics = source_run.data.metrics
    wanted_cycles = st.selectbox("Select evaluation scope", ["Final observed cycle per engine", "All eligible test cycles"])

    if wanted_cycles == "All eligible test cycles":
        st.caption("Saved evaluation on all eligible cycles of the 100 FD001 test engines(11,196 observations, starting at cycle 20).")
        mae = metrics["mae"]
        rmse = metrics["rmse"]
        r2 = metrics["r2"]
    else:
        mae = metrics["official_test_endpoint_mae"]
        rmse = metrics["official_test_endpoint_rmse"]
        r2 = metrics["official_test_endpoint_r2"]
        st.caption("Saved evaluation on the final observed cycle of each of the 100 FD001 test engines.")        

    mae_column, rmse_column, r2_column = st.columns(3)
    mae_column.metric("MAE(cycles)", f"{mae:.2f}")
    rmse_column.metric("RMSE(cycles)", f"{rmse:.2f}")
    r2_column.metric("R2", f"{r2:.3f}")

    # Choose one saved engine curve using a label-to-artifact dictionary.
    st.subheader("Actual versus predicted RUL")

    plot_options = {
        "Best engine": "plots/best_engine.png",
        "Middle-ranked engine": "plots/average_engine.png",
        "Worst engine": "plots/worst_engine.png",
    }
    selected_plot = st.selectbox("Select an engine curve", list(plot_options.keys()))
    artifact_path = plot_options[selected_plot]

    try :
        plot_path = mlflow.artifacts.download_artifacts(
            run_id=source_run_id,
            artifact_path=artifact_path,
        )
    except (MlflowException, OSError):
        st.warning(f"Could not download the plot '{selected_plot}'. Check that the artifact exists in MLflow.")

    else:
        st.image(
            plot_path,
            caption=f"{selected_plot}: saved actual-versus-predicted RUL curve",
            width="stretch",
        )
        st.caption(
            "Engines are ranked by MAE across their eligible test cycles. "
            "Each plot shows one engine's full evaluated trajectory, starting at cycle 20. "
            "The middle-ranked engine is not an average curve across engines. "
            "These saved trajectories do not change with the evaluation scope selected above."
        )

    # Display the residual histogram already saved during Phase 7.
    st.subheader("Prediction error distribution")

    try:
        residual_plot_path = mlflow.artifacts.download_artifacts(
            run_id=source_run_id,
            artifact_path="phase_7/plots/residual_distribution.png",
        )
    except (MlflowException, OSError):
        st.warning("Could not download the residual distribution plot. Check that the artifact exists in MLflow.")

    else:
        st.image(
            residual_plot_path,
            caption="Saved distribution of prediction residuals",
            width="stretch",
        )
        st.caption(
            "Residual = actual RUL minus predicted RUL, measured in cycles. "
            "Negative residuals mean remaining life was overestimated (optimistic); "
            "positive residuals mean it was underestimated (pessimistic). "
            "Values near zero indicate small errors. This histogram covers all 11,196 "
            "eligible test observations, regardless of the metric evaluation scope selected above."
        )

    # Historical baseline comparison from Notebook 06's saved outputs.
    st.subheader("Phase 5 baseline comparison")

    baseline_results = [
        {
            "Model": "Mean RUL",
            "MAE (cycles)": 52.27230702650257,
            "RMSE (cycles)": 63.15659578065061,
            "R²": -0.0002068681218732049,
        },
        {
            "Model": "Linear Regression",
            "MAE (cycles)": 26.975360864588414,
            "RMSE (cycles)": 36.325707664853546,
            "R²": 0.6691123380062189,
        },
        {
            "Model": "Random Forest",
            "MAE (cycles)": 26.280738453276047,
            "RMSE (cycles)": 38.858964457592464,
            "R²": 0.6213527103303564,
        },
        {
            "Model": "XGBoost",
            "MAE (cycles)": 26.714122772216797,
            "RMSE (cycles)": 39.18028180631011,
            "R²": 0.6150648593902588,
        },
    ]
    baseline_comparison_df = pd.DataFrame(baseline_results)
    st.dataframe(baseline_comparison_df, hide_index=True, width="stretch")
    st.caption(
        "Historical results copied from the saved outputs of 06_baseline_models.ipynb. "
        "All four models used the same 80-engine training / 20-engine validation split, "
        "with evaluation across eligible validation cycles starting at cycle 20. "
        "These are Phase 5 baseline results, not the final model's official-test results, "
        "and they do not change with the evaluation scope selected above."
    )
    st.write(
        "Random Forest advanced because it balanced MAE, prediction stability, "
        "and accuracy near failure. Linear Regression achieved better overall RMSE and R². "
        "The final registered Random Forest was subsequently tuned and trained in Phase 6."
    )

if page == "Data Quality":
    st.subheader("Data Quality")

    load_dotenv(project_root / ".env")

    BUCKET_NAME = os.environ["MINIO_BUCKET"]

    s3_client = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
        region_name="us-east-1",
        config=Config(signature_version="s3v4"),
    )

    try:
        metadata_df = read_parquet_prefix(
            s3_client,
            BUCKET_NAME,
            "metadata/etl_metadata/"
        )
    except EndpointConnectionError:
        st.error("Cannot connect to MinIO. Check that minio-cmapss is running and the configured endpoint is correct.")
        st.stop()
    except FileNotFoundError:
        st.error("No ETL manifests were found in MinIO.")
        st.stop()
    # Handle error responses while listing or reading manifests.
    except ClientError:
        st.error("MinIO rejected the metadata request. Check the configured bucket, access permissions, and manifest objects.")
        st.stop()

    required_raw_datasets = {"train_data", "test_data", "RUL_data"}
    successful_raw_metadata = metadata_df[
        (metadata_df["layer"] == "raw")
        & (metadata_df["status"] == "success")
        & metadata_df["dataset_name"].isin(required_raw_datasets)
    ]

    datasets_by_run = successful_raw_metadata.groupby("run_id")["dataset_name"].apply(set)
    valid_raw_run_ids = sorted(
        run_id
        for run_id, datasets in datasets_by_run.items()
        if required_raw_datasets.issubset(datasets)
    )

    if not valid_raw_run_ids:
        st.error("No successful Raw run contains train, test, and RUL data.")
        st.stop()

    pipeline_run_id = valid_raw_run_ids[-1]
    selected_raw_metadata = successful_raw_metadata[
        successful_raw_metadata["run_id"] == pipeline_run_id
    ]

    st.write("Selected Raw pipeline run ID:", pipeline_run_id)
    st.dataframe(
        selected_raw_metadata,
        hide_index=True,
        width="stretch",
    )

    latest_record_time = pd.to_datetime(selected_raw_metadata["recorded_at"], utc=True).max()
    st.write("Latest record time in this Raw run:", latest_record_time.strftime("%Y-%m-%d %H:%M:%S.%f %Z"))

    row_count = selected_raw_metadata.groupby("dataset_name")["row_count"].first().to_dict()
    st.write("Row counts by dataset in this Raw run:", row_count)

    st.subheader("Saved Raw monitoring report")

    load_date = selected_raw_metadata["load_date"].iloc[0]

    try:
        quality_report = read_quality_report(
            s3_client, BUCKET_NAME, pipeline_run_id, load_date
        )

    except EndpointConnectionError:
        st.warning("Cannot connect to MinIO to read the monitoring report.")

    except ClientError as error:
        error_code = error.response["Error"]["Code"]

        if error_code in ("NoSuchKey", "404", "NotFound"):
            st.info("No saved monitoring report exists for this Raw run.")
        else:
            st.warning("MinIO rejected the monitoring report request.")

    except ValueError:
        st.warning("The saved monitoring report could not be decoded.")

    else:
        if quality_report["raw_checks_passed"]:
            st.success("All monitored Raw quality checks passed.")
        else:
            st.error("Raw quality checks failed. Inspect the report below.")

        st.write("Monitored at (UTC):", quality_report["monitored_at"])

        with st.expander("View complete monitoring report"):
            st.json(quality_report)

    st.subheader("Data drift")

    try:
        drift_report = read_drift_report(s3_client, BUCKET_NAME, pipeline_run_id, load_date)
    except EndpointConnectionError:
            st.warning("Cannot connect to MinIO to read the monitoring report.")
    
    except ClientError as error:
        error_code = error.response["Error"]["Code"]

        if error_code in ("NoSuchKey", "404", "NotFound"):
            st.info("No saved monitoring report exists for this Raw run.")
        else:
            st.warning("MinIO rejected the monitoring report request.")

    except ValueError:
        st.warning("The saved monitoring report could not be decoded.")

    else:
        st.write("Monitored at (UTC):", drift_report["monitored_at"])
        st.write("Reference Metadata run ID :", drift_report["reference_metadata"]["run_id"])

        # NEW: Explain the sample scope and the rules stored with this report.
        st.caption(
            "Compares the selected run's Gold test features with the model's training features, "
            "using cycles 20 onward and the saved model inputs. Training histories continue "
            "through failure; test histories stop earlier, so differences can reflect engine life stages."
        )
        thresholds = drift_report["thresholds"]
        st.caption(
            f"Mean-shift flags: absolute mean change / reference standard deviation "
            f">= {thresholds['mean_shift_threshold']}. Spread flags: current/reference "
            f"standard deviation < {thresholds['lower_ratio']} or > {thresholds['upper_ratio']}. "
            "These are initial demonstration thresholds."
        )
        # END NEW: Drift comparison scope and thresholds.

        feature_count, mean_shift_flag_count, spread_change_flag_count = st.columns(3)
        feature_count.metric("Number of features", drift_report["summary"]["feature_count"])
        mean_shift_flag_count.metric("Mean shift flag count", drift_report["summary"]["mean_shift_flag_count"])
        spread_change_flag_count.metric("Spread change flag count", drift_report["summary"]["spread_change_flag_count"])

        drift_df = pd.DataFrame(drift_report["feature_comparison"])
        st.dataframe(
            drift_df[["feature","mean_shift_in_reference_std","std_ratio","mean_shift_flag","spread_change_flag"]],
            hide_index=True,
            width="stretch",
        )

        # NEW: Preserve the distinction between unavailable assessments and passed rules.
        st.caption(
            "Blank scores or flags mean the assessment is unavailable. Flags prompt investigation; "
            "they do not measure prediction accuracy, and unflagged features may still differ in other ways."
        )

    dataset = st.selectbox("Select a dataset to inspect", ["train_data", "test_data", "RUL_data"])

    if dataset:
        st.write("Selected dataset:", dataset)
        filtered_metadata = selected_raw_metadata[selected_raw_metadata["dataset_name"] == dataset]
        if len(filtered_metadata)!= 1:
            st.error(f"Expected one record for {dataset} in this Raw run, found {len(filtered_metadata)}.")
            st.stop()
        object_key = filtered_metadata["object_key"].values[0]
        # Handle connection failures and error responses for the selected dataset.
        try:
            dataset_minio = read_parquet_object(s3_client, BUCKET_NAME, object_key)
        except EndpointConnectionError:
            st.error("Cannot connect to MinIO to read the selected dataset. Check that minio-cmapss is running and the configured endpoint is correct.")
            st.stop()
        except ClientError:
            st.error("MinIO rejected the selected dataset request. Check that its recorded object exists and your credentials have read access.")
            st.stop()

        # Check columns and numeric types before running row-level checks.
        if dataset in ["train_data", "test_data"]:
            expected_columns = (
                ["unit_number", "time_in_cycles"]
                + [f"operational_setting_{i}" for i in range(1, 4)]
                + [f"sensor_measurement_{i}" for i in range(1, 22)]
            )
        else:
            expected_columns = ["unit_number", "remaining_useful_life"]

        missing_columns = set(expected_columns) - set(dataset_minio.columns)
        unexpected_columns = set(dataset_minio.columns) - set(expected_columns)

        if missing_columns or unexpected_columns:
            st.error(
                f"Unexpected schema for {dataset}. "
                f"Missing columns: {sorted(missing_columns)}. "
                f"Unexpected columns: {sorted(unexpected_columns)}."
            )
            st.stop()

        numeric_columns = dataset_minio.select_dtypes(include="number").columns
        non_numeric_columns = set(expected_columns) - set(numeric_columns)

        if non_numeric_columns:
            st.error(f"Expected numeric columns, but found non-numeric types in: {sorted(non_numeric_columns)}.")
            st.stop()

        if (dataset == 'train_data') or (dataset == 'test_data'):
            invalid_rows = (dataset_minio["unit_number"].isna()) | (dataset_minio["unit_number"]<=0) | (dataset_minio["unit_number"] % 1 != 0) |(dataset_minio["time_in_cycles"].isna()) | (dataset_minio["time_in_cycles"]<=0) | (dataset_minio["time_in_cycles"] % 1 != 0)
        else : 
            invalid_rows = (dataset_minio["unit_number"].isna()) | (dataset_minio["unit_number"]<=0) | (dataset_minio["unit_number"] % 1 != 0) | (dataset_minio["remaining_useful_life"].isna()) | (dataset_minio["remaining_useful_life"]<0) 
        
        finite_mask = np.isfinite(dataset_minio.select_dtypes(include=[np.number])).all(axis=1)
        # Non-finite includes NaN and positive/negative infinity.
        non_finite_rows = dataset_minio[~finite_mask]
        summary = {
            "Actual row count": len(dataset_minio),
            "Missing cells" : dataset_minio.isna().sum().sum(),
            "missing rows" : dataset_minio.isna().any(axis=1).sum(),
            "Duplicate rows" : dataset_minio.duplicated().sum(),
            "Rows failing ID/cycle/RUL checks" : invalid_rows.sum(),
            "Rows with non-finite numeric values" : len(non_finite_rows),
        }

        st.write("Data quality summary for this dataset:")
        st.dataframe(
            pd.DataFrame([summary]),
            hide_index=False,
            width="stretch",
        )

        if len(dataset_minio) != filtered_metadata["row_count"].values[0]:
            st.warning(
                f"Row count mismatch: metadata says {filtered_metadata['row_count'].values[0]}, "
                f"but the dataset has {len(dataset_minio)} rows."
            )
        else:
            st.success("Row count matches metadata.")


        # Preview rows failing either set of checks, counting each row once.
        failed_checks_mask = invalid_rows | ~finite_mask
        failed_check_rows = dataset_minio[failed_checks_mask]

        st.subheader("Examples of rows failing ID/cycle/RUL or numeric-finiteness checks")
        if failed_check_rows.empty:
            st.info("No rows failed the ID/cycle/RUL or numeric-finiteness checks.")
        else:
            st.write("Rows failing these checks:", len(failed_check_rows))
            st.dataframe(failed_check_rows.head(10), hide_index=False, width="stretch")
            st.caption(
                "Showing up to ten affected rows from the selected stored Raw dataset. "
                "Rows failing multiple rules appear only once. Duplicate-row checks are separate."
            )
        
    st.caption("This page inspects stored Raw data from the selected ingestion run. It does not monitor a live sensor stream")
