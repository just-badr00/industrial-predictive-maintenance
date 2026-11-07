# Phase 10 — Airflow data and prediction workflow

Completed 2026-10-30 (UTC). One local manual DAG, `cmapss_data_prediction`:

```text
check_source_data -> ingest_raw_data -> validate_raw_data
  -> transform_to_silver -> generate_features -> validate_features
  -> predict_current_health -> publish_dashboard_dataset
```

The separate training DAG was explicitly skipped by Badr. Section 25 of `project_complete_handoff_through_phase_8.md` records the Phase 10 implementation. Section 26 describes the current ten-task workflow and its Phase 11 monitoring reports. The startup and manual run commands below also apply to the current DAG.

## Routine operation

Run commands from `C:\Users\Badr\Desktop\Python_3.0`.

Check existing services first:

```powershell
docker compose -f airflow/docker-compose.yaml ps -a
docker ps -a
```

If the existing project MinIO or prediction API is stopped, start that container:

```powershell
docker start minio-cmapss
docker start maintenance-api
```

Start the Airflow services if needed:

```powershell
docker compose -f airflow/docker-compose.yaml up -d airflow-apiserver airflow-scheduler airflow-dag-processor
```

PostgreSQL and initialization are dependencies; `airflow-init` ending with code 0 is expected. An image rebuild is only needed when image/dependencies change, not for mounted Python edits. The existing model bundle remains mounted into maintenance-api; do not rerun its exporter just to start Airflow.

Open `http://localhost:8081`, select `cmapss_data_prediction`, unpause if needed, and trigger a **new manual run**. Credentials: Simple Auth Manager user `badr`; generated password is in `airflow/config/simple_auth_passwords.json`. Passwords and MinIO credentials are deliberately omitted here.

View each task's Logs in the UI. Prediction prints `Predicted 10/100 test engines.` through `100/100`, after each batch completes. Initialization does not run the data DAG. There is no time schedule.

Start the dashboard using the existing environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/streamlit_app.py
```

Open `http://localhost:8501`, choose Overview or Maintenance Queue, then `Latest Airflow run`. Predictions are saved snapshots; rerun/reload the page to read the latest published snapshot. Selecting this source does not call FastAPI or trigger Airflow. Engine Explorer retains three demo histories.

## Saved outputs

The manifest is `metadata/etl_metadata/load_date=<UTC date>/run_id=<timestamp ID>/manifest.parquet`.

| Layer | Dataset names | Typical FD001 shape |
|---|---|---|
| Raw | train_data / test_data / RUL_data | 20,631x26 / 13,096x26 / 100x2 |
| Silver | train_data / test_data / RUL_data | 20,631x28 / 13,096x28 / 100x4 |
| Gold | train_features / test_features | 20,631x195 / 13,096x194 |
| Gold | maintenance_predictions | 100x11 |
| Gold | current_engine_health | 100x12 |

A complete run has ten manifest records. Feature tables retain cycles 1–19; rolling-window NaNs there are expected. Eligible model inputs start at cycle 20 and select the saved 137 features in training order. EMA is calculated over complete engine history.

The latest audited snapshot on 2026-10-30 was run `20261027T234511255393Z`: 100 engines, 82 healthy, 16 warning, 2 critical; average predicted RUL displayed as 87.8, model version 1. These are saved results, not current live measurements.

## Addresses and storage

- Windows MinIO/API: `localhost:9010` / `127.0.0.1:8001`.
- Airflow MinIO/API: `host.docker.internal:9010` / `host.docker.internal:8001`.
- Airflow internal execution API: `airflow-apiserver:8080/execution/`.
- `env_file: ../.env` injects project settings; Compose overrides the container MinIO endpoint.
- `..:/opt/cmapss` exposes the project; `PYTHONPATH=/opt/cmapss` enables `src` imports.
- PostgreSQL `postgres-data` is Airflow metadata storage; MLflow still uses root SQLite and datasets stay in MinIO.
- SANLAM's old Airflow stack was stopped and retained; it is not this project's deployment.

## Checks and limitations

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_airflow_data_stages.py -q
docker compose -f airflow/docker-compose.yaml config --quiet
docker compose -f airflow/docker-compose.yaml logs --tail 60 airflow-scheduler airflow-dag-processor airflow-apiserver
```

There are 45 offline pipeline tests. Final full project suite on 2026-10-30: **73 passed, six dependency deprecation warnings**, using `.venv` Python and `pytest tests -q`. Storage/API failures are simulated in the pipeline tests; actual DAG retry configuration was inspected, without deliberately stopping live dependencies. Local sandbox sockets stalled the first full-suite attempt; the permitted rerun passed. Distinguish that restriction from a project failure.

All edges require upstream success. Storage/network tasks get one retry after 30 seconds; source/validation tasks have zero retries. Retrying the same task uses the same run partition. New manual runs create new partitions even for identical source files. Completed-DAG clear/reset semantics may change `dag_run.start_date`; use new manual runs for fresh executions.

Manifest writes are not transactions across several objects. Partial uploads may remain after failure; successful publication is recorded only after the stage's uploads. Out-of-band concurrent writers, source-file mutation during a run, retention/cleanup, caching, production authentication and full monitoring remain outside this local phase. Broad dependency versions are not a reproducible lockfile. Preserve named volumes and historical MLflow/data artifacts.
