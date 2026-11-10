# Industrial Predictive Maintenance MLOps Platform

An educational MLOps platform that predicts engine Remaining Useful Life
(RUL) using NASA C-MAPSS FD001 sensor histories.

The platform includes pandas data processing, MinIO storage, a registered
Random Forest model, FastAPI predictions, SHAP explanations, a Streamlit
dashboard, Airflow orchestration, monitoring and GitHub Actions checks.

## Local deployment

The root `docker-compose.yaml` coordinates:

- PostgreSQL for Airflow metadata.
- MinIO for datasets, manifests and monitoring reports.
- Airflow initialization, API server, scheduler and DAG processor.
- MLflow for saved experiments and registered models.
- FastAPI for predictions.
- Streamlit for the dashboard.

This deployment reuses prepared project data and artifacts.
A fresh Git clone alone does not contain everything needed to run it.

## Required local resources

Before startup, supply:

- Docker Desktop running Linux containers.
- A `.env` file with valid settings; see `.env.example`.
- FD001 files in `CMAPSSData`: `train_FD001.txt`, `test_FD001.txt`
  and `RUL_FD001.txt`.
- The exported model bundle in `output/serving_model`, including
  the `model` folder and both metadata JSON files.
- The prepared Docker MLflow database at `output/docker_mlflow/mlflow.db`.
- Historical artifact files under the existing `mlruns` directories.
- Existing Docker volumes `minio_cmapss_data` and
  `cmapss-airflow_postgres-data`.

The MinIO volume must contain the project bucket and historical training
Gold run referenced by the serving model's metadata. Creating an empty
volume does not restore this data.

Data files, credentials, databases and model artifacts are excluded from Git.

## MLflow database preparation

The preparation scripts are specific to the original Windows project path.
Review their paths before using them elsewhere.

For initial preparation only:

```powershell
python mlflow/prepare_database.py
python mlflow/update_artifact_paths.py
```

The first script copies the original database and refuses to overwrite an
existing Docker copy. The second changes artifact addresses in that copy
to use `/opt/cmapss`.

The original database is preserved. Subsequent changes to it are not
automatically synchronized to the Docker copy.

## Start and stop

Run commands from the project root.

Validate configuration:

```powershell
docker compose config --quiet
```

Build images and start services:

```powershell
docker compose up -d --build
```

Start again using existing images:

```powershell
docker compose up -d --no-build
```

Inspect status and logs:

```powershell
docker compose ps -a
docker compose logs --tail 50
```

Stop services:

```powershell
docker compose stop
```

`airflow-init` finishing with exit code 0 is expected.
Startup does not automatically trigger the prediction DAG.

Keep the previous standalone deployment stopped when using this stack:
it shares ports and persistent storage with this deployment.

## Browser addresses

| Interface | Address |
|---|---|
| Streamlit | http://localhost:8502 |
| FastAPI documentation | http://localhost:8001/docs |
| MLflow | http://localhost:5001 |
| Airflow | http://localhost:8081 |
| MinIO console | http://localhost:9011 |

Airflow uses the configured `badr` account. Its generated password file is
under `airflow/config`; credentials must not be committed.

## Run the pipeline

Open Airflow and trigger a new manual run of `cmapss_data_prediction`.

The ten-task workflow ingests and validates data, generates features,
checks drift, requests predictions, records model monitoring and publishes
a dashboard snapshot.

After success, open Streamlit and select Overview → Latest Airflow run.
Refresh the page to retrieve the latest published snapshot.

The workflow does not retrain or promote models. Engine Explorer remains
an interactive three-engine demonstration.

## Scope

This is a local educational system using simulated benchmark data.
Monitoring covers the documented project scope; it is not a certified
aircraft-maintenance system.

The detailed implementation history and decisions are recorded in
`docs/project_complete_handoff_through_phase_8.md`.