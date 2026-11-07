# Phase 8: FastAPI service and Docker guide

Verified on 2026-10-18. The API runs locally inside Docker Desktop; it has not been deployed to the cloud.

## What is built

- `GET /health`: checks that the app has a loaded model.
- `GET /model-info`: returns the model name, version, training run ID, and Gold dataset run ID.
- `POST /predict`: predicts the latest cycle for one engine.
- `POST /predict/batch`: predicts the latest cycle for each submitted engine.
- Responses include RUL, healthy/warning/critical status, model version, and three leading SHAP factors.
- Pydantic validation, feature checks, JSON logging, request IDs, error handling, and 28 automated tests are implemented.

Requests contain raw observations for one engine from cycle 1 through the current cycle, in order without gaps, with at least 20 observations. Full history preserves the EMA calculation used in training. The model receives the saved list of 137 features in its original order. This supersedes the earlier handoff's suggestion that only the latest 20 cycles are sufficient.

Status thresholds: RUL <= 10 is critical; 10 < RUL <= 30 is warning; RUL > 30 is healthy.

## How loading works

`src/modeling/export_model.py` finds registered model `cmapss_fd001_rul_random_forest`, version 1, through the root `mlflow.db`. It copies the saved model and feature metadata into `output/serving_model` and writes `model_info.json`. It does not retrain the model. It refuses to overwrite an existing export folder.

`src/api/model_loader.py` contains two loaders. Without `MODEL_DIRECTORY`, startup uses the original registry loader. When `MODEL_DIRECTORY` is supplied, startup uses the exported-folder loader. Both return the same dictionary structure, including the model and SHAP explainer. A failed exported-folder load stops startup; it does not fall back to the registry.

Docker supplies `MODEL_DIRECTORY=/model`. A read-only bind mount makes the Windows export folder accessible at `/model`. The model is not copied into the image.

## Build the image

Run from `C:\Users\Badr\Desktop\Python_3.0`:

```powershell
docker build -f api/Dockerfile -t maintenance-api .
```

The final dot chooses the project root as the build context. The root `.dockerignore` filters that input. The Dockerfile installs packages from `api/requirements.txt`, copies `src` into `/app/src`, and sets the default Uvicorn startup command. Building does not start the API.

## Create and start the container

Use this only when creating the container for the first time. If `maintenance-api` already exists, use `docker start maintenance-api` instead.

```powershell
docker run -d --name maintenance-api -p 127.0.0.1:8001:8000 --mount "type=bind,source=C:\Users\Badr\Desktop\Python_3.0\output\serving_model,target=/model,readonly" maintenance-api:latest
```

- `-d` runs the container in the background.
- `--name` assigns its name.
- Port 8001 on the laptop forwards to Uvicorn's port 8000 in the container.
- `--mount` connects the existing Windows folder to `/model`; it does not copy it.
- `readonly` prevents the container from modifying files through this mount.
- The final argument selects the image.

Open http://127.0.0.1:8001/docs or http://127.0.0.1:8001/health after startup finishes.

## Everyday commands

```powershell
docker ps
docker logs --tail 30 maintenance-api
docker stop maintenance-api
docker start maintenance-api
```

`docker start` preserves the existing container's mount and port settings. Stopping leaves the image, container, and host model files in place. A stopped server produces a browser connection failure, not an HTTP 503 response. A running app returns 503 if its model state is unavailable; a model-loading failure during startup prevents normal service startup.

Rebuilding an image does not update an existing container. Applying changed source code requires building the image and creating a replacement container from it.

## Verification completed

- All 28 automated tests passed before the image build, with four dependency deprecation warnings.
- Image built successfully with Python 3.14 and the pinned API/model packages.
- Container loaded exported version 1 without needing the Windows MLflow database.
- Health, model information, and interactive documentation returned HTTP 200.
- Single and batch predictions matched the earlier local results: engine 1 approximately 182.74044, engine 2 approximately 140.56869, engine 3 approximately 46.02538 cycles.
- Real predictions included SHAP factors.
- Missing sensor, duplicate cycle, and empty batch requests returned HTTP 422 with request IDs. The service remained healthy afterward.

For manual prediction requests, use the JSON examples in `docs/examples/predict_engine_1.json` and `docs/examples/predict_batch_engines_1_2_3.json` through the interactive documentation.

To run automated tests from the project root with the virtual environment active:

```powershell
python -m pytest tests -v
```

The tests use TestClient and controlled substitutes; they do not require the Docker API to be running.
