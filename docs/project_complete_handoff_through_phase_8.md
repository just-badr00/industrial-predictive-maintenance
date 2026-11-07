# Industrial Predictive Maintenance MLOps Platform — Complete Handoff Through Phase 11

**Updated:** 2026-11-06 (UTC)  
**Project root:** `C:\Users\Badr\Desktop\Python_3.0`  
**Dataset:** NASA C-MAPSS FD001  
**Status:** Phases 0–11 cover the agreed local scope. The ten-task Airflow DAG publishes fleet predictions and saves Raw quality, feature drift, and model monitoring reports. Streamlit displays the saved reports. **Badr explicitly removed the separate training DAG from Phase 10 on 2026-10-30.** Phase 12 covers broader testing and CI/CD.  
**Audience:** Badr and the assistant continuing the project in another conversation.

**Filename retained deliberately:** `project_complete_handoff_through_phase_8.md` now covers Phase 11. Read Sections 1–4, 21 and **26** for the present state. Sections 22 and 25 record the Phase 9 and 10 implementations. Section 26 supersedes their older workflow and monitoring descriptions. The short Airflow operational reference is `docs/phase10_airflow_guide.md`.

## How to use this document

This is the single combined handoff. It contains:

1. The authoritative current state and corrections to older assumptions.
2. A complete Phase 8 record: architecture, files/functions, contracts, learning decisions, tests, Docker, commands, and verification.
3. The Phase 9–11 implementation records, verification, the training DAG scope decision, and the Phase 12 continuation prompt.
4. The full earlier Phase 0–4 and Phase 5–7 handoff texts, preserved as historical appendices in this same file.

**Read the current-state sections first.** The historical appendices intentionally preserve their original wording and dates. Their statements such as “Phase 5 is next,” “Phase 8 is next,” and “latest 20 cycles” describe earlier stages; they are superseded here. Section 20's teaching sequence is the original proposal before Phase 9, not pending implementation. There is no need to repeat completed model training, API implementation, Docker serving setup, or dashboard construction.

Sources: the earlier Phase 8 record; the complete Phase 9 conversation through documentation on 2026-10-21; the current source files (especially `dashboard/streamlit_app.py`); saved notebook outputs and MLflow records; `docs/project_progress_summary_detailed.md`; `docs/project_handoff_phases_5_to_7.md`; `docs/phase8_service_guide.md`; and the original roadmap conversation, read directly through the chat tool:

`thread://01a039e9-3038-7243-8daa-1d60821395be?hostId=local` — **Recall machine learning ideas chat**.

## 1. Collaboration and teaching preferences — important for the next assistant

Badr is learning the concepts while building the project. He wants to work together, one step at a time, rather than receive an entire implementation at once.

- Explain the purpose and logic before, or alongside, code. Explaining only what each line does is insufficient.
- Use simple language and concrete examples. Introduce unfamiliar terms explicitly.
- For genuinely new concepts, code can be supplied directly and explained block by block.
- For familiar Python patterns, help Badr work them out himself unless he asks for the direct solution.
- Keep the same variable names and structures used in the notebooks and recently written code. Do not introduce alternative APIs, clever path discovery, new abstractions, or different coding styles without a reason.
- In particular, keep the familiar explicit `project_root`, `mlflow_database`, SQLite URI, `MlflowClient`, `source_run_id`, and artifact-loading pattern.
- Do not rewrite notebooks unless explicitly requested. Preserve previous working analyses and results.
- Add short, simple explanatory comments near important code when requested. Existing schema and lifespan comments are deliberate teaching aids.
- Teach the user to run and understand commands; do not assume that successfully executing them means the user understood them.
- Agree on a small roadmap and a finish line at the start of a phase. Group familiar changes into useful blocks and keep routine reviews fast. Explain unfamiliar logic with a small example before moving on.
- At the beginning of each step, briefly explain what we are going to do and why. Then explain the relevant new concept or give Badr a manageable implementation task.
- Do not implement familiar Python/pandas logic on Badr's behalf by default. Give the purpose, hints, and placement; let him write it. He explicitly objected when the assistant did too much automatically.
- Read the saved file when he asks “what do you think?” Do not assume the last suggested code was implemented. Review without editing unless he asks for edits.
- When his work is correct, briefly confirm and automatically introduce the next small task. Do not wait for another “let's continue” merely to choose that task.
- An explicit “do this by yourself” or “add them yourself” authorizes the requested step, not all future implementation. Complete that step and explain it; then return to the collaborative teaching pattern.
- For direct code edits, add clear `# NEW:` comments and, where useful, `# END NEW:` around the latest changes. Remove or convert old NEW markers so only the newest changes are highlighted. Preserve ordinary explanatory comments.
- A file once reverted to its earlier on-disk content after an assistant edit. The cause was not established. On disagreement, reread disk instead of insisting it is merely a browser/editor refresh problem. Preserve user edits, reapply only authorized work, and verify the save. Warn about overwriting newer disk changes with an older unsaved editor buffer only when relevant.
- Phase 10 teaching update: Badr authorized extracting notebook logic into reusable modules while he wrote DAG wrappers/dependencies. He later requested a slower pace for unfamiliar inference and dashboard code. Explain each function block by block, including inputs, outputs, syntax and purpose; a high-level summary alone is insufficient for unfamiliar code. Broad file-reading restrictions were lifted. Preserve his DAG/dashboard edits.
- Training automation was deliberately omitted on 2026-10-30: repeated ingestion of the same FD001 files adds storage versions, not new training information. Do not restart the training-DAG discussion unless Badr requests it.

## 2. Current project objective and phase status

Predict an engine's Remaining Useful Life (RUL), measured in operating cycles, from its operational settings and sensor history. Derive a health status and explain important prediction factors.

| Phase | Current state |
|---|---|
| 0 — Contract | Complete: objective, target, grain, metrics, architecture, limitations |
| 1 — Ingestion | Complete: FD001 parsing, validation, data dictionary, MinIO Raw |
| 2 — Data lake | Complete: run-scoped Raw/Silver/Gold and ETL manifest lineage |
| 3 — EDA | Complete: sensor trends, distributions, correlation, RUL interpretation |
| 4 — Features | Complete: grouped, causal time-series features and Gold feature tables |
| 5 — Baselines | Complete: Mean RUL, Linear Regression, Random Forest, XGBoost |
| 6 — Final model / MLflow | Complete: engine folds, Optuna cost selection, final training, evaluation, registry |
| 7 — Explainability | Complete: native/permutation importance, SHAP, local cases, residuals, MLflow logging |
| 8 — FastAPI | Complete: endpoints, validation, reusable inference, SHAP, logging, errors, tests, Docker |
| 9 — Streamlit | Complete; five original views plus Phase 10 source selection for Airflow fleet snapshots on Overview/Queue; Explorer remains three-engine demo |
| 10 — Airflow workflows | Complete. Section 25 records the original eight-task workflow. Section 26 gives the current ten-task DAG. The training DAG was omitted by user decision. |
| 11 — Monitoring | Complete for the local scope. Saved Raw quality, feature drift, prediction statistics, labeled accuracy, batch request timing and an MAE alert. See Section 26 for deferred checks. |
| 12 — Wider testing / CI/CD | Next. Existing API, pipeline, and monitoring tests provide a starting point for broader coverage and CI. |
| 13 — Integrated Docker deployment | Planned; API Docker support is done, whole-platform Compose integration remains |

Original plans mentioned PySpark and PostgreSQL among possible stack components. Processing uses pandas, not PySpark. PostgreSQL is now implemented for **Airflow metadata only**; MLflow still uses root SQLite and ETL manifests still live in MinIO. Do not claim whole-platform Compose integration or automated training.

## 3. Identities and resources to preserve

| Item | Value |
|---|---|
| Authoritative MLflow database | `C:\Users\Badr\Desktop\Python_3.0\mlflow.db` |
| Other database | `notebooks/mlflow.db` exists; do not accidentally switch to it |
| Registered model | `cmapss_fd001_rul_random_forest` |
| Selected model version | `1` |
| Registry alias | `candidate` |
| Final training/source run | `360c5414c7d841a38bcf9480521a7b59` |
| Gold pipeline run | `20260902T120041719003Z` |
| Optuna parent run | `b692bdfb2e5e43d7abcc1e3d243a4241` |
| Best tuning child | `9ec1ce8c389c40a5877e8c36a1778ea8` |
| Comparison reference run | `08f9cb42ee224fcf8298ce2d82589c5f` |
| Tuning experiment | `cmapss_phase_6_random_forest_operational_cost_folds` |
| Final training features | 137 named, ordered inputs |
| Model | `RandomForestRegressor`, 153 trees, `min_samples_leaf=2`, `max_features=None`, `random_state=42` |
| Saved serialization | `model.skops` inside the MLflow model artifact folder |
| MinIO container | `minio-cmapss` |
| MinIO persistent volume | `minio_cmapss_data` |
| MinIO API / console | `http://localhost:9010` / `http://localhost:9011` |
| API image | `maintenance-api:latest` |
| API container | `maintenance-api` |
| Container API / docs | `http://127.0.0.1:8001` / `http://127.0.0.1:8001/docs` |

Original model files are located under:

```text
notebooks/mlruns/0/models/m-ab9d4c450c5d437c8e6a2faac2280200/artifacts/
```

Original feature metadata is located under:

```text
notebooks/mlruns/0/360c5414c7d841a38bcf9480521a7b59/artifacts/metadata/final_training_features.json
```

Keep both root `mlruns` and `notebooks/mlruns`. The registry and experiments contain historical artifact-location choices. Do not delete or migrate them casually. Credentials remain in `.env`; no credential values belong in the handoff.

The authoritative root database and its registered model were confirmed. The discussion about which database was “original” should not be resolved by assuming that notebook working directories determine the intended current database. Use the verified root database.

## 4. Corrections to the older handoffs

### 4.1 Full history is required for exact EMA, not just the latest 20 rows

Rolling mean/std/min/max and slope use a trailing 20-cycle window with `min_periods=20`. EMA uses:

```python
column.ewm(span=20, adjust=False, min_periods=1).mean()
```

`span=20` determines the exponential weighting; it does not discard everything before the latest 20 cycles. The recursive EMA retains influence from earlier history.

`min_periods=1` controls how early an EMA value may be emitted; it is not the reason its memory extends beyond 20 cycles. Removing early training rows after computing features does not reset later EMA values.

The adopted API contract therefore requires **all observed cycles from cycle 1 to the requested current cycle**, with at least 20 observations. Cycles must start at 1, be ordered, and be consecutive. Do not change this to accepting a trailing 20-row segment unless a new stateful EMA strategy is explicitly designed and verified.

The valid first complete 20-row window ends at cycle 20; cycles 1–19 lack a complete window. Feature calculation happens before selecting eligible/latest rows.

### 4.2 Feature selection is fixed by training metadata

Training removed constant columns using training data. Inference creates the broader set of features and selects exactly the saved 137 columns:

```python
model_input = latest_row.loc[:, feature_columns].copy()
```

This selects columns by name and places them in training order. Extra engineered columns are expected. Comparing the full generated-column count/order directly with the model feature list would incorrectly fail.

Do not independently discover constants on an incoming engine history. A feature temporarily constant in one request may still be a trained model input.

### 4.3 Earlier Notebook 05 metadata-filter TODO is resolved in source

The handoff author inspected Notebook 05's saved code on 2026-10-18. Its filter now replaces only the current run's Gold records whose dataset names are `train_features` or `test_features`. It preserves unrelated Gold records, including Notebook 03's output. The earlier “apply this filter if pending” instruction is no longer a code-edit task. This inspection did not rerun the entire notebook or audit all MinIO objects again.

## 5. Phase 8 learning path and concepts established

The phase was implemented interactively in this order: requests/responses; first health endpoint; interactive documentation; model loading; model info; request schemas; history checks; DataFrame conversion; shared feature engineering; single prediction; error handling; batch prediction; SHAP factors; structured logging/middleware; automated tests; Docker export/loading/build/run.

### Client, server, request, and response

- A client sends an HTTP request. A browser, Swagger documentation page, Python program, or future Streamlit app can be a client.
- `POST http://127.0.0.1:8001/predict` describes a request's method and address; it is not itself the client.
- The request body contains engine history. The response body contains the prediction.
- HTTP 200 indicates a successful response. A health field such as `warning` describes engine condition and is separate from HTTP status.
- `/docs` provides interactive Swagger UI that can send actual HTTP requests. It is an API client, not the ML model.

### FastAPI and Uvicorn

FastAPI defines routes, validation, and application behavior. Uvicorn is the server process listening for network connections and running the FastAPI application.

Memory sentence: **Uvicorn receives the HTTP traffic; FastAPI decides how the application handles it.**

The route decorator determines the URL. Renaming a Python function alone does not change a route such as `/health`; changing the decorator path does.

### Startup, lifespan, yield, and shutdown

`@asynccontextmanager` turns the async generator into the context manager used by FastAPI's lifespan mechanism. Code before `yield` runs during startup. `yield` gives control back while the service handles requests. Code after it runs during normal shutdown.

Startup loads a model once per application process and stores the resulting dictionary on `app.state.model_data`. It does not reload the model for every prediction. Development `--reload` can restart the app and run startup again. Shutdown clears the app's reference; saved model files remain on disk.

### Pydantic machinery

Type annotations describe the fields. Inheriting `BaseModel` provides validation and methods such as `model_validate()` and `model_dump()`. `Field(...)` supplies constraints; Pydantic enforces them. Python annotations alone do not validate arbitrary input.

`PredictionRequest.model_validate(data)` is inherited from `BaseModel`; it need not be defined in our subclass. A successful validation returns a `PredictionRequest` object. Its `history` is a list of `CycleObservation` objects, not a list of dictionaries anymore. `observation.model_dump()` converts an observation back to a dictionary for DataFrame construction.

`@field_validator("history", mode="after")` runs after nested field validation. `@classmethod` makes Python supply the class as `cls`; Pydantic supplies the history value. The validator returns valid history or raises `ValueError`.

### Imports and project-root execution

Use `python -m src.api.practice_schema` from the project root for the practice module. Directly executing a nested file changes Python's import-search starting point and can produce `ModuleNotFoundError: No module named 'src'`. A local `from schemas import ...` might work in one invocation context but does not preserve the project's consistent `src...` imports.

## 6. Phase 8 architecture and file map

```text
Browser / Swagger / future Streamlit client
                |
                | HTTP JSON
                v
Uvicorn -> FastAPI main.py -> request logging middleware
                                |
                                v
                    Pydantic request validation
                                |
                                v
                  shared feature_engineering.py
                                |
                     latest row, 137 columns
                                |
                                v
                 loaded Random Forest + SHAP explainer
                                |
                       status + top factors
                                |
                                v
                     validated JSON response

Startup -> choose registry or exported-folder loader
        -> store shared model_data on app.state
```

| File | Responsibility |
|---|---|
| `src/api/main.py` | App, lifespan, loader selection, middleware, guards, four routes |
| `src/api/schemas.py` | Cycle/request/batch/response/factor validation models |
| `src/api/model_loader.py` | Registry loader and exported-folder loader |
| `src/api/practice_schema.py` | Learning/practice script; not an endpoint |
| `src/transformations/feature_engineering.py` | Raw history -> exact latest model input |
| `src/modeling/predict.py` | `health_status()` thresholds |
| `src/modeling/explain.py` | `explain_predictions()` for one or many rows |
| `src/modeling/export_model.py` | Copy registered model and metadata into a serving bundle |
| `src/logging_config.py` | JSON formatter and named application logger |
| `docs/examples/predict_engine_1.json` | Real, valid engine 1 history through cycle 31 |
| `docs/examples/predict_batch_engines_1_2_3.json` | Valid raw histories for three engines |
| `tests/test_api.py` | Endpoint success and selected validation/unavailability tests |
| `tests/test_service_validation.py` | Additional validation, errors, model info, request logging |
| `tests/test_explanations.py` | Factor ranking, values/signs, row alignment |
| `tests/test_predict.py` | Health threshold boundaries |
| `api/requirements.txt` | Pinned serving dependencies |
| `api/Dockerfile` | API image recipe |
| `.dockerignore` | Exclusions from project-root build context |
| `output/serving_model/` | Exported bundle mounted into the container |
| `docs/phase8_service_guide.md` | Shorter operational guide |

## 7. Request and response contracts

### CycleObservation

Every cycle requires `time_in_cycles`, `operational_setting_1` through `_3`, and `sensor_measurement_1` through `_21`.

- `time_in_cycles`: integer > 0.
- Settings and sensors: numeric float fields; finite values required.
- `ConfigDict(allow_inf_nan=False, strict=True, extra="forbid")` rejects non-finite numbers, unwanted type coercion, and unexpected fields. Ordinary integer numeric inputs can be accepted for float fields; do not describe strict float validation as requiring a decimal point in JSON.
- All fields are required; there are no implicit sensor defaults.

### PredictionRequest

```text
engine_id: positive integer
history: list[CycleObservation], minimum length 20
```

It also has strict validation and forbids extras. `cycle_check()` collects cycle numbers and requires `cycles[i] == i + 1`, ensuring history starts at 1 with no gaps, duplicates, or reordering.

### BatchPredictionRequest

```json
{
  "engines": [
    {"engine_id": 1, "history": ["full observation objects here"]},
    {"engine_id": 2, "history": ["full observation objects here"]}
  ]
}
```

The strings above are explanatory placeholders, not valid observations. Use the complete JSON example file for an executable request. `engines` requires at least one `PredictionRequest`; each engine can have a different history length.

### PredictionResponse

```json
{
  "engine_id": 1,
  "cycle": 31,
  "predicted_rul": 182.74043987965555,
  "status": "healthy",
  "top_factors": [
    {"feature": "time_in_cycles", "value": 31.0, "shap_value": 31.1830253051468},
    {"feature": "sensor_measurement_3_ema_20", "value": 1585.931380295147, "shap_value": 11.241872789525857},
    {"feature": "sensor_measurement_17_ema_20", "value": 391.91525787460785, "shap_value": 9.20650874952135}
  ],
  "model_version": "1"
}
```

Response fields validate positive engine/cycle IDs, finite nonnegative RUL, status in `healthy|warning|critical`, factor feature/value/signed SHAP contribution, and string model version. Batch returns a JSON list of these responses in submitted order.

The real factors are structured objects, not the human-written risk phrases from the initial roadmap example. Do not present engineered names as automatically generated maintenance diagnoses.

## 8. Shared feature-preparation functions

`history_to_dataframe(request)` loops over `request.history`, applies `model_dump()` to each observation, constructs a DataFrame, and adds `unit_number = request.engine_id`.

`prepare_feature_input(dataframe)` copies data, prefixes raw sensor column names with `current_`, sorts by engine/cycle, and resets the index. Requests do not contain the Silver-only metadata columns from the notebook.

`add_time_series_features(dataframe, sensors)` reuses Notebook 05's logic and naming:

- Group by `unit_number` so engines never mix.
- Rolling mean/std/min/max: window 20, minimum periods 20.
- One-cycle difference.
- Percentage change with `fill_method=None`; replace +/- infinity with NaN rather than inventing zero changes.
- Rolling slope: `np.polyfit(np.arange(len(x)), x, 1)[0]`, window 20.
- Recursive EMA: span 20, adjust False, minimum periods 1, across complete history.

`prepare_model_input(request, feature_columns)` performs those steps, takes `tail(1)`, checks required features exist, selects `latest_row.loc[:, feature_columns].copy()`, and rejects missing/non-finite selected values.

- Missing required feature columns raise ordinary `ValueError`: a code/contract problem, not an ordinary invalid-history response.
- Undefined numeric features raise `InvalidEngineHistoryError`, a subclass of `ValueError`; the routes convert that specific error into HTTP 422.
- A zero previous sensor value can cause undefined percentage change even when the JSON is structurally valid.

Feature equivalence with Notebook 05 was checked on real histories from engines 1 and 2 at multiple cycles. Values matched; some raw numeric dtypes differed without changing values. No new model was trained for Phase 8.

## 9. Model loading, startup, and model state

### Registry loader: load_registered_model()

Keeps the familiar explicit project root and database setup:

```python
project_root = Path(r"C:\Users\Badr\Desktop\Python_3.0")
mlflow_database = (project_root / "mlflow.db").resolve().as_posix()
tracking_uri = f"sqlite:///{mlflow_database}"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)
```

`sqlite:///` is SQLAlchemy's SQLite connection-URI prefix. It is not a Windows folder name. Tracking stores experiment/run information; registry stores named versions. Here both point to the same database.

The loader finds version 1 using `MlflowClient`, retrieves its source run and `gold_run_id`, loads `models:/cmapss_fd001_rul_random_forest/1`, and reads feature metadata through the established API:

```python
feature_metadata_path = mlflow.artifacts.download_artifacts(
    run_id=source_run_id,
    artifact_path="metadata/final_training_features.json",
)
```

Checks compare feature count with `n_features_in_` and feature names/order with `feature_names_in_.tolist()`. It creates `shap.TreeExplainer(final_model, feature_perturbation="tree_path_dependent")`.

### Exported-folder loader: load_exported_model(serving_directory)

Reads `model_info.json` and `final_training_features.json` using `with open` / `json.load`, then loads `serving_directory / "model"` with `mlflow.sklearn.load_model()`. It performs the same feature checks and builds the same explainer.

It does not need the Windows tracking database to locate the model. It still uses the MLflow Python package to interpret the saved model.

Both loaders return a dictionary with `model`, `model_name`, `model_version`, `source_run_id`, `gold_run_id`, `feature_columns`, and `explainer`.

### Startup selection

`lifespan()` reads `os.getenv("MODEL_DIRECTORY")`:

- Absent -> original registry loader.
- Present and nonblank -> exported-folder loader with that path.
- Blank -> `ValueError`.
- Bad/missing exported files -> startup fails; there is no silent fallback.

The returned dictionary is stored in `app.state.model_data`. `get_model_data()` returns it or raises HTTP 503 with `{"detail": "The prediction model is unavailable."}`.

This guard handles missing state in a running app. A failed startup prevents normal HTTP service startup; it does not itself create a 503 response. A stopped container causes a connection failure, not an HTTP response.

## 10. Route behavior and batch logic

| Route | Behavior |
|---|---|
| `GET /health` | Requires loaded model; returns status, running message, `model_loaded: true` |
| `GET /model-info` | Returns model identity and training/Gold lineage IDs |
| `POST /predict` | Validate history -> latest feature row -> predict -> explain -> status -> one response |
| `POST /predict/batch` | Validate all engines -> prepare one row per engine -> concatenate -> predict/explain together -> ordered response list |

Batch inference does not concatenate raw engine histories into an ungrouped feature calculation. Each request is prepared separately, then its final feature row is concatenated using `ignore_index=True`. One model prediction call and one explanation call process the resulting rows. `zip(request.engines, predictions, all_factors)` associates each result with its engine.

Batch uses whole-request failure: invalid Pydantic input or invalid engineered values in one engine produce HTTP 422 for the request. There is no partial-success response contract.

`health_status()` uses exactly:

```text
RUL <= 10          -> critical
10 < RUL <= 30     -> warning
RUL > 30           -> healthy
```

## 11. SHAP factors

`explain_predictions(explainer, model_input)` calls the explainer with `check_additivity=True`. It converts SHAP values to a DataFrame with matching feature names, then constructs a separate factor list for each row.

Each factor stores the feature name, its input value, and its signed SHAP value. Ranking uses:

```python
factors.sort(
    key=lambda factor: abs(factor["shap_value"]),
    reverse=True,
)
```

`key` computes the size used for sorting; the lambda reads one factor's absolute contribution. `reverse=True` places the largest magnitude first. The original signed value is preserved. Return the first three factors per row.

Positive SHAP contributions raise predicted RUL relative to the baseline; negative contributions lower it. These are the strongest influences, not necessarily the three most negative “risk signals.” The complete prediction uses all feature contributions, not just the displayed three.

Use row positions consistently (`iloc`) so non-default DataFrame index labels do not misalign observations and explanations. Build fresh dictionaries/lists per iteration to avoid shared-object mistakes.

## 12. Structured logging, middleware, and error behavior

`JsonFormatter(logging.Formatter).format(record)` produces JSON with UTC timestamp, level, message, and a `details` dictionary. It includes formatted exception traceback when `record.exc_info` exists.

`configure_logging()` configures `maintenance.api` at INFO, disables propagation, and attaches one formatted StreamHandler if the logger has none. App logs go to the process output, which Docker exposes through `docker logs`. Uvicorn also prints its own server/access messages; not every output line is our JSON format.

`log_requests(request: Request, call_next)` wraps downstream request processing:

1. Record start time using `time.perf_counter()`.
2. Generate a request ID using `uuid4().hex` and capture available model version.
3. `await call_next(request)` lets validation/routing/endpoint code produce a response.
4. Log method, path, HTTP status, duration in milliseconds, request ID, model version.
5. Add `X-Request-ID` to the returned response and return it to the client.

`Request` is FastAPI/Starlette's HTTP request object. It is different from our Pydantic `PredictionRequest`, which represents the prediction body. `call_next` is supplied by the middleware mechanism; it passes the request onward.

| Situation | HTTP/client behavior | Logging behavior |
|---|---|---|
| Successful prediction | 200 with JSON | Completion log |
| Pydantic validation failure | 422 with error locations | Completion log with status 422 |
| Invalid numeric engineered history | 422 with detail text | Completion log with status 422 |
| Missing model state in running app | 503 with detail | Completion log with status 503 |
| Unexpected programming/runtime exception | Normally generic 500 if a response can be delivered | ERROR log with traceback; exception re-raised |
| Container stopped / startup failed | Connection failure or unavailable server | No normal endpoint response |

Handled `HTTPException` and validation errors normally become responses before `call_next` returns, so 422/503 do not enter our unexpected-exception block. Unexpected exceptions are logged with `logger.exception()` and a bare `raise` preserves propagation. A generic 500 can still be visible in the browser; traceback details stay server-side under the normal non-debug setup. The exception branch does not execute our normal response-header assignment, so do not promise `X-Request-ID` on every unexpected 500.

`return response` is necessary to pass the downstream response back through middleware. Logging it alone does not send it to the client.

## 13. Automated tests and lessons

Command from the project root with `.venv` active:

```powershell
python -m pytest tests -v
```

Last full suite verification after startup-selection changes: **28 passed, four dependency deprecation warnings**. Parameterization means there are more collected cases than named test functions.

Coverage includes:

- Health success and unavailable model.
- Model-info identity.
- Controlled single prediction response, features, factor/explainer plumbing.
- Batch row ordering, engine/result pairing, thresholds, and one predict/explain call.
- Missing sensor and duplicate cycles rejected before feature preparation.
- Invalid IDs, string IDs, short history, gaps, unexpected fields.
- Empty batches and undefined numeric features, including batch failure.
- 503 on model-dependent routes.
- Request logs, request IDs, response headers, handled 200/422/503, unexpected 500 propagation.
- Health thresholds around 10 and 30.
- SHAP ranking with large positive/negative values and correct row/value alignment.

Tests use `TestClient`; a separately running Uvicorn server is not required. Use another terminal if the current terminal is occupied by Uvicorn. Entering `with TestClient(api.app)` runs startup/shutdown for the in-process test app.

`monkeypatch.setattr(api, "load_registered_model", fake_model_loader)` temporarily replaces the attribute looked up by `main.py`; it does not rewrite the source file. The original is restored after the test. Likewise patching `api.app.state.model_data` changes that test app's state temporarily. Patching must occur where the code looks up the name.

Tests replace loading with controlled objects so they remain fast and do not depend on real MLflow artifacts. A tripwire function such as `unexpected_feature_preparation(*args, **kwargs)` raises immediately if invalid requests reach feature code. Successful 422 assertions then demonstrate rejection at the validation boundary.

`any(...)` checks whether at least one returned error matches the expected location/message. The custom history validator produces location `["body", "history"]`; a missing nested sensor produces `["body", "history", 0, "sensor_measurement_7"]` because nested validation knows the exact observation and field.

Test caveat for later conversations: most loader mocks patch the registry path. An externally set `MODEL_DIRECTORY` would select another branch. Unset it in the test terminal if needed; this does not change the Docker container's environment.

Known warnings: Starlette's TestClient warned that its httpx integration is deprecated in favor of httpx2; SHAP's plotting imports emitted matplotlib deprecation warnings. These were not failures and were not independently refactored during this phase.

## 14. Exporting the model for Docker

Windows MLflow database/artifact references should not be assumed to work unchanged inside a Linux container. The adopted solution is an explicit exported serving bundle.

`src/modeling/export_model.py`:

1. Uses the familiar explicit Windows project root and root SQLite database.
2. Checks the database exists before connecting, avoiding an accidental empty database.
3. Finds the registered model version, its training run, and Gold lineage ID.
4. Locates the model directory using `download_artifacts(artifact_uri=model_uri)`.
5. Retrieves feature metadata using the familiar `run_id` / `artifact_path` call.
6. Creates `output/serving_model` with `mkdir(parents=True)`.
7. Copies the entire model directory with `shutil.copytree()`.
8. Copies the one feature JSON file with `shutil.copy2()`.
9. Writes `model_info.json` using `with open(..., "w")` and `json.dump(..., indent=2)`.

`Path(...)` constructs a location; it does not create the directory. `mkdir` creates it. `copytree` copies all files/subdirectories from an existing folder; this is why one line creates many files. `copy2` copies one file and preserves file metadata where supported. Export copies artifacts; it does not retrain, re-register, or remove the originals.

```text
output/serving_model/
├── model/
│   ├── MLmodel
│   ├── model.skops
│   ├── requirements.txt
│   └── environment/example/supporting files
├── final_training_features.json
└── model_info.json
```

The exported files were checked against the originals. Model directory files matched by SHA256 and feature metadata matched byte-for-byte. The feature count is 137.

Run from the project root only when a new export is actually needed:

```powershell
python -m src.modeling.export_model
```

The destination already exists. The exporter deliberately raises `FileExistsError` rather than overwriting it. Do not delete or rerun exports unnecessarily when beginning Phase 9.

One temporary alternative implementation manually assembled an artifact URI. Badr requested the simpler familiar artifact call. It was restored and verified successfully outside the assistant's restricted execution environment. An earlier failure came from that restriction, not a reason to change the learned coding pattern.

## 15. Docker implementation and concepts

### Dependency file

`api/requirements.txt` pins the working environment's serving dependencies:

```text
fastapi==0.141.1
uvicorn==0.52.4
pydantic==2.13.5
mlflow==3.16.0
numpy==2.5.2
pandas==3.0.5
scipy==1.18.1
scikit-learn==1.9.0
skops==0.14.0
psutil==7.2.2
shap==0.52.0
```

`pip` also installs transitive dependencies. This file pins the listed packages; it is not a full lock of every transitive dependency. The broader `requirements-phase1.txt` still exists for project work, including notebooks. The container installs its own Linux-compatible packages instead of copying the Windows `.venv`.

### Dockerfile

The active `api/Dockerfile` is equivalent to:

```dockerfile
FROM python:3.14-slim
WORKDIR /app
COPY api/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt
COPY src /app/src
ENV MODEL_DIRECTORY=/model
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- `FROM`: existing small Linux/Python base image.
- `WORKDIR`: create/select `/app` inside the image. Python can then import `/app/src` as `src`.
- `COPY`: copy selected build-context files into the image.
- `RUN`: execute a build-time command, here package installation.
- `ENV`: default setting inherited by containers; it does not create/mount the folder.
- `CMD`: default command executed when a container starts. It is not executed as the running service during the image build.
- `src.api.main:app`: Python module and FastAPI object name.
- `--host 0.0.0.0`: listen on container network interfaces so forwarded requests can reach Uvicorn.
- `--port 8000`: container port. No development `--reload` in the image command.

### .dockerignore and build context

The build command runs from the project root. Its final `.` selects that directory as the build context. `-f api/Dockerfile` independently selects the instruction file.

Docker normally reads `.dockerignore` from the build-context root, so it belongs at `Python_3.0/.dockerignore`, not simply beside `api/Dockerfile`.

The ignore file excludes `.venv`, caches, `.git`, `.env` files, root databases/logs, notebooks, source data, `mlruns`, `output`, tests, and unrelated project folders. `api` and `src` remain available. Build output reported about 28.77 KB of transferred context.

Important distinction: `.dockerignore` filters build input; `COPY` selects what enters the image. Files not copied do not automatically enter the image even without an ignore file. Explicit ignores avoid unnecessary input and reduce accidental broad copying. Modern builders can also skip unused files. Ignoring `output` during build does not prevent mounting it when running a container.

### Why code is copied but model is mounted

Code is packaged into the image. The model folder is supplied at runtime so a compatible exported bundle can be changed without building it into another image. Copying the model into the image would also be valid, but that is not the chosen setup.

`output/serving_model` remains its Windows name. `/model` is the independently chosen location inside the container; the host folder is not renamed.

```text
Windows source:
C:\Users\Badr\Desktop\Python_3.0\output\serving_model
                         |
                 read-only bind mount
                         |
Container destination: /model
                         |
ENV MODEL_DIRECTORY=/model -> Python loads this folder
```

The Dockerfile does not supply the host source path. `docker run --mount` supplies it. `ENV` tells Python where to look; the mount makes the files accessible there. The source model stays outside the image, so distributing the image alone does not distribute the model.

### Build, create, stop, and restart

Build from `C:\Users\Badr\Desktop\Python_3.0`:

```powershell
docker build -f api/Dockerfile -t maintenance-api .
```

Build succeeded. The tag defaults to `latest`, a name rather than an automatic-update mechanism. Build output saved `maintenance-api:latest` and did not start the API.

Create/start a container for the first time:

```powershell
docker run -d --name maintenance-api -p 127.0.0.1:8001:8000 --mount "type=bind,source=C:\Users\Badr\Desktop\Python_3.0\output\serving_model,target=/model,readonly" maintenance-api:latest
```

- `-d`: background/detached execution.
- `--name`: container name.
- `-p 127.0.0.1:8001:8000`: laptop loopback port 8001 forwards to container port 8000.
- `--mount`: read-only access to the host bundle at `/model`.
- Final argument: image to use.

The image and container share the convenient name `maintenance-api`, but are different objects. The container already exists; use `docker start`, not another identical `docker run`, to restart it.

```powershell
docker ps
docker logs --tail 30 maintenance-api
docker stop maintenance-api
docker start maintenance-api
```

Stop asks the server to shut down. The browser stays open, but refreshing `/health` cannot connect while the server is stopped. Restart preserves the container's existing mount and port configuration and reloads the model. Neither stop nor start rebuilds the image. Image rebuilds do not update existing containers; changed image code requires a replacement container.

Docker Desktop hosts this container locally on Badr's laptop. Nothing was deployed to a cloud or remote computer. The browser is the client, not something hosted in the container. The container hosts Uvicorn/FastAPI, serving its docs page and prediction endpoints.

## 16. Verified Phase 8 results

The real registry-loaded service, exported-folder loading, and container-loaded service were exercised. Exported-folder startup was also tested with the registry loader replaced by a tripwire, proving that branch did not call the registry loader.

| Engine | Latest cycle | Predicted RUL | Status |
|---|---:|---:|---|
| 1 | 31 | 182.74043987965555 | healthy |
| 2 | 49 | 140.5686871044714 | healthy |
| 3 | 126 | 46.02538126361656 | healthy |

Minor floating-point differences around the last decimal occurred in one standalone load check; comparisons used tolerance `1e-9`.

Completed checks:

- All 28 automated tests passed after startup changes.
- Image build installed the pinned Linux/Python 3.14 packages and completed successfully.
- Container loaded the exported model; startup log reported model version 1.
- `/health`, `/model-info`, `/docs` returned 200.
- Single and batch predictions matched the known values and preserved engine order.
- Real SHAP factors were returned.
- A missing sensor returned 422 at `["body", "history", 0, "sensor_measurement_7"]`.
- A duplicate cycle returned 422 at `["body", "history"]`.
- Empty batch returned 422 at `["body", "engines"]`.
- Those invalid responses included request IDs; the API remained healthy afterward.
- Last observed container state during Phase 8 verification: running, published at `127.0.0.1:8001->8000/tcp`. Recheck state in a new conversation; processes are not guaranteed to remain running indefinitely.

No destructive MLflow cleanup, model retraining, model re-registration, MinIO volume changes, or modifications to other running project services were required for Docker serving.

## 17. Commands for continuing work

Activate the existing environment from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
```

Inspect/start the existing Docker service:

```powershell
docker ps -a
docker start maintenance-api
docker logs --tail 30 maintenance-api
```

Only use `docker start` if it is stopped. Wait for application startup/model loading before making requests.

Container browser addresses:

```text
http://127.0.0.1:8001/health
http://127.0.0.1:8001/model-info
http://127.0.0.1:8001/docs
```

Optional Windows development server using the registry loader:

```powershell
Remove-Item Env:MODEL_DIRECTORY -ErrorAction SilentlyContinue
python -m uvicorn src.api.main:app --reload
```

Optional Windows development server using the exported folder:

```powershell
$env:MODEL_DIRECTORY = "C:\Users\Badr\Desktop\Python_3.0\output\serving_model"
python -m uvicorn src.api.main:app --reload
```

These development commands default to laptop port 8000; the Docker API remains on 8001. Set environment variables in the same terminal before starting the program. They do not change settings in an already-running container.

MLflow UI from the project root:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Tests:

```powershell
python -m pytest tests -v
```

The assistant's restricted execution environment sometimes blocked Docker configuration locks, MLflow temporary artifact copies, local HTTP connections, and TestClient event-loop sockets. Running the same authorized checks outside that sandbox succeeded. Do not infer that working project code is broken from a restriction-only error. The local Python launcher sometimes printed a “Failed to find real location” message and still ran successfully; check exit status and actual output.

## 18. Model metrics available for Phase 9

These values were read read-only from the real source run's `latest_metrics` in the root database on 2026-10-18. They are saved evaluation results, not live dashboard measurements.

| Scope | MAE | RMSE | R² |
|---|---:|---:|---:|
| Official endpoint: one last row per test engine | 19.39744761810448 | 26.075723488548313 | 0.6062566881418086 |
| All eligible official-test rows | 31.512585272930266 | 41.74546676510065 | 0.44357960362033577 |

Additional source-run values: critical MAE `2.301758328603773`; warning MAE `6.929061251137193`; total operational cost `13400.0`; best cross-validation operational cost `8560.0`; average warning lead time `28.40909090909091`; 13 missed-critical observations; 33 false-alarm observations. Do not compare metrics from different sample scopes without labeling them.

Phase 7 residual metrics: mean `5.122498172239152`, median `-2.2127917833800304`, optimistic prediction percentage `53.09932118613791`. SHAP combined sample: 1,523 rows; predicted-critical explanation subset: only five rows. These sample limitations remain.

Full history, baseline comparisons, model selection rationale, importance tables, local cases, and other metrics remain in the historical Phase 5–7 appendix below.

## 19. Boundaries and remaining work

**Historical Phase 8/9 boundary record. Section 25 supersedes the statements below about absent Airflow orchestration and persistent health datasets.**

- Phases 8 and 9 are complete for the agreed educational/local/demo scope, not a claim of production readiness or cloud deployment.
- The Streamlit app exists at `dashboard/streamlit_app.py`. Swagger remains API documentation; it is not the dashboard. Phase 9 details, scope limits, and remaining integration work are recorded in Sections 22–24.
- The API is stateless with respect to engine histories: clients send full history for each prediction. It does not store incoming engine data or predictions in MinIO/a database.
- No live fleet inventory or automatically populated current-health table was added by the API.
- SHAP top factors are exact feature-level influences; natural-language sensor diagnoses remain a future design choice.
- No automatic model promotion, live reload of model artifacts, complete platform Compose file, CI pipeline, or project-specific Airflow orchestration was added here.
- Existing Airflow/MinIO infrastructure must be preserved. Keep unrelated running containers untouched.
- Git setup/commit logging and cleanup of historical MLflow databases/artifact layouts remain deferred unless deliberately addressed later.
- Official test data has already been used for final evaluation/explanation. Do not tune on it again without a new validation strategy.
- Historical notebooks are working educational artifacts; the whole training pipeline has not been rewritten into production scripts merely because inference logic now lives in `src`.

## 20. Phase 9 — confirmed original roadmap

The original roadmap explicitly defines Phase 9 as **Build the Streamlit dashboard** with these pages:

| Page | Planned content |
|---|---|
| Overview | Number of monitored engines; healthy/warning/critical counts; average predicted RUL; model version |
| Engine Explorer | Engine selection; sensor trends; predicted RUL; health status; risk factors |
| Model Performance | MAE, RMSE, R²; prediction plots; error distribution; model comparison |
| Data Quality | Latest ingestion time; row count; missing values; invalid records; drift indicators |
| Maintenance Queue | Engine, predicted RUL, status, priority |

Historical teaching sequence proposed before Phase 9 (retained to explain the original intent; actual implementation is documented in Section 22):

1. Explain what Streamlit does and how it differs from FastAPI and Swagger.
2. Agree on a small Phase 9 roadmap together.
3. Start a minimal Streamlit page and teach its rerun/widget behavior.
4. Connect it as an HTTP client to the existing API at `http://127.0.0.1:8001` while Streamlit runs on Windows.
5. Use one complete known engine history first, then add Engine Explorer and batch-based Overview/Maintenance Queue.
6. Decide the dashboard data source deliberately: raw Silver histories are suitable for API requests. Do not submit Gold engineered rows to the raw-history endpoint. Existing JSON examples are a useful initial demonstration source.
7. Add Model Performance from existing MLflow evaluation artifacts/metrics, labeling endpoint vs row-level scope. Do not retrain simply to display results.
8. Add Data Quality based on actual ETL metadata/validation outputs. Live drift indicators belong to later monitoring work unless explicitly implemented; display unavailable/not implemented rather than invented values.
9. Handle API loading delays and connection/422/503/500 failures visibly and simply.
10. Validate dashboard results against the existing API, then document Phase 9.

Localhost depends on where the client runs: a Windows Streamlit process can call laptop port 8001. A future Streamlit container would need a deliberate Docker-network address; `127.0.0.1` inside that container would refer to itself. Streamlit containerization/networking is not already implemented.

## 21. Suggested first message in the next conversation — Phase 12

```text
We are continuing my Industrial Predictive Maintenance MLOps project.

Project root: C:\Users\Badr\Desktop\Python_3.0

Read docs/project_complete_handoff_through_phase_8.md carefully. Despite its
filename, it now covers everything through Phase 11. Read the current status,
teaching preferences, and Section 26 especially. Inspect the existing tests and
source when needed. Preserve the historical appendices.

Phases 0–11 are complete within their documented local scope.
The FastAPI API is Dockerized as maintenance-api,
available at http://127.0.0.1:8001 when the existing container is running.
The exported model is mounted read-only from output/serving_model to /model.

The Streamlit dashboard is dashboard/streamlit_app.py. It has five views.
Overview and Maintenance Queue offer Demo engines or Latest Airflow run.
The ten-task cmapss_data_prediction DAG publishes 100 test-engine predictions,
health snapshots, and three monitoring reports to MinIO. Data Quality shows Raw
quality and feature drift. Model Performance shows the latest published batch's
report alongside the historical MLflow evaluation. Engine Explorer is a
three-engine demo.

We are starting Phase 12: broader testing and CI/CD. First inspect the tests
and project setup, then agree on a small roadmap and a finish line. The
separate training DAG was explicitly skipped by my decision. The deferred
monitoring checks in Section 26 are outside the Phase 11 closeout scope.

Teach one step at a time. Let me implement familiar Python/pandas patterns
with hints. Review my saved work, and when it is correct, continue to the
next small task automatically. Edit directly only when I request it; mark
only the latest code changes with NEW comments. Keep our existing naming
and explicit project_root / MLflow / MinIO patterns. Do not rewrite the
notebooks, retrain the model, or redo the API/dashboard unnecessarily.
```

---

## 22. Phase 9 — complete implementation and learning record

### 22.1 Scope, dates, and final delivery

Phase 9 was built interactively in this conversation, beginning on 2026-10-18 and concluding with verification/documentation on 2026-10-21. Badr started with no Streamlit experience. The assistant first reviewed the original roadmap chat, the combined handoff, source, notebooks, artifact files, and the root tracking database. Badr confirmed that `maintenance-api` was running.

The final implementation is a single Python entrypoint with five sidebar-selected views, not five separate page modules. The original roadmap's five user-facing areas are all represented. It is a local educational dashboard with deliberately limited data sources, not a live operational fleet system.

| File/resource | Phase 9 role |
|---|---|
| `dashboard/streamlit_app.py` | All five views, shared controls/state, MinIO helpers, requests, validation displays, and error handling |
| `requirements-phase1.txt` | Added `streamlit` and `requests` to the existing project dependency list |
| `docs/examples/predict_engine_1.json` | Existing complete engine 1 history used for the first single-engine lesson and verification |
| `docs/examples/predict_batch_engines_1_2_3.json` | Existing three-engine source used by final Explorer and batch Overview |
| Root `mlflow.db` and original artifact folders | Existing model identity, saved evaluation metrics, and plot artifacts; not retrained or re-registered |
| MinIO ETL manifests and Raw objects | Actual ingestion lineage, recorded row counts, timestamps, and selected-dataset quality inspection |
| This combined handoff | Updated in place through Phase 9; filename deliberately unchanged |
| `mlflow-ui-20261020.stdout.log`, `mlflow-ui-20261020.stderr.log` | Output from the MLflow UI process restarted during this conversation |

Installed versions checked during documentation: Streamlit `1.66.0`, Requests `2.34.2`, MLflow `3.16.0`. The requirements file lists package names without pinning those versions. Existing API pins in `api/requirements.txt` were not changed for the dashboard. No new dashboard-specific Dockerfile, Compose service, training artifacts, or persistent dashboard test files were created.

### 22.2 How the phase was built

| Step | Work and learning |
|---|---|
| 1 | Installed Streamlit into the existing `.venv`; created a title, explanation, practice engine dropdown, and selected-value display |
| 2 | Added a button calling `/health` and `/model-info`; explained request objects, HTTP errors, JSON decoding, and `try/except/else` |
| 3 | Added session state so API responses survive reruns; cleared old results before a fresh check |
| 4 | Submitted the existing full engine 1 JSON history to `/predict`; displayed RUL/status/version/factors |
| 5 | Made the engine dropdown select the correct full history for engines 1, 2, or 3; cleared results when the selected engine changed |
| 6 | Added a sensor dropdown and observed-cycle chart; moved history preparation outside the prediction button |
| 7 | Replaced raw factor JSON with a readable SHAP table and signed-effect labels |
| 8 | Added `/predict/batch`, fleet session state, health counts, average RUL, and model versions |
| 9 | Added priority mapping, sorted maintenance queue, health-status filtering, and an empty-result message |
| 10 | Added sidebar radio navigation, separated shared file loading from engine selection, and made views independent |
| 11 | Added Model Performance from the registered model's real source run; clarified endpoint versus all-cycle metrics |
| 12 | Added selectable saved engine curves, residual histogram, and a source-verified historical baseline table |
| 13 | Added Data Quality: MinIO connection, manifest selection, ingestion records, exact-object loading, schema/numeric checks, quality counts, and failed-row examples |
| 14 | Explicitly marked drift monitoring as not implemented until Phase 11 |
| 15 | Added missing-database/model/run, artifact-download, API, and MinIO error handling; verified success paths and nine simulated failures |

Several pandas tasks were implemented by Badr and reviewed; several later steps were directly edited only after he asked the assistant to do them. His questions and corrections changed the collaboration rules recorded in Section 1. Do not resume by implementing the entire Airflow phase autonomously.

### 22.3 Architecture and data sources

```text
Browser
  -> Windows Streamlit process: dashboard/streamlit_app.py
       |
       +-> saved JSON demo histories
       |      -> HTTP requests to FastAPI at 127.0.0.1:8001
       |      -> RUL, status, version, top factors
       |
       +-> root SQLite MLflow tracking/registry
       |      -> registered model version 1 -> source run
       |      -> saved metrics and downloaded plot artifacts
       |
       +-> .env -> boto3 -> MinIO
              -> ETL manifests -> latest complete successful Raw run
              -> exact selected Raw object -> read-only quality checks
```

The browser displays the app; Streamlit's Python process sends requests. FastAPI remains responsible for feature engineering, model prediction, status, and SHAP. Streamlit does not load the Random Forest for inference. The shared “Check API connection” control is visible on every view, but clicking it is not a prerequisite for viewing saved model evaluation or Data Quality.

`api_url` is explicitly `http://127.0.0.1:8001`; `project_root` remains the familiar explicit Windows path. These are local choices, not portable/container-network configuration. The MLflow UI web server is useful for manual inspection but is not required by Model Performance, which connects to SQLite directly.

Important source separation:

- Explorer/Overview/Queue use the existing JSON examples, not MinIO Silver histories or all 100 test engines.
- Data Quality independently selects a complete Raw ingestion run. It does not change the demo prediction inputs and is not restricted to the trained model's Gold run.
- Model Performance uses saved results for the registered model's source run.
- The initial suggestion to use Silver histories for a wider fleet remains future integration work. Never describe that suggestion as implemented.

### 22.4 Streamlit behavior and session state

Streamlit was introduced as a Python framework for browser-based interactive applications. The usual widget interaction reruns this script from top to bottom. Widgets return their selected values. A button returns `True` on the rerun caused by clicking it, then returns `False` on later unrelated reruns.

Initially, API responses displayed only inside `if st.button(...)`, so they disappeared after a dropdown change. The adopted pattern is to fetch/save inside the button branch, then display saved state outside it.

| Session-state key | Stored value / behavior |
|---|---|
| `health_data` | Last successful decoded `/health` response; initialized to `None` |
| `model_info` | Last successful decoded `/model-info` response; initialized to `None` |
| `prediction` | One last successful single-engine response; initialized to `None` |
| `fleet_predictions` | Last successful batch response list; initialized to `None` |
| `selected_engine_id` | Explorer's remembered engine ID, initialized on first Explorer visit; used to restore the dropdown on return |

Initialization uses `if "key" not in st.session_state:` so reruns do not overwrite existing results with `None`. A fresh API check clears old health/model responses; a fresh single/batch prediction clears its old result before attempting the new request. State is saved only after success, avoiding stale success displays following a failed new request.

When Explorer's selected engine differs from the saved prediction's `engine_id`, the prediction is cleared. Switching sensors preserves the current engine's prediction and does not send another prediction request. Switching views preserves state; returning to Explorer restores the selected engine via the separate `selected_engine_id` value. This is one saved engine result, not a dictionary of cached results for every engine. Changing to another engine clears it, so returning to the first engine may require a new prediction.

State belongs to the browser session; browser refresh/reconnection can reset it. It is not durable database storage or a live status cache. Captions explicitly identify API/fleet responses as results of the last check, not continuously refreshed measurements.

### 22.5 Shared setup and navigation

`st.sidebar.radio("Choose a view", ...)` returns `page`, with options in the saved source: Overview, Engine Explorer, Maintenance Queue, Model Performance, Data Quality. Independent `if page == ...` branches show the selected view. This is conditional navigation in one script, not `st.navigation`, `st.Page`, or a `pages/` module system.

The title and API connection section remain shared. The temporary `st.write(page)` debugging output was removed. All single-engine controls were grouped together under Engine Explorer rather than leaving the selector on every view.

The example JSON file is read once per script run, before the page branches, with `with open(..., encoding="utf-8")` and `json.load()`. `OSError` and `ValueError` show messages and call `st.stop()`. This keeps `example_data` available to both Explorer and Overview. It also means that an unavailable/malformed demo JSON file currently blocks even views that would otherwise not need it; this is a known implementation boundary, not a pending emergency fix.

When moving Overview behind a page condition, Queue was changed to reconstruct its own `fleet_df` from session state. It must not rely on a temporary DataFrame created only when Overview executes on that rerun.

### 22.6 Engine Explorer and exact known results

The engine selector offers IDs 1, 2, 3. A loop selects the matching whole request dictionary from `example_data["engines"]`; it does not replace one engine's ID while reusing another engine's measurements. No matching history produces `st.error()` and `st.stop()`.

`history_df = pd.DataFrame(prediction_request["history"])` supplies the chart. Sensor columns are selected by `sensor_measurement_` prefix. The sensor dropdown starts at `index=2`, meaning sensor 3, and `st.line_chart()` uses `time_in_cycles` for X and the selected sensor for Y. The chart shows observed raw measurements, not a forecast or a recomputed prediction trajectory. Selecting a sensor does not select the model's inputs: FastAPI still receives all required settings, all 21 sensors, and the full history.

The prediction button sends `requests.post(f"{api_url}/predict", json=prediction_request, timeout=120)`. Health/model-info use GET with `timeout=10`; batch POST also uses `120`. Requests' timeout concerns connection/read waiting, not a guaranteed total wall-clock limit for an entire operation. The longer prediction timeout accommodates feature/SHAP work. `raise_for_status()` checks HTTP failure; `.json()` decodes the body. RUL is displayed to one decimal using `st.metric`; the stored response keeps full precision.

| Engine | Complete submitted history | Predicted RUL | Status |
|---|---|---:|---|
| 1 | cycles 1–31 | 182.74043987965555 | healthy |
| 2 | cycles 1–49 | 140.5686871044714 | healthy |
| 3 | cycles 1–126 | 46.02538126361656 | healthy |

These matched real API checks during Phase 9, allowing negligible floating-point differences. The full-history EMA contract and exact 137-feature selection remain unchanged. Status still uses critical `<=10`, warning `>10 and <=30`, healthy `>30`.

The factor table uses the API's three ranked `top_factors`, with Feature, Feature value, SHAP contribution (cycles), and Effect. A positive contribution is labeled “Raises predicted RUL,” a negative one “Lowers predicted RUL,” and zero “No contribution.” No physical diagnosis or natural-language maintenance conclusion is inferred. These are three largest-magnitude contributions, not necessarily three negative risks, and their sum alone does not reconstruct the prediction.

### 22.7 Overview and Maintenance Queue

“Predict all demo engines” posts the complete existing `{"engines": [...]}` JSON dictionary to `/predict/batch`. The returned list is saved in `fleet_predictions`. Overview converts it to `fleet_df`, counts each status using Boolean comparisons plus `.sum()`, averages `predicted_rul`, and uses two rows of `st.columns(3)` for six metrics.

```python
model_versions = ", ".join(sorted(fleet_df["model_version"].unique()))
```

This takes unique version strings, sorts them, and joins them for display. Current output is `"1"`; it is string ordering, not a newly designed semantic version-ordering system.

Verified Overview: 3 engines, 3 healthy, 0 warning, 0 critical, average predicted RUL approximately `123.1` cycles, version `1`. These are three snapshots at different current cycles, not a live census or the entire FD001 fleet.

Queue takes engine ID, predicted RUL, and status from its own session-state DataFrame copy. Priority is a simple display rule: critical -> High, warning -> Medium, healthy -> Low. This mapping is not a learned second model. Sorting is ascending by predicted RUL, with engine ID breaking ties; the index is reset and display headings renamed.

The health filter offers lowercase `all`, healthy, warning, critical. The code's `if health_status == "all": pass` is valid; simplifying it was optional. Filtering affects only the queue copy, not the Overview data. If the result is empty, `st.info()` appears instead of an empty table, while the dropdown remains available. With current data, warning/critical filters return no rows. Queue order is engine 3, then 2, then 1; all priorities are Low. An unpopulated queue directs the user to Overview to run the batch button.

### 22.8 Model Performance: lookup, metrics, and evaluation scopes

Model Performance checks that the root database file exists before connecting. It constructs `mlflow_database`, the explicit `sqlite:///...` URI, configures tracking and registry, and creates `MlflowClient`. It retrieves registered model `cmapss_fd001_rul_random_forest`, version 1, then derives `source_run_id = model_version_details.run_id`. Missing source ID displays an error and stops. Separate `MlflowException` handlers protect model-version and source-run lookups.

Reading `.run_id` is attribute access on an already retrieved object; it does not require its own MLflow network-request handler. Badr initially wrapped it separately; that redundant handler was removed. Separate handlers for the two actual lookups were retained for clearer messages.

`metrics = source_run.data.metrics` is a dictionary. The evaluation-scope dropdown selects the appropriate keys/caption, then a single common three-column block displays the values. Do not duplicate the metric display inside both branches.

| Scope | Observations | MAE | RMSE | R² | Metric keys |
|---|---|---:|---:|---:|---|
| Final observed cycle per engine | 100, one per official-test engine | 19.39744761810448 | 26.075723488548313 | 0.6062566881418086 | `official_test_endpoint_mae`, `official_test_endpoint_rmse`, `official_test_endpoint_r2` |
| All eligible test cycles | 11,196, starting at cycle 20 | 31.512585272930266 | 41.74546676510065 | 0.44357960362033577 | `mae`, `rmse`, `r2` |

Both MAEs are averages of absolute errors. The difference is which observations are averaged. All-cycle evaluation gives longer observed histories more rows/weight; endpoint evaluation contributes one row per engine. MAE/RMSE are in cycles; R² is unitless. Display rounding is two decimals for MAE/RMSE and three for R².

Badr explicitly questioned whether `official_test_endpoint_mae` existed. It was confirmed directly in root SQLite `latest_metrics`, then through the restarted MLflow server's run API. It is a saved real metric, not an invented dashboard value. The actual run is `final_random_forest_training`, ID `360c5414c7d841a38bcf9480521a7b59`, in experiment **Default (0)**. The folds experiment contains the comparison reference run; do not confuse that with the source run.

### 22.9 Saved curves, residual histogram, and baseline comparison

The curve dropdown maps labels to existing artifacts:

| Label | Artifact path within the source run |
|---|---|
| Best engine | `plots/best_engine.png` |
| Middle-ranked engine | `plots/average_engine.png` |
| Worst engine | `plots/worst_engine.png` |

One `download_artifacts(run_id=source_run_id, artifact_path=artifact_path)` call and one `st.image(..., width="stretch")` display the selected curve. “Average engine” in the stored filename means the middle-ranked engine by eligible-cycle MAE; it is not an average curve across engines. Each image shows that engine's actual/predicted RUL trajectory from cycle 20. Switching metric scope does not regenerate or rescope these static trajectories.

The error-distribution image is `phase_7/plots/residual_distribution.png`, loaded into `residual_plot_path`. It covers all 11,196 eligible test observations regardless of the dropdown. Residual = actual RUL - predicted RUL. Negative means overestimated life/optimistic; positive means underestimated life/pessimistic. Near zero means small error. This is the original saved histogram, not a newly generated endpoint-only distribution.

Each artifact download has its own `try/except (MlflowException, OSError)/else`. Download failure displays a warning without `st.stop()`, allowing other plots and baseline results to remain visible. The corresponding image and caption are inside `else`. These handlers protect retrieval; rendering is outside the `try`, so they do not promise to catch every corrupt-image/rendering failure.

Baseline comparisons were found in Notebook 06's saved output cells 16, 20, 23, and 26, not in separate baseline MLflow runs. The app intentionally records those historical results as `baseline_results`, converts them to a DataFrame, and labels their provenance and 80-training-engine / 20-validation-engine scope. All 12 copied metric values were checked against the saved outputs exactly. No model was rerun merely to populate the table.

| Phase 5 model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean RUL | 52.27230702650257 | 63.15659578065061 | -0.0002068681218732049 |
| Linear Regression | 26.975360864588414 | 36.325707664853546 | 0.6691123380062189 |
| Random Forest | 26.280738453276047 | 38.858964457592464 | 0.6213527103303564 |
| XGBoost | 26.714122772216797 | 39.18028180631011 | 0.6150648593902588 |

The table is independent of the final-model evaluation selector and must not compare those validation scores directly against the final Random Forest's official-test score as if the samples were identical. Random Forest advanced for the balance of MAE, stability, and near-failure accuracy; Linear Regression had better overall RMSE/R². Final tuning/training occurred later in Phase 6. This table is not live experiment discovery.

### 22.10 Data Quality: MinIO access and run selection

`load_dotenv(project_root / ".env")` explicitly selects the project configuration file, rather than relying on `load_dotenv()` discovery from the execution context. Values come from existing environment names `MINIO_BUCKET`, `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, and `MINIO_SECRET_KEY`. No credential values are recorded here.

The boto3 connection is created only inside Data Quality. Helpers are defined at module level, adapted from Notebook 02:

- `read_parquet_prefix(s3_client, bucket_name, prefix)`: paginate keys, retain `.parquet`, read each object through `BytesIO` using `engine="fastparquet"`, concatenate with `ignore_index=True`. No files raises `FileNotFoundError`. Existing `print()` calls show loaded keys/shapes in the terminal, not the browser.
- `read_parquet_object(s3_client, bucket_name, object_key)`: read exactly one recorded object as a DataFrame, also with fastparquet.

Data Quality reads `metadata/etl_metadata/`, then filters to successful Raw records whose dataset names belong to `{train_data, test_data, RUL_data}`. Grouping `dataset_name` by `run_id` and applying `set` identifies coverage. A run qualifies only when all three required names are a subset of that coverage. The qualifying timestamp-based run IDs are sorted; the last one is selected. No qualifying run displays an error and stops.

`selected_raw_metadata` contains only successful Raw records from `pipeline_run_id`. The page displays the run ID and table. Latest recording time is the maximum of `pd.to_datetime(selected_raw_metadata["recorded_at"], utc=True)`, displayed with an explicit UTC zone. It is a manifest recording time for that selected run, not an independently measured live source-ingestion timestamp.

Per-dataset recorded row counts use `groupby("dataset_name")["row_count"].first().to_dict()`. This display can hide duplicate manifest records; qualification checks coverage, not record uniqueness. The subsequent selected-dataset inspection explicitly requires exactly one metadata row and stops if there are zero or multiple matches. Do not claim all manifest duplicates have been globally audited or deduplicated.

The dataset dropdown offers train_data, test_data, RUL_data. It filters the selected metadata, enforces `len(filtered_metadata) == 1`, retrieves the exact object_key, and loads `dataset_minio`. No broad Raw-prefix concatenation is used for that dataset.

### 22.11 Data Quality: schemas and measurements

Before row checks, the app defines expected columns:

- Train/test: `unit_number`, `time_in_cycles`, `operational_setting_1`–`_3`, and `sensor_measurement_1`–`_21` (26 total).
- RUL: `unit_number`, `remaining_useful_life` (2 total).

Set differences find missing and unexpected names. Any difference displays an error and stops. Expected names are also compared with `dataset_minio.select_dtypes(include="number").columns`; non-numeric expected columns display an error and stop. Data is not silently coerced or cleaned. These checks concern the existing Raw schema; Silver-only metadata columns are not expected here. Column-set checks do not enforce order or duplicate column-name uniqueness and do not constitute every possible pipeline schema check.

Row-level rules:

- All datasets: engine ID missing, non-positive, or non-integer is invalid.
- Train/test: cycle missing, non-positive, or non-integer is invalid.
- RUL: target missing or negative is invalid. Zero is allowed. The earlier use of `<=0` and an unnecessary integer-only RUL restriction were corrected.
- Integer-like checks use `% 1 != 0`. Boolean conditions combine with `|` into `invalid_rows` so overlapping failures count once.

The one-row summary uses `pd.DataFrame([summary])`:

| Summary value | Calculation / meaning |
|---|---|
| Actual row count | `len(dataset_minio)` |
| Missing cells | `dataset_minio.isna().sum().sum()` |
| Missing rows (saved heading is lowercase `missing rows`) | `.isna().any(axis=1).sum()` |
| Duplicate rows | `.duplicated().sum()`, counting repeated copies after the first, not every member of a duplicate group |
| Rows failing ID/cycle/RUL checks | `invalid_rows.sum()` |
| Rows with non-finite numeric values | numeric columns -> `np.isfinite(...).all(axis=1)` -> invert -> count selected rows |

`finite_mask` means every numeric cell in that row is finite. Its reverse includes NaN and positive/negative infinity. The earlier labels `infinite_rows` / “Rows with infinite values” were corrected to `non_finite_rows` / “Rows with non-finite numeric values.” Missing-value and non-finite counts overlap; do not add them or the invalid-rule counts to invent a total-invalid count.

The actual row count is compared with the selected manifest's row_count. A match displays success; a mismatch displays both counts in a warning. This does not modify either the object or its metadata. Expected original FD001 sizes are 20,631 train rows, 13,096 test rows, 100 RUL rows; keep their meanings separate rather than summing them into a misleading fleet/data total.

Failed examples use:

```python
failed_checks_mask = invalid_rows | ~finite_mask
failed_check_rows = dataset_minio[failed_checks_mask]
```

This is elementwise OR of aligned Boolean Series, not concatenation. Each failed row appears once even if it fails both groups. If empty, show the specific no-failures message; otherwise show the count and `.head(10)` with original row indexes. The preview explicitly covers ID/cycle/RUL or numeric-finiteness failures, not duplicate-row membership or every possible validation rule.

The page does not yet check conflicting engine-cycle duplicates, gap-free sequences, full cross-dataset label alignment, physical sensor range plausibility, or reference-distribution drift. Those existing notebook validations are not automatically all reimplemented in the dashboard.

### 22.12 Explicitly deferred drift monitoring

At page-level indentation, outside `if dataset:`, the app displays a Data drift heading and `st.info("Drift monitoring is not implemented yet. Planned for Phase 11.")`. Its caption says this page inspects stored Raw data from the selected run and does not monitor a live sensor stream.

Badr asked explicitly whether drift would wait until Phase 11; this was confirmed. Do not invent “no drift,” a zero drift score, green monitoring status, distribution thresholds, or completed monitoring jobs merely because this heading exists. Phase 11 must define reference/current samples, calculate distribution comparisons, choose thresholds, and feed real results into the dashboard.

### 22.13 Error-handling contract and current limits

| Operation | Current handling |
|---|---|
| Shared demo JSON file | `OSError` or `ValueError` -> readable error + stop |
| API health/model info | Timeout, connection, HTTP, other Requests error, or decoding failure -> readable error; old paired state cleared before retry |
| Single/batch prediction | Same general request pattern; HTTP errors include status; previous prediction state cleared before attempt |
| Missing root MLflow DB | File existence check -> error + stop before database setup |
| Model version / source run lookup | Separate `MlflowException` handlers -> error + stop |
| Model version with no source run | Explicit `None` check -> error + stop |
| Plot download | `(MlflowException, OSError)` -> warning; no stop; independent plots and baseline section can continue |
| MinIO manifest access | EndpointConnectionError, FileNotFoundError, ClientError -> specific error + stop |
| Selected Raw object access | EndpointConnectionError or ClientError -> specific error + stop |
| No complete Raw run, ambiguous selected object, missing/nonnumeric schema | Explicit errors + stop |

`EndpointConnectionError` means the endpoint could not be reached; `ClientError` means an error response such as access denied or missing object/bucket. A typo `st.errot` in a missing-manifest handler was found during review and fixed to `st.error`.

The app does not yet handle every possible configuration/corruption case: missing environment keys during boto3 setup, all SDK timeout variants, corrupt Parquet, malformed manifests, missing saved metric keys, successful-but-malformed API payloads, and image rendering errors can still surface exceptions. The implemented error paths are verified as described below; no blanket production-resilience claim is intended. Do not make these routine future-hardening items reasons to undo completed work or restart the phase.

### 22.14 Concepts Badr asked about and understood through examples

- `if st.button(...)` controls whether requests run. The `else` aligned with `try/except` executes when the try body completes without exception; it is not the button's “not clicked” case. Indentation determines the association.
- Separate try blocks can provide operation-specific messages. One combined block is also valid for a dependent lookup sequence. Do not wrap all unrelated rendering in one giant handler.
- `except (MlflowException, OSError)` catches either type; both need not happen. No stop is appropriate for optional plots, while stopping is appropriate when required data was not retrieved.
- `requests.get` reads health/identity; POST with `json=...` sends histories. Decoding JSON is not proof of HTTP success, hence `raise_for_status()`.
- `st.session_state` acts like a persistent-per-rerun dictionary; initialization must be conditional. It is session memory, not saved files or API history storage.
- `st.columns()` supplies containers; `.metric()` on one places the value there. Sidebar radio/selectbox returns a choice just like a normal variable value.
- `.unique()`, `sorted()`, and `", ".join(...)` successively deduplicate, order, and format version strings.
- Boolean `.sum()` counts True values. `|` combines row conditions using OR; `~` reverses Boolean values.
- `.isna().sum()` returns missing counts per column. A second sum counts all missing cells. `.any(axis=1)` asks whether any column is missing in each row, so summing counts affected rows instead of cells.
- Example used in teaching: rows `[10,20]`, `[NaN,25]`, `[NaN,NaN]` contain three missing cells but only two affected rows.
- A dictionary of scalar summary values cannot determine a DataFrame index on its own; `pd.DataFrame([summary])` means a list containing one row dictionary. A list with two dictionaries would form two rows.
- Dictionary lookups retrieve MLflow metrics or select an artifact path. Copying historical baseline values is valid only with explicit fixed-evaluation provenance and scope.
- Explicit `load_dotenv(project_root / ".env")` chooses the same intended project config deterministically; bare `load_dotenv()` discovers the file using execution context/parent search.
- RUL zero is valid; positive engine IDs/cycles and nonnegative RUL have different rules.

These concepts were explained in conversation rather than assumed from passing code. Continue teaching at this level; do not treat a working dashboard as evidence that unfamiliar Airflow concepts need no explanation.

### 22.15 Commands and operational notes

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run dashboard/streamlit_app.py
```

Default browser address: `http://localhost:8501`. Keep the terminal running; Ctrl+C stops Streamlit. The module form avoids relying on a separately launched Windows console executable. For installation on a recreated environment, use the existing project dependency file; Streamlit and Requests are included there. Do not create a new virtual environment per phase.

API inference requires `maintenance-api` on laptop port 8001. Inspect container state before starting anything; use `docker start maintenance-api` only if stopped, not another `docker run` with the same name. Data Quality requires `minio-cmapss` at the existing endpoint from `.env`; preserve its persistent volume. No other project's containers should be stopped or replaced for dashboard checks.

On 2026-10-20, MLflow UI was unavailable because no server was listening on 5000, not because endpoint metrics were absent. The assistant started it in the background with the existing `.venv` Python, hidden window, root cwd, and logs `mlflow-ui-20261020.stdout.log` / `.stderr.log`. HTTP health returned 200 and the REST run lookup returned all three endpoint metrics. This is a historical observation; do not assume that process is still running in a later chat.

To start the UI manually when port 5000 is free:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///C:/Users/Badr/Desktop/Python_3.0/mlflow.db --host 127.0.0.1 --port 5000
```

Direct saved-run browser URL:

```text
http://127.0.0.1:5000/#/experiments/0/runs/360c5414c7d841a38bcf9480521a7b59
```

Early installation/local checks were blocked by sandbox DNS/socket/Docker/process-inspection restrictions. Authorized retries outside those restrictions succeeded. One initial Streamlit rendering check stalled in the restricted session and was interrupted; a permitted rerun verified the chart and selections. Distinguish environmental restrictions from actual application failures. The Python “Failed to find real location” launcher message sometimes preceded a successful run; inspect exit status/output.

### 22.16 Verification evidence and its limits

Verification occurred throughout Phase 9, not as a new production CI pipeline:

1. Syntax checks passed after edits. Streamlit installation/version was verified.
2. Real API `/health` and `/model-info` returned the expected loaded-model identity. Real single-engine requests for all three examples matched the existing known predictions within floating-point tolerance.
3. Streamlit `AppTest` checks rendered the sensor chart, changed engine to 3, changed sensor to 17, displayed a real factor table, and retained prediction display after sensor changes.
4. Real batch Overview displayed 3/3/0/0, average 123.1, version 1. Its saved metrics remained on unrelated reruns.
5. A navigation check exercised cold Overview/Queue, Explorer engine 2 prediction, Overview batch prediction, Queue order [3,2,1], then return to Explorer with engine 2 and its prediction preserved.
6. Baseline table's 12 metric values were compared exactly with the saved notebook stream outputs. Plot file existence and residual PNG readability were checked.
7. Badr manually confirmed the navigation/saved-state sequence “works perfectly.” He subsequently confirmed completion of the Model Performance and Data Quality checklist: both metric scopes, all three curves, histogram/baseline table, train/test/RUL row counts matching manifest, zero indicated problems on the validated stored datasets, and explicit unavailable drift status. These are user-confirmed checks, not fabricated new assistant MinIO audit outputs.

On 2026-10-21, nine controlled error-path checks passed using `streamlit.testing.v1.AppTest` and temporary `unittest.mock` replacements:

| # | Simulated condition | Observed behavior |
|---|---|---|
| 1 | Requests connection failure on API check | Readable connection error, no unhandled app exception |
| 2 | API check timeout | Readable timeout error |
| 3 | API check HTTP 503 | Status-specific error |
| 4 | Single prediction HTTP 422 | Error displayed and saved `prediction` cleared |
| 5 | Both plot downloads raise MlflowException | Two warnings; three metric displays and baseline table remain visible |
| 6 | MinIO manifest endpoint connection failure | Readable MinIO connection error and clean stop |
| 7 | No manifest Parquet objects | No-manifests error |
| 8 | MinIO manifest listing AccessDenied ClientError | Metadata-request rejection error |
| 9 | Selected dataset NoSuchKey ClientError after valid synthetic metadata | Selected-object rejection error |

The final test used synthetic manifest bytes in memory, not new objects uploaded to MinIO. These simulations did not stop containers, mutate source data, retrain models, or edit source. Checks ran as temporary inline commands, not persistent dashboard test modules committed under `tests/`. Some setup paths used real local model metadata lookups; network/storage failures were injected. Do not describe these nine checks as nine new pytest tests or claim every possible exception branch was exercised.

The existing Phase 8 record of 28 passed tests remains historical; the full API pytest suite was not rerun simply for this documentation update. No fresh full data-ingestion/training pipeline was executed for Phase 9.

## 23. Current completion boundaries and future work after Phase 9

**Historical end-of-Phase-9 scope. Phase 10 added fleet inference, durable prediction tables, dashboard snapshot loading and its own Airflow Compose; see Section 25 for current boundaries.**

Phase 9 is complete for the scope actually built and verified above. Preserve these distinctions in any CV, README, future assistant response, or integration design:

- Five views exist, but inference/Overview/Queue cover three demo histories. No automatic discovery, arbitrary-history upload, all-engine MinIO inference integration, or live fleet table was added.
- Predictions and health checks are button-driven snapshots in browser session state. No durable prediction-history dataset, automatic refresh, event ingestion, or persistence to MinIO/database exists.
- Model identity is fixed to registered version 1 for the evaluation view. It is not automatic candidate promotion or a live comparison against newly trained models. API-exported bundle and local registry can diverge after future model changes unless deliberately coordinated; Phase 9 used matching version 1.
- Baseline comparison is a documented fixed validation snapshot. Evaluation images are static saved artifacts. No new model selection was performed on official test results.
- Data Quality reads actual selected Raw objects and runs the listed checks. It is not full pipeline-wide quality monitoring, sequence validation, data quarantine, or sensor-distribution drift detection.
- SDK/configuration edge cases and successful-response schema corruption are not exhaustively handled. Expensive file/metadata/artifact reads are not cached with `st.cache_data` or resources; no new concurrency/performance/load guarantee was made.
- Dashboard code remains one educational script with explicit Windows paths and ordinary sequential page branches. No broad refactor was requested. Do not replace it with abstractions merely to begin Phase 10.
- No Streamlit container/whole-platform Compose, CI workflow, cloud deployment, or new Airflow DAG exists from this phase.
- Existing model, model registry, ETL lineage, MinIO volumes, root/notebook artifact folders, notebooks, API, and other services must be preserved.

## 24. Historical Phase 10 plan before implementation

**Retained as planning history. Phase 10 is now complete under Section 25. Its proposed separate training DAG was explicitly omitted, rather than left pending.**

The original roadmap's next phase is **Automate workflows with Airflow**. Existing Airflow containers are infrastructure only; project DAGs must still be designed and implemented. At this documentation review, the project `airflow` directory contained no files. First inspect the actual existing Airflow deployment/configuration and available mounts without changing unrelated services. Do not assume an Airflow version, connection mechanism, executor, installed providers, or source-code accessibility.

Original inference/data DAG intent:

```text
check_source_data -> ingest_raw_data -> validate_raw_data
  -> transform_to_silver -> generate_features -> validate_features
  -> predict_current_health -> monitor_data_and_model
  -> publish_dashboard_dataset
```

Original separate training DAG intent:

```text
validate_training_data -> generate_training_features
  -> train_candidate_models -> evaluate_models
  -> register_best_model -> approve_model
```

These are target workflows, not implemented DAGs or permission to rerun training now. Full monitoring belongs to Phase 11; unavailable steps should be planned honestly instead of inserting fake successful monitoring tasks. Approval/promotion policy still needs deliberate design.

Recommended opening sequence (proposal for the next conversation, not completed work):

1. Read this handoff, especially Section 1 and Sections 22–24. Briefly explain DAGs, tasks, dependencies, scheduler, and how Airflow differs from the dashboard/API.
2. Inspect the existing Airflow environment and notebook/source dependencies. Identify which notebook logic must become reusable callable tasks without rewriting the educational notebooks.
3. Agree on a small initial DAG scope and a non-destructive development path. Preserve run_id/object-key lineage, complete-run selection, full-history EMA, trained feature order, and storage ownership.
4. Teach one task at a time. Introduce genuinely new orchestration concepts with examples; let Badr implement familiar Python himself, review saved changes, and continue automatically when correct.
5. Define idempotency, retries, logging, failure handling, validation, run/dataset IDs, and access to MinIO/model artifacts. Explicitly decide how a workflow will publish persistent prediction/dashboard data; the current dashboard examples do not supply that system automatically.
6. Keep training separate from routine inference. Do not tune on official test data again. Do not retrain or register a model just to test a DAG or redraw the dashboard.

The broader remaining roadmap is Phase 10 Airflow, Phase 11 data/drift/model monitoring, Phase 12 wider testing and CI/CD, Phase 13 integrated local Docker deployment, plus final documentation/Git work as deliberately scheduled. This document is the continuation source; there is no need to reconstruct Phase 9 from scratch.

---

## 25. Phase 10 — completed data and prediction orchestration

This section records the Phase 10 configuration. Section 26 describes the current workflow and dashboard.

### 25.1 Final scope and explicit decision

**Completed:** 2026-10-30 (UTC), after development beginning 2026-10-21. User-triggered real DAG runs completed successfully, followed by saved-output/dashboard verification. The agreed scope is one manually triggered data/inference workflow. The originally proposed training DAG was **explicitly skipped by Badr**, not implemented and not pending. It would repeat fitting on the same fixed benchmark; new run partitions do not supply additional engines or labels. There is no tuning, retraining, registration, comparison gate, model approval or promotion in this phase. Existing version 1 and its MLflow artifacts remain authoritative.

### 25.2 Actual workflow

```text
check_source_data -> ingest_raw_data -> validate_raw_data
  -> transform_to_silver -> generate_features -> validate_features
  -> predict_current_health -> publish_dashboard_dataset
```

`airflow/dags/predictive_maintenance_pipeline.py` defines DAG `cmapss_data_prediction` with TaskFlow decorators from `airflow.sdk`, `schedule=None`, `catchup=False`, `max_active_runs=1`, and UTC start date 2026-10-22. LocalExecutor parallelism is 2. All task edges use the default `all_success` rule. Ingestion, Silver, feature generation, prediction and publication each have one retry after 30 seconds. Source and validation tasks have zero retries; a failed check raises rather than silently allowing downstream work. No fake monitoring-success task was inserted; monitoring remains Phase 11.

TaskFlow calls inside the DAG build dependencies; task bodies run later. XCom carries string paths initially, then only `{run_id, load_date, metadata_key}`. DataFrames stay in memory within a task or in MinIO, not in XCom. Data/model exceptions propagate into task failure logs.

### 25.3 Environment and services

The old running Airflow deployment belonged to `C:\Users\Badr\Desktop\Python 2.0\SANLAM_PROJECT\airflow`, used CeleryExecutor/Redis and mounted SANLAM. At Badr's request its Airflow, PostgreSQL and Redis containers were stopped; images/containers/volumes were retained. Do not restart or modify SANLAM automatically.

This project has `airflow/Dockerfile`, `airflow/requirements-airflow.txt`, `airflow/docker-compose.yaml`, and `dags`, `logs`, `config` folders. Image `cmapss-airflow:3.3.1` extends `apache/airflow:3.3.1`; pip explicitly retains the base Airflow version while installing pandas, numpy, fastparquet, boto3, python-dotenv and requests. Those additional dependencies are presently unpinned. Airflow and Windows `.venv` are separate Python environments; model-serving packages were not installed into Airflow merely for inference.

Compose project `cmapss-airflow` contains PostgreSQL 16, a one-shot `airflow-init` (`db migrate`), API server (`api-server`), scheduler (`scheduler`) and DAG processor (`dag-processor`). Shared YAML settings use `x-airflow-common`, `&airflow-common` and `<<: *airflow-common`. Initialization overrides shared dependencies to wait only for healthy PostgreSQL; long-running services wait for database health and successful initialization. PostgreSQL uses its own `postgres-data` named volume at `/var/lib/postgresql/data`; a stopped init container with exit 0 is expected.

The UI is `http://localhost:8081` (loopback mapping 8081 -> container 8080). Simple Auth Manager has user `badr` with admin role. The generated password is in `airflow/config/simple_auth_passwords.json`; **do not copy credentials into this document**. The Compose settings are for local learning, not a production security design. There is no separate worker, Redis or triggerer in this project's stack.

### 25.4 Mounts, configuration and addresses

All relative host paths are resolved from the `airflow` Compose directory:

| Setting | Actual effect |
|---|---|
| `./dags:/opt/airflow/dags` | DAG definitions are visible to Airflow |
| `./logs:/opt/airflow/logs` | Execution logs remain accessible on Windows |
| `./config:/opt/airflow/config` | Config/generated login file is accessible on Windows |
| `..:/opt/cmapss` | The existing whole project is visible at `/opt/cmapss` |
| `PYTHONPATH=/opt/cmapss` | Makes `src...` imports available |
| `env_file: ../.env` | Compose injects project settings into containers; this is not a file mount |
| `MINIO_ENDPOINT=http://host.docker.internal:9010` | Overrides only the container endpoint; Windows `.env` is not edited |

Windows clients use MinIO `localhost:9010` and FastAPI `127.0.0.1:8001`. Airflow uses the same laptop ports through `host.docker.internal`. `localhost` inside a container refers to that container. MinIO maps laptop 9010 -> internal 9000; API maps laptop 8001 -> internal 8000. API client URL comes from an explicit argument, optional `PREDICTION_API_URL`, or default `http://host.docker.internal:8001`, with trailing slashes stripped. We did not require the optional environment variable to be set. Internal Airflow execution API is `http://airflow-apiserver:8080/execution/`; it is a different API from maintenance-api. The scheduler and Airflow API server share JWT signing configuration.

### 25.5 Module/function map

| File | Functions/responsibilities |
|---|---|
| `src/ingestion/source_checks.py` | `check_source_data`: three required FD001 files must exist and be nonempty; return string paths |
| `src/ingestion/raw_ingestion.py` | Parse Notebook 01 inputs; create MinIO client; read/upload exact Parquet objects; validate/read manifests; read named layer datasets; replace only owned manifest records; ingest Raw |
| `src/validation/raw_validation.py` | `validate_raw_frames`, `validate_raw_data`: schemas, numeric/finite/missing values, positive integer IDs/cycles, duplicate conflicts, consecutive histories and test/RUL alignment |
| `src/transformations/raw_to_silver.py` | `prepare_silver_data`, `transform_to_silver`: copies, exact deduplication, sorting, processing timestamp/status, Silver uploads and manifest update |
| `src/transformations/silver_to_features.py` | `prepare_gold_features`, `generate_features`: remove Silver-only columns, reuse established feature functions, add training RUL, upload final Gold feature tables |
| `src/validation/feature_validation.py` | `validate_feature_frames`, `validate_features`: Gold schemas, keys/source measurements, training target, history coverage and saved model-input contract |
| `src/modeling/batch_inference.py` | `build_engine_requests`, `check_api_identity`, `request_predictions`, `prediction_table`, `save_health_dataset`, `predict_current_health` |
| `src/transformations/publish_health.py` | `prepare_dashboard_health`, `publish_dashboard_dataset`: complete fleet/current-cycle validation, model/run consistency, SHAP JSON decoding/validation, priority, sorted durable snapshot |
| `dashboard/streamlit_app.py` | Existing readers plus `load_latest_airflow_health`; Airflow source selection, metrics, filtered queue, run/publication time |
| `tests/test_airflow_data_stages.py` | 45 offline tests using in-memory object storage and controlled API responses |

Notebooks were preserved. Existing API feature mathematics was reused, not replaced. Raw validation permits exact duplicate copies for Silver to remove but rejects conflicting duplicate engine/cycle records. Gold keeps all historical rows; first 19 cycles naturally lack complete rolling windows. EMA is calculated from cycle 1. `train_features` has 20,631 x 195; `test_features` has 13,096 x 194. Eligible rows are 18,731 and 11,196 respectively. The saved ordered 137 inputs are selected from the broader table; extra/removed features can contain undefined percentage changes, whereas selected inputs must be finite at cycles >=20. Engine ID and target cannot be model inputs. All engines must have eligible history.

### 25.6 Run lineage and publication rules

Run IDs derive from the DAG run's UTC `start_date`, formatted `%Y%m%dT%H%M%S%fZ`; dates use `%Y-%m-%d`. Automatic task retries within that execution use the same IDs/keys. Different manual runs create separate partitions even on the same day. Do not equate new partitions with new source information. Do not promise immutable dataset identity across every possible clear/reset of a completed DAG run: the current code uses `dag_run.start_date`, not an immutable externally supplied dataset ID. Prefer a new manual run for a fresh full execution.

Each run starts with three Raw records in one manifest under `metadata/etl_metadata/load_date=.../run_id=.../manifest.parquet`. Silver and Gold update that manifest while preserving other owned-stage records. `read_layer_data` verifies a unique successful dataset record, matching object prefix and row count. It supports explicit dataset names for Gold. A Raw rerun preserves downstream records and refuses changed source contents when downstream records exist. Manifest updates are sequential read/modify/write operations, not cross-object transactions or distributed locks. The local DAG is sequential with one active run; out-of-band concurrent writers are not coordinated.

Eight-task completion produces ten manifest records: Raw train/test/RUL; Silver train/test/RUL; Gold train_features/test_features/maintenance_predictions/current_engine_health. Gold prediction/health tables each have one row per test engine. Stages publish metadata after successful uploads. Failure tests confirm that a fresh incomplete stage does not publish new success records. Partial objects may remain; reruns target the same keys. No quarantine, rollback, versioned object retention or exactly-once transactional guarantee is claimed.

### 25.7 Inference and dashboard contracts

FastAPI receives complete raw test-engine histories from **Silver**, not engineered Gold rows. The task compares Silver/Gold engine-cycle keys first, then builds the API schema, checking full history through `BatchPredictionRequest.model_validate(...).model_dump()`. Defaults: batches of 10, request connection timeout 10 seconds/read timeout 180 seconds. These are Requests timeouts, not a total task deadline. API identity is compared with the serving bundle before/after all batches; each response is checked for schema, count/order, engine, latest cycle, version and 10/30 health thresholds. Progress is printed with flushing. No new snapshot is saved until all batches pass. Airflow does not directly load the RF/SHAP/MLflow model.

`maintenance_predictions` has 11 columns: engine_id, cycle, predicted_rul, status, model_version, top_factors_json, pipeline_run_id, prediction_time, model_name, source_run_id, model_gold_run_id. `prediction_time` is one UTC timestamp while assembling the completed snapshot, not per-engine measured latency. `top_factors_json` is JSON text, decoded by `json.loads` when needed. Training run/Gold lineage remain separate from the current input pipeline run.

`current_engine_health` adds priority (critical High, warning Medium, healthy Low), checks full current-engine coverage against Silver, and sorts by RUL then engine ID. Both use run-scoped Gold paths, not a mutable global latest pointer. The dashboard finds the greatest timestamp-formatted run ID among successful current_engine_health records, requires a unique record, reads the exact object and checks row count. It ignores newer runs that never published health; it does not query the overall Airflow DAG state or implement automatic fallback from a corrupt published object.

Overview/Queue offer `Demo engines` and `Latest Airflow run`. The Airflow branch loads on script reruns, displays run ID, UTC manifest publication time and engine count, uses stored results for six Overview metrics, and filters a DataFrame copy for Queue. Empty filters show an info message. `st.stop()` exits that branch before old demo code; it does not stop the server. The demo-history caption is conditional. No cache, timer-based refresh, new prediction request or DAG trigger is introduced by selecting Airflow data. Browser rerun/reload retrieves a newer snapshot. Engine Explorer remains a three-engine interactive demo; Model Performance remains saved evaluations; Data Quality remains stored-Raw checks with drift explicitly unavailable.

### 25.8 Verification and current saved evidence

- The user successfully triggered the expanding 1-, 2-, 4-, 6- and final 8-task workflows during development; publication succeeded before dashboard integration.
- Offline pipeline checks: 45 passed. They cover Raw/Silver/Gold, notebook feature equivalence, causality, input immutability, manifest/rerun preservation, failed uploads/batches, API identity/response failures and publication coverage. These use controlled storage/API fixtures, not production outage injection.
- Final full suite on 2026-10-30: **73 passed, six dependency deprecation warnings**, using `.venv` Python and `pytest tests -q`. The first restricted run stalled at FastAPI TestClient after the 45 pipeline tests; only that turn's identified pytest processes were stopped, and the permitted rerun completed in 24.90 seconds. Compose validation and dashboard/DAG syntax checks also passed. No code changes were required during phase closure.
- Full FD001 feature calculation/validation passed inside the scheduler against the saved 137-feature contract, with no MinIO writes.
- Three real demo API predictions matched the prior known values within tolerance; factor handling and queue sorting passed, with no stored output changes.
- Streamlit AppTest on real saved MinIO data verified Overview 100/82/16/2, average RUL 87.8, version 1; all three status filters returned the corresponding counts and Demo Overview remained available. Checks did not issue new predictions or write storage.
- Closure audit on 2026-10-30 imported the real DAG in the scheduler and verified its eight edges/tasks, retry policies, `all_success` dependencies and one active-run limit. It read the latest published run `20261027T234511255393Z` (load date 2026-10-27): ten unique manifest records, maintenance_predictions 100 x 11, current_engine_health 100 x 12, and health counts 82 healthy/16 warning/2 critical. No DAG run, retraining or storage mutation was triggered by that audit.
- Live task failures/retry transitions were not forced against the real MinIO/API deployment for closure. Failure propagation is covered by offline tests; retry/dependency configuration was inspected in the actual Airflow runtime. Do not describe this as a live chaos/outage test.

### 25.9 Completion boundary and next phase

Phase 10 is complete for this local scope. No training DAG needs to be completed. The model, registry, notebooks, historical manifests and existing artifact directories are preserved. Remaining roadmap: Phase 11 data/drift/model monitoring; Phase 12 broader CI/CD; Phase 13 integrated deployment and final documentation/Git work. Existing validation/logs are not a full drift/performance monitoring system. No automatic model promotion, continuous ingestion, adaptive learning, real-time sensor stream, cloud deployment or production guarantee is claimed. See `docs/phase10_airflow_guide.md` for routine commands.

## 26. Phase 11 — monitoring

### 26.1 Current workflow and saved reports

`cmapss_data_prediction` remains a manual local DAG. Its ten tasks run in this order:

```text
check_source_data → ingest_raw_data → monitor_raw_quality → transform_to_silver
  → generate_features → validate_features → monitor_feature_drift
  → predict_current_health → monitor_model → publish_dashboard_dataset
```

The three monitoring tasks each have one retry after 30 seconds. They save JSON in MinIO and return the small lineage dictionary through XCom. Each report key includes `load_date=<UTC date>/run_id=<pipeline run ID>/` between the prefix and filename.

| Task and module | Prefix | Filename | What it checks |
|---|---|---|---|
| `monitor_raw_quality` in `src/monitoring/data_monitoring.py` | `monitoring/data_quality/` | `raw_quality_report.json` | Schema and order, numeric and integer dtypes, missing or non-finite values, duplicate conflicts, cycle continuity, test/RUL alignment, and actual row counts versus manifest counts |
| `monitor_feature_drift` in `src/monitoring/drift_monitoring.py` | `monitoring/data_drift/` | `feature_drift_report.json` | Reference and current summaries for the 137 saved model inputs, with mean shift, spread ratios, percentile changes, and flags |
| `monitor_model` in `src/monitoring/model_monitoring.py` | `monitoring/model/` | `model_monitoring_report.json` | Prediction statistics, health counts, model version, label coverage, MAE/RMSE, HTTP request timing, and the MAE alert |

Raw monitoring is the single Raw gate in the DAG. It saves the report before raising on failed checks. The reusable Raw validator still protects Silver and feature preparation. Exact duplicate copies are reported and Silver removes them. Drift and accuracy flags are findings. They allow publication to proceed when monitoring succeeds. Input, read, and upload errors still fail the corresponding monitoring task.

### 26.2 Reference data, metrics, and dashboard behavior

- Drift uses `train_features` from the serving model's `gold_run_id` (`20260902T120041719003Z` for version 1) and `test_features` from the current pipeline run. Both use cycles 20 onward and the saved 137 inputs in order. Full training histories and truncated test histories can differ because of engine life stage. The report measures descriptive differences in center and spread.
- The demonstration mean-shift rule is `abs(current_mean - reference_mean) / reference_std >= 1.0`. The spread rule flags `current_std / reference_std` below 0.5 or above 2.0. Zero reference spread leaves these ratios and flags unavailable, represented as JSON `null`. Thresholds are recorded in each report.
- Model monitoring reads Gold `maintenance_predictions` from the current run. Silver `test_data` supplies the final observed cycle, and optional Silver `RUL_data` supplies the terminal labels. This FD001 DAG explicitly enables benchmark labels and records `fd001_benchmark` when it evaluates predictions. The reusable monitoring function defaults to labels disabled. Absent labels leave MAE/RMSE unavailable. Partial labels produce metrics for the matched engines and report their coverage.
- `src/modeling/batch_inference.py` times each validated `/predict/batch` HTTP request using `time.perf_counter()`. The model report stores the request count, mean duration, and maximum duration in milliseconds under `current_metadata.prediction_timing`. These are batch HTTP durations, including API processing and transfer. `prediction_time` is the UTC snapshot timestamp.
- `assess_accuracy_alert()` evaluates fully labeled batches. MAE at or below the limit is `within_limit`. MAE above it is `above_limit`. Partial or missing labels produce `unavailable`. `build_model_monitoring_report()` accepts `mae_limit`, with a 25-cycle demonstration default. Streamlit warns when the report says `above_limit`.
- Data Quality displays Raw quality and drift for the selected complete Raw run before the individual dataset checks. Model Performance reads the monitoring report for the same latest published snapshot used by Overview and Maintenance Queue. It shows batch accuracy, coverage, label source, and request timing alongside the historical MLflow evaluation. Older reports can omit timing and the alert.
- Airflow gets its MinIO environment from Compose. The Windows dashboard reads the project `.env`. The project mount remains `/opt/cmapss` in Airflow. Existing API, MLflow, MinIO, and training artifacts remain the inputs to the workflow.

### 26.3 Verification

The 2026-11-06 verification run `20261106T175911730403Z` completed all ten Airflow tasks and published its dashboard snapshot. The three MinIO reports matched that run ID.

| Check | Result |
|---|---|
| Raw quality | All checks passed |
| Feature drift | 137 features assessed, 0 mean-shift flags, 3 spread flags, 0 unavailable assessments |
| Model and coverage | Model version 1, 100 engines, 100% FD001 benchmark label coverage |
| Fleet health | 82 healthy, 16 warning, 2 critical |
| Endpoint accuracy | MAE 19.40 cycles and RMSE 26.08 cycles, consistent with the saved MLflow endpoint evaluation |
| Request timing | 10 requests, mean 5,590 ms, maximum 6,057 ms on this run |
| MAE alert | `within_limit` at the default 25-cycle limit |

Streamlit was checked with the saved MinIO report alongside the historical MLflow evaluation. Badr confirmed the timing display after a live run. Controlled alert checks covered values below, at, and above the limit, partial and absent labels, an older report without the alert, and an explicit threshold override. The above-limit warning was exercised with simulated data.

The focused pytest run passed all 137 cases across `test_data_monitoring.py`, `test_drift_monitoring.py`, and `test_model_monitoring.py`, with third-party pytest plugin discovery disabled. The full-suite retry exited before reporting test results. Reestablishing a reliable full-suite command belongs to Phase 12.

### 26.4 Local scope and next phase

The Phase 11 finish line covers the three saved reports, dashboard display, request timing, and an MAE alert. The following original checklist items were deferred for the local demo:

- Explicit sensor range rules.
- Detection of new engine IDs against a reference fleet.
- Alerts for abnormal row-count changes across runs. The implemented count check compares the object with its own manifest.

Each run records its own results. FD001 replays reuse the same benchmark data, so run history does not demonstrate a fleet changing over time. Phase 12 begins with a review of the existing tests and a bounded plan for broader testing and CI/CD.

## Historical appendices

The following two source documents are included in full to preserve the detailed earlier project record. Their old current-status statements and next-step instructions are historical. Sections 1–4, 21 and 25 above define the present state; other phase records preserve their historical scope.

<!-- HISTORICAL_HANDOFFS_START -->

## Appendix A — Original detailed history through Phase 4

Historical source: `docs/project_progress_summary_detailed.md`. Original content preserved below with heading levels adjusted. Its old next-step/status statements do not override the current handoff above.

Source SHA256: `b41eff36339513da9d2a7c0c325555ece676320431f2ecce891e42de22c06bb3`

### Industrial Predictive Maintenance MLOps Platform

#### Detailed project history and current status

**Date of update:** 2026-09-04  
**Dataset:** NASA C-MAPSS FD001  
**Main objective:** Predict the Remaining Useful Life (RUL) of an industrial engine from its operating history and sensor measurements.

This document summarizes the project direction, implementation decisions, problems discussed, work completed, and remaining work from the beginning of the project until the current feature-engineering stage.

---

#### 1. Project idea and business objective

The project is an end-to-end industrial predictive-maintenance platform.

The system is intended to:

- Ingest industrial sensor data.
- Store data in a MinIO-based data lake.
- Validate and transform data through Raw, Silver, and Gold layers.
- Create time-series features.
- Predict Remaining Useful Life.
- Convert predictions into health statuses.
- Track experiments and models with MLflow later.
- Serve predictions through FastAPI later.
- Display engine health through Streamlit later.
- Automate the workflow with Airflow later.
- Add data-quality, drift, and model monitoring later.

The main prediction target is:

~~~text
remaining_useful_life
~~~

RUL means the estimated number of operating cycles remaining before an engine reaches failure.

The initial health-status rules are:

~~~text
Healthy:  RUL > 30 cycles
Warning:  10 < RUL <= 30 cycles
Critical: RUL <= 10 cycles
~~~

The project is educational and portfolio-oriented. C-MAPSS is simulated benchmark data and is not real aircraft-maintenance data.

---

#### 2. Dataset and prediction contract

The first release uses the NASA C-MAPSS FD001 dataset because it has one main operating condition and one main degradation mode. The other C-MAPSS subsets can be considered later.

The dataset contains:

- Training data: engine histories that continue until failure.
- Test data: engine histories that stop before failure.
- RUL data: one terminal RUL value for each test engine.
- 26 columns in the original files:
  - unit_number
  - time_in_cycles
  - 3 operational settings
  - 21 sensor measurements

The intended row grain is:

~~~text
one row = one engine at one operating cycle
~~~

The initial prediction contract is:

~~~text
Input:  engine ID, cycle, operational settings, sensor measurements
Output: predicted RUL, health status, model version
~~~

For future feature engineering, the recent history of an engine can be represented using the latest 20 cycles.

The most important modeling rules established from the beginning are:

- Split data by engine ID, not by random rows.
- Never allow future cycles to influence features for an earlier cycle.
- Fit scalers and other learned preprocessing only on training engines.
- Preserve original input columns and add derived columns instead of silently replacing source information.

---

#### 3. Planned project roadmap

The original roadmap was designed as a complete MLOps project, but implementation is being done one stage at a time.

##### Phase 0 — Project contract

Define the business problem, data, target, users, metrics, architecture, limitations, and definition of success.

##### Phase 1 — Dataset understanding and ingestion

Understand the source files, create a data dictionary, validate the data, and upload the parsed Raw representation to MinIO.

##### Phase 2 — Data lake layers

Create Raw, Silver, and initial Gold datasets with validation, lineage, and reproducible object paths.

##### Phase 3 — Exploratory analysis

Study sensor distributions, correlation, sensor trends, degradation behavior, RUL relationships, outliers, stable sensors, and operating-condition behavior.

##### Phase 4 — Feature engineering

Create leakage-safe time-series features such as rolling statistics, differences, percentage changes, EMA, slopes, engine age, and volatility.

##### Phase 5 — Baseline models

Start with a mean RUL predictor, Linear Regression, Random Forest, and XGBoost. Evaluate with MAE, RMSE, R², error by RUL range, and business-oriented metrics.

##### Later phases

The planned later work includes:

- Final model training.
- MLflow experiment tracking and model registry.
- Explainability and error analysis.
- FastAPI prediction service.
- Streamlit dashboard.
- Airflow orchestration.
- Data-quality, data-drift, and model monitoring.
- Tests, CI/CD, Docker, and final documentation.

The project is currently between Phase 4 and the beginning of Phase 5.

Important numbering clarification: the feature-engineering notebook is called 05_Feature engineering_Final.ipynb, but feature engineering is Phase 4 in the project roadmap. The notebook number and phase number are not identical.

---

#### 4. Project environment and MinIO setup

A single project-specific virtual environment was created at:

~~~text
C:\Users\Badr\Desktop\Python_3.0\.venv
~~~

The same environment is intended to be used throughout the project rather than creating one environment per phase.

The main dependencies include:

- pandas
- numpy
- matplotlib
- seaborn
- Jupyter
- pytest
- python-dotenv
- boto3
- pyarrow
- fastparquet

fastparquet is used for Parquet input and output because PyArrow native components were affected by Windows Application Control.

MinIO is used as the object-storage data lake. The project-specific setup includes:

- Container: minio-cmapss
- Volume: minio_cmapss_data
- API endpoint: http://localhost:9010
- Console endpoint: http://localhost:9011

The port change was necessary because another MinIO container was already using ports 9000 and 9001. The project keeps MinIO’s internal ports unchanged but maps them to 9010 and 9011 on the host.

The .env file stores the endpoint, bucket, and credentials. Credential values were intentionally not copied into this documentation.

An important MinIO lesson was established: stopping or deleting a container does not delete its persistent volume. The volume preserves both the stored objects and the credentials initialized with that volume.

---

#### 5. Phase 0 completion: project contract

The project contract is stored at:

~~~text
docs/project_contract.md
~~~

It documents:

- The predictive-maintenance business problem.
- RUL as the main target.
- FD001 as the initial dataset.
- The input and output contract.
- Health-status thresholds.
- Intended users.
- Evaluation metrics.
- Leakage rules.
- Planned architecture.
- Limitations and exclusions.

The contract also makes clear that the project is a reproducible portfolio system rather than a certified safety system.

---

#### 6. Phase 1: dataset understanding and Raw ingestion

The current Phase 1 notebook is:

~~~text
notebooks/01_dataset_understanding_and_raw_ingestion.ipynb
~~~

##### Source-file handling

The original C-MAPSS files are space-separated text files without usable column headers. The notebook:

- Reads the training file.
- Reads the test file.
- Reads the RUL file.
- Assigns meaningful column names.
- Converts the values to numeric types.
- Adds unit_number to the RUL data so each RUL value can be associated with its test engine.

##### Validation and documentation

The notebook includes:

- Dataset shape checks.
- Missing-value checks.
- Duplicate checks.
- Engine-order checks between test and RUL data.
- Cycle-order checks.
- RUL consistency checks.
- Numeric-type checks.
- A detailed data dictionary.

The verified dataset sizes were:

~~~text
Training: 20,631 rows × 26 columns
Test:     13,096 rows × 26 columns
RUL:         100 rows × 2 columns
~~~

The data dictionary distinguishes identifiers, time variables, model inputs, and targets. The meanings and units of the 21 sensors were documented from the C-MAPSS information.

##### Raw-layer design decision

We discussed whether Raw should be an exact byte-for-byte copy of the original text files.

The current Raw layer is not the original text file itself. Before upload, the notebook:

- Parses the text into DataFrames.
- Assigns column names.
- Converts values to numeric types.
- Adds unit_number to the RUL table.

This is acceptable for this educational project because the Raw layer is an immutable parsed representation of the source data. The important rule is that after upload, Raw is not modified by later processing.

The distinction is documented conceptually: Raw is not an untouched byte copy, but it is the first immutable structured representation used by the pipeline.

##### Raw object versioning

The Raw upload was changed to use a generated run_id and a date-partitioned object path:

~~~text
raw/train_data/load_date=YYYY-MM-DD/run_id=<run_id>/train.parquet
raw/test_data/load_date=YYYY-MM-DD/run_id=<run_id>/test.parquet
raw/RUL_data/load_date=YYYY-MM-DD/run_id=<run_id>/RUL.parquet
~~~

This prevents repeated pipeline executions from being confused with one another and gives later stages a precise source version.

---

#### 7. The rerun problem and the run_id solution

One of the first important issues identified was that rerunning notebooks can duplicate data or make later notebooks read the wrong version.

The problem is not only duplicate filenames. It is also ambiguity:

- A notebook may upload a second copy of the same logical dataset.
- A later notebook may read every file under a prefix.
- Multiple files may represent different pipeline runs.
- The consumer may not know which files belong together.

The solution was to give each pipeline execution a run_id.

The same run_id is attached to the Raw, Silver, and derived Gold objects belonging to that pipeline lineage.

For example:

~~~text
run_id = 20260904T101530Z_a1b2c3
~~~

The ID itself does not select data automatically. Its purpose is to group related objects and let metadata identify the correct object keys.

---

#### 8. ETL metadata in MinIO

Because the project uses CSV/text source files and MinIO rather than a relational database, we decided not to introduce Oracle or PostgreSQL only for ETL metadata.

Instead, ETL metadata is stored in MinIO as Parquet data.

A metadata table in MinIO is not a database table with physically separate folders for each column. It is a normal Parquet file whose rows contain metadata records. Folder-like prefixes are used only to organize objects.

The manifest is stored under a path similar to:

~~~text
metadata/etl_metadata/load_date=YYYY-MM-DD/run_id=<run_id>/manifest.parquet
~~~

Typical metadata columns are:

~~~text
run_id
load_date
layer
dataset_name
object_key
status
row_count
recorded_at
~~~

One row represents one dataset produced by one pipeline stage.

For example, a Raw run can have three successful records:

~~~text
raw / train_data / success
raw / test_data  / success
raw / RUL_data   / success
~~~

##### Selecting the correct input

When Phase 2 reads Raw data, it:

1. Reads all metadata manifests.
2. Keeps successful Raw records.
3. Groups them by run_id.
4. Finds a run containing train, test, and RUL data.
5. Selects the latest valid run.
6. Reads the exact object_key recorded for that run.

The same principle is used later when reading Silver data.

This is why read_parquet_object is used after metadata selection: metadata has already identified one exact file. read_parquet_prefix is used when intentionally discovering and combining multiple Parquet files under a prefix.

##### Manifest updates

When a later stage writes new datasets, it:

- Reads the existing manifest.
- Removes only metadata records that the current stage is replacing.
- Adds the new records.
- Writes the updated manifest back to the same manifest key.

This makes reruns idempotent at the metadata level.

---

#### 9. Phase 2: Raw to Silver

The current Phase 2 notebook is:

~~~text
notebooks/02_raw_to_silver_layer.ipynb
~~~

Its responsibility is to read the selected Raw run from MinIO, validate it, create Silver DataFrames, and write Silver data back to MinIO.

##### Silver behavior

Silver DataFrames are created as copies of the MinIO-loaded Raw DataFrames so that Raw remains unchanged.

Silver contains:

- The original standardized columns.
- processing_timestamp.
- data_quality_status.

##### Validation and cleaning

The notebook validates:

- Missing values.
- Duplicate records.
- Conflicting duplicate engine-cycle records.
- Engine and cycle ordering.
- Complete cycle sequences where required.
- Test and train structural consistency.
- RUL constraints.

The work was expanded so Silver does more than merely repeat validation:

- Exact duplicate rows are removed.
- Conflicting duplicates for the same unit_number and time_in_cycles are rejected.
- Data is sorted by unit_number and time_in_cycles.
- Indexes are reset after cleaning.

The earlier issue where cleaned aliases were later overwritten by untouched Raw data was corrected. Silver is now written from the cleaned Silver working DataFrames.

##### Silver object organization

Silver objects are written under run-scoped paths such as:

~~~text
silver/train_data/load_date=YYYY-MM-DD/run_id=<run_id>/train.parquet
silver/test_data/load_date=YYYY-MM-DD/run_id=<run_id>/test.parquet
silver/RUL_data/load_date=YYYY-MM-DD/run_id=<run_id>/RUL.parquet
~~~

The Silver records are added to the ETL metadata manifest.

---

#### 10. Phase 3: Silver to Gold

The current Gold notebook is:

~~~text
notebooks/03_silver_to_gold_layer.ipynb
~~~

It remains a separate notebook by design.

Its current responsibility is to:

- Read Silver data.
- Create rul_training_features from the Silver training data.
- Calculate per-cycle training RUL.
- Remove the temporary last_cycle column.
- Write the result to the Gold layer.

The RUL calculation is:

~~~text
remaining_useful_life = final observed training cycle - current cycle
~~~

The dataset grain is:

~~~text
one row = one engine at one operating cycle
~~~

The current object organization is based on:

~~~text
gold/rul_training_features/
~~~

The notebook also documents planned Gold datasets such as:

~~~text
gold/current_engine_health/
gold/maintenance_predictions/
gold/model_performance/
~~~

Those datasets are intentionally not populated before the required health, prediction, and evaluation processes exist.

##### Decision about Notebook 03 and Notebook 05

We discussed merging the Silver-to-Gold and feature-engineering notebooks. The current decision is to keep Notebook 03 independent for now.

Notebook 05 reads directly from Silver, performs feature engineering, calculates the training target, and writes the final feature tables. This creates some duplicated RUL-calculation logic with Notebook 03, but it preserves the existing notebook structure and avoids changing Notebook 03 at this stage.

Later, the architecture can be improved by making Notebook 03 the creator of a reusable Gold base dataset and making Notebook 05 consume that base dataset. That is a future refactoring, not part of the current change.

---

#### 11. Phase 3: exploratory analysis

The EDA notebook is:

~~~text
notebooks/04_eda_and_sensor_analysis.ipynb
~~~

The notebook studies:

- Sensor distributions.
- Constant sensors.
- Correlations between sensors.
- Sensor trends over engine life.
- Early-life versus near-failure behavior.
- Train/test sensor similarities.
- Engine-level behavior.
- Potential degradation patterns.
- Outliers and noise.
- RUL distribution.
- Sensor-to-RUL relationships.

##### Main EDA observations

Sensors that generally increase over engine lifetime include:

- sensor_measurement_2
- sensor_measurement_3
- sensor_measurement_4
- sensor_measurement_8
- sensor_measurement_11
- sensor_measurement_13
- sensor_measurement_15
- sensor_measurement_17

Sensors that generally decrease include:

- sensor_measurement_7
- sensor_measurement_12
- sensor_measurement_20
- sensor_measurement_21

Some sensors are constant or nearly constant in FD001. They were not permanently removed only because their correlation was low or because they looked stable. Feature-selection decisions will be made later with explicit explanations and model evidence.

Some apparent outliers may be genuine degradation signals rather than errors, so they should not be removed automatically.

FD001 has one main operating condition, so a complete comparison across multiple operating regimes is not possible in this phase. The operational settings are preserved for future extensions.

##### Correct RUL interpretation in EDA

Training RUL is calculated for every cycle:

~~~text
training per-cycle RUL = final training cycle - current cycle
~~~

For test data, the provided RUL value corresponds to the final observed test cycle. To obtain a per-cycle test RUL for analysis only:

~~~text
failure cycle = maximum observed test cycle + supplied terminal RUL
per-cycle RUL = failure cycle - current cycle
~~~

This was checked and corrected during the discussion. The RUL is not constant across all test cycles after this conversion.

Test RUL labels should not be used as model input or as the main basis for feature selection. The final training target comes from the training data, while test RUL is primarily reserved for evaluation.

---

#### 12. Phase 4: feature-engineering concepts established

Before implementing features, the following concepts were clarified.

##### Current sensor value

The current sensor value is the original sensor measurement at the current engine cycle. It is retained as a feature, for example:

~~~text
current_sensor_measurement_7
~~~

##### A 20-cycle window

A row at cycle 30 uses the trailing history ending at cycle 30:

~~~text
cycles 11 through 30
~~~

It does not use cycles 31 or later.

For strict 20-cycle rolling features, the first 19 rows of each engine do not have enough history and therefore contain NaN.

##### Time-series features

Time-series features describe how a sensor changes over time rather than looking only at one isolated measurement.

##### Sensor slope

Sensor slope is numeric, not a text value such as increasing or decreasing.

It is the slope of a fitted straight line through the sensor values in the recent window:

~~~text
slope > 0  → sensor is generally increasing
slope < 0  → sensor is generally decreasing
slope ≈ 0  → sensor is relatively stable
~~~

Unlike a one-cycle difference, slope describes the overall direction across the window and is less sensitive to one individual jump.

##### Recent volatility

Recent volatility means how much a sensor has fluctuated recently. Rolling standard deviation is being used as the volatility measure.

##### EMA

The Exponential Moving Average gives more influence to recent values while gradually reducing the influence of older values.

For span=20:

~~~text
alpha = 2 / (20 + 1)
~~~

The recursive logic is:

~~~text
EMA_current = alpha × current_value
              + (1 - alpha) × previous_EMA
~~~

The mean in ewm(...).mean() calculates one EMA value for each row. It does not calculate the average of all EMA outputs afterward.

For an increasing sensor, the average of the EMA outputs can be lower than the average of the raw sensor values because the early EMA values were calculated before the later high values appeared. The value relevant to cycle 20 is the EMA at cycle 20, not the average of EMA values from cycles 1 through 20.

##### Wide feature table decision

The idea of storing each sensor as a separate table was considered. The current approach keeps one wide table with:

- Original/current sensor values.
- Derived features for each sensor.
- Engine and cycle identifiers.
- The training target where appropriate.

This is simpler for baseline model training because one row still represents one engine-cycle observation and all required model columns are together.

---

#### 13. Phase 4 implementation in Notebook 05

The current final notebook is:

~~~text
notebooks/05_Feature engineering_Final.ipynb
~~~

It reads the selected Silver run through ETL metadata and exact object keys.

##### Input preparation function

The reusable function is:

~~~text
prepare_feature_input(dataframe)
~~~

It does the following:

- Creates a copy.
- Removes Silver-only metadata columns.
- Renames sensor measurements with the current_ prefix.
- Sorts by unit_number and time_in_cycles.
- Resets the index.

Sorting is performed inside the function so it is not lost when the notebook is rerun or when the function is applied to test data.

##### Feature function

The reusable function is:

~~~text
add_time_series_features(dataframe, sensors)
~~~

It applies the same feature logic to any prepared dataset.

The current implementation creates:

1. Current sensor values.
2. 20-cycle rolling means.
3. 20-cycle rolling standard deviations.
4. 20-cycle rolling minimums.
5. 20-cycle rolling maximums.
6. Difference from the previous cycle.
7. Percentage change from the previous cycle.
8. Numeric 20-cycle sensor slopes using numpy.polyfit.
9. Exponential moving averages with span=20.

The rolling statistics and slope use:

~~~python
min_periods=20
~~~

This enforces a complete 20-cycle history. The EMA uses min_periods=1 because EMA can start at the first observed cycle and is not a strict fixed-size rolling window.

##### Applying the function to train and test

The same function is called for both datasets:

~~~python
train_df_minio = add_time_series_features(train_df_minio, sensors)
test_df_minio = add_time_series_features(test_df_minio, sensors)
~~~

The same sensor list and parameters are used for both, preventing train/test feature mismatch.

##### Percentage-change safety

Percentage change is calculated separately within each engine:

~~~python
percentage_change = (
    dataframe.groupby("unit_number")[sensors]
    .pct_change(fill_method=None)
    .replace([np.inf, -np.inf], np.nan)
)
~~~

This handles two cases safely:

- The first cycle of an engine has no previous value, so its percentage change is NaN.
- If the previous sensor value is zero, division can produce infinity. Infinity is replaced with NaN rather than being passed to a model.

The undefined value is not replaced with zero because an undefined percentage change does not mean that the sensor did not change.

##### Training RUL target

After feature creation, the training table receives:

~~~text
remaining_useful_life = last_cycle - time_in_cycles
~~~

The RUL column is a target, not an input feature. During modeling it must be separated:

~~~python
X_train = train_features.drop(columns=["remaining_useful_life"])
y_train = train_features["remaining_useful_life"]
~~~

The test feature table does not contain RUL as an input. Test RUL remains a separate evaluation label.

The target calculation uses future failure information only to create the label. It does not place future sensor values into the model features, so it is not feature leakage.

---

#### 14. Exporting engineered features to MinIO

The final feature-engineering notebook writes two Gold datasets:

~~~text
gold/train_features/load_date=YYYY-MM-DD/run_id=<run_id>/train_features.parquet
gold/test_features/load_date=YYYY-MM-DD/run_id=<run_id>/test_features.parquet
~~~

The training output contains:

- Engine ID.
- Cycle number.
- Operational settings.
- Current sensor values.
- Engineered features.
- remaining_useful_life as the target.

The test output contains the corresponding inputs and engineered features, but not the RUL target.

Each upload creates a metadata record containing:

- run_id
- load_date
- layer = gold
- Dataset name
- Exact object key
- status = success
- Row count
- Recording timestamp

The intended metadata update removes only existing records for:

~~~python
{"train_features", "test_features"}
~~~

for the current run_id and Gold layer. This preserves Notebook 03’s rul_training_features metadata and preserves Gold records from other runs.

The final metadata filter still needs to be applied to the notebook if it has not yet been changed manually. The approved filter is:

~~~python
etl_metadata_df = etl_metadata_df[
    ~(
        (etl_metadata_df["run_id"] == pipeline_run_id)
        & (etl_metadata_df["layer"] == "gold")
        & etl_metadata_df["dataset_name"].isin(
            {"train_features", "test_features"}
        )
    )
]
~~~

This replaces only the feature-engineering outputs while keeping the independent Gold output from Notebook 03.

---

#### 15. Current notebook inventory

~~~text
notebooks/01_dataset_understanding_and_raw_ingestion.ipynb
    Source understanding, validation, data dictionary, Raw ingestion

notebooks/02_raw_to_silver_layer.ipynb
    Raw selection, validation, Silver cleaning, Silver upload

notebooks/03_silver_to_gold_layer.ipynb
    Initial Gold RUL training dataset

notebooks/04_eda_and_sensor_analysis.ipynb
    Exploratory sensor and RUL analysis

notebooks/05_Feature engineering.ipynb
    Earlier non-final feature-engineering version

notebooks/05_Feature engineering_Final.ipynb
    Reusable feature functions applied to train and test, RUL target, Gold export
~~~

The final feature-engineering notebook was created as a new file so the earlier notebook remains available for comparison.

---

#### 16. Important issues identified and their current resolution

##### Rerunning notebooks can duplicate or confuse data

Resolved conceptually through run-scoped object keys and ETL metadata. Final execution and read-back verification are still required.

##### Raw is a parsed representation rather than a byte-for-byte source copy

Accepted and documented. The current Raw layer is an immutable structured representation of the original files.

##### Silver initially validated more than it transformed

Improved by adding duplicate removal, conflicting-duplicate rejection, sorting, and index resetting while preserving Raw.

##### Validation coverage was stronger for test than train

Reviewed and strengthened so train and test receive appropriate structural validation.

##### Gold contained Silver metadata columns

Corrected by removing processing_timestamp and data_quality_status before Gold and EDA use.

##### Notebook names were unclear

The active notebook names now communicate their responsibilities more clearly for the first three stages. The feature-engineering final notebook was created separately.

##### Portability and Git

These remain intentionally deferred until the notebook work is finished. The workspace was previously found without an initialized Git repository.

##### Gold metadata replacement scope

The replacement must be limited to train_features and test_features. Removing every Gold record would unnecessarily remove Notebook 03’s independent Gold dataset.

---

#### 17. Current completion status

##### Completed or substantially completed

- Project idea and objective.
- Project contract.
- Dataset understanding.
- Data dictionary.
- Environment setup.
- MinIO setup and endpoint correction.
- Raw ingestion.
- Run ID design.
- ETL metadata design in MinIO.
- Raw-to-Silver workflow.
- Silver validation and cleaning.
- Initial Silver-to-Gold workflow.
- EDA notebook and conclusions.
- Per-cycle RUL interpretation.
- Feature-engineering design.
- Reusable feature-engineering functions.
- Train/test feature calculation logic.
- Training RUL target creation.
- Gold feature-table upload method.

##### Still pending or requiring verification

- Apply the final metadata filter in Notebook 05 if it has not already been applied.
- Run the final notebook successfully against the intended MinIO instance.
- Read the uploaded train and test feature objects back from MinIO.
- Verify row counts and feature-column consistency.
- Decide how to handle rows with incomplete 20-cycle windows before training.
- Confirm there are no remaining infinite values or unexpected nulls in model-input columns.
- Perform engine-level train/validation splitting.
- Fit any scaler only on training engines.
- Build and evaluate baseline models.
- Add MLflow, API, dashboard, Airflow, monitoring, tests, CI/CD, and Docker later.

---

#### 18. Recommended next sequence

1. Apply the approved metadata filter in 05_Feature engineering_Final.ipynb.
2. Run the final notebook after confirming the project MinIO container and endpoint are active.
3. Verify that train_features and test_features exist at the expected run-scoped keys.
4. Read both objects back using their metadata records.
5. Check:
   - Shapes.
   - Columns.
   - Engine counts.
   - Cycle ordering.
   - RUL presence only in training.
   - Null and infinite values.
6. Decide whether the first 19 rows per engine should be dropped before model training.
7. Begin Phase 5 baseline modeling with an engine-level split.

The most important immediate rule is that the model must receive engineered inputs from train_features and test_features, while remaining_useful_life is separated from X_train and used only as y_train.

---

## Appendix B — Original handoff for Phases 5–7

Historical source: `docs/project_handoff_phases_5_to_7.md`. Original content preserved below with heading levels adjusted. Its old next-step/status statements do not override the current handoff above.

Source SHA256: `bcd223e95c32a3624b9fa85925d372c560639167734dc3b8ac4f1284af645100`

### Industrial Predictive Maintenance MLOps Platform

#### Detailed handoff after Phases 5, 6, and 7

**Last updated:** 2026-10-06  
**Project root:** `C:\Users\Badr\Desktop\Python_3.0`  
**Dataset:** NASA C-MAPSS FD001  
**Current status:** Phases 0 through 7 are considered complete. Phase 8, the FastAPI service, is next.

This document continues `docs/project_progress_summary_detailed.md`, which covers the project origin through Phase 4. Read that file first for the contract, MinIO lake, Raw/Silver/Gold processing, EDA, and feature engineering. This file records the later modeling, MLflow, explainability, troubleshooting, and handoff decisions.

---

#### 1. Collaboration preferences

- Explain concepts in beginner-friendly language before giving large code blocks.
- Keep code close to the user's existing style and avoid unnecessary abstractions.
- Keep related expressions on one line where reasonable.
- Do not directly modify a notebook unless the user explicitly asks.
- During reviews, explain errors and let the user correct them unless direct editing was requested.
- When editing directly, change only the requested part.
- Do not repeat analyses already completed in earlier phases.
- Distinguish carefully between index labels and positions, feature values and SHAP values, model logging and registration, and MLflow runs and registered models.
- Work one question and one step at a time.

---

#### 2. Current important files

##### Project history

- `docs/project_contract.md`
- `docs/project_progress_summary_detailed.md`

##### Modeling notebooks

- `notebooks/06_baseline_models.ipynb`
- `notebooks/07_mlflow_tracking.ipynb`
- `notebooks/08_explainability_and_error_analysis.ipynb`

##### Environment

- Virtual environment: `.venv`
- Dependency file: `requirements-phase1.txt`
- Correct tracking database: `C:\Users\Badr\Desktop\Python_3.0\mlflow.db`
- An older second database exists at `notebooks/mlflow.db`; do not accidentally use it.

There is no root `requirements.txt`. The existing dependency file now contains:

```text
pandas
numpy
pyarrow
fastparquet
matplotlib
seaborn
jupyter
pytest
python-dotenv
boto3
scikit-learn
xgboost
mlflow
optuna
shap
```

Observed installed versions on 2026-10-06:

```text
pandas 3.0.5
numpy 2.5.2
scikit-learn 1.9.0
xgboost 3.4.1
mlflow 3.16.0
optuna 4.9.0
shap 0.52.0
```

No extra dependency is needed for a square root. Use `math.sqrt()` from the standard library or `** 0.5`.

---

#### 3. Modeling contract

- One row represents one engine at one operating cycle.
- The feature window is 20 cycles.
- Cycles 1 through 19 are removed because they lack a complete window.
- Engine IDs are grouping metadata and never model features.
- `remaining_useful_life` is the target and never a model feature.
- Constant columns are learned from training data and removed from validation/test.
- Validation and folds split by engine, never by random rows.
- Model inputs must use the exact training feature order.
- The official test set was held back until the final family and settings were selected.

Health ranges:

```text
Healthy:  RUL > 30
Warning:  10 < RUL <= 30
Critical: RUL <= 10
```

For test data, the supplied C-MAPSS RUL belongs to the final observed cycle. Per-cycle actual RUL is reconstructed as:

```text
per-cycle RUL = supplied terminal RUL
                + final observed cycle
                - current cycle
```

---

#### 4. Verification before baseline modeling

##### Notebook 02

Raw-to-Silver validation was strengthened to:

- Raise an error for null values.
- Verify required columns.
- Reject NaN and positive or negative infinity in numeric data with `np.isfinite()`.
- Keep IDs and cycles numeric, integer-like, positive, and structurally consistent.

`np.where(~finite_mask)` identifies the row and column positions of invalid numeric values so the exception can show examples.

##### Notebook 03

The Silver-to-Gold notebook was corrected so it replaces only the `rul_training_features` metadata/object information it owns instead of removing unrelated Gold metadata.

##### Notebook 04

Training RUL appears correctly in the sensor-correlation heatmap. No further correction was needed for that point.

---

#### 5. Difference between Phases 5 and 6

Phase 5 compared model families under one common baseline setup. It answered which family should advance.

Phase 6 selected the exact configuration, checked it across different engine folds, logged the complete run, evaluated once on the official test set, and registered the final model.

The mean RUL predictor is a model baseline, not a metric. It always predicts the mean training RUL. MAE, RMSE, and R² then measure that constant model.

---

#### 6. Phase 5: `06_baseline_models.ipynb`

Notebook 06:

- Loads Gold features.
- Applies the 20-cycle rule.
- Separates engine ID and RUL from model inputs.
- Uses an engine-level 80/20 split.
- Detects constants from training data.
- Trains Mean RUL, Linear Regression, Random Forest, and XGBoost.

The correct warning condition is:

```python
(remaining_useful_life > 10) & (remaining_useful_life <= 30)
```

The shared evaluation function calculates:

- MAE, RMSE, and R².
- Actual/predicted range counts.
- MAE, RMSE, and signed error by healthy/warning/critical range.
- MAE, RMSE, and signed error by engine.
- Maximum engine MAE.
- Prediction changes and stability.
- Best, average, worst, and unstable engine curves.
- Missed critical observations and false alarms.
- Percentage of failures identified early.
- Average warning lead time.
- Premature, timely, and late first alerts.
- Engines without alerts.

Signed error is:

```text
actual RUL - predicted RUL
```

Positive means the model predicts too little remaining life and is pessimistic/early. Negative means it predicts too much life and is optimistic/late.

##### Saved baseline results

| Model | MAE | RMSE | R² | Critical MAE | Warning MAE | Mean absolute cycle change | Early failures identified |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean RUL | 52.2723 | 63.1566 | -0.0002 | — | — | — | — |
| Linear Regression | 26.9754 | 36.3257 | 0.6691 | 20.9992 | 17.9353 | 4.71 | 90% |
| Random Forest | 26.2807 | 38.8590 | 0.6214 | 3.0968 | 5.9064 | 3.56 | 100% |
| XGBoost | 26.7141 | 39.1803 | 0.6151 | 3.0794 | 6.8070 | 7.19 | 100% |

| Model | Missed critical rows | False alarms | False-alarm rate | Warning lead time | Premature / timely / late first alerts |
|---|---:|---:|---:|---:|---:|
| Linear Regression | 29 | 56 | 1.80% | 25.65 | 6 / 12 / 2 |
| Random Forest | 54 | 44 | 1.42% | 28.20 | 6 / 14 / 0 |
| XGBoost | 52 | 39 | 1.26% | 29.20 | 7 / 13 / 0 |

Observation-level missed critical rows and engine-level early identification measure different things. A model can miss some critical cycles yet still warn every engine before failure.

##### Phase 5 conclusion

Random Forest advanced because it gave the best overall balance:

- Lowest learned-model MAE.
- Lowest mean absolute cycle-to-cycle change.
- Strong critical and warning errors.
- 100% engine-level early identification in that split.
- No late first alerts.

Linear Regression had better RMSE/R² and fewer missed critical rows, so it remained a useful comparator. The decision was not based on one metric. The result used one 20-engine validation split, motivating five engine-level folds in Phase 6.

---

#### 7. Environment and service problems solved

##### Pandas

A broken/inconsistent pandas installation caused a partially initialized module error involving `_pandas_datetime_CAPI`. The environment was repaired, and later notebooks imported pandas successfully. If it returns, inspect the active kernel and pandas/NumPy installation before changing notebook code.

##### Slow first cells

First import cells sometimes appeared to hang. This was a kernel/environment issue rather than a modeling issue. Restarting the intended `.venv` kernel restored normal execution.

##### MinIO

Metadata-loading cells failed when MinIO was stopped. Data remained in the persistent Docker volume. Expected endpoints:

```text
API:     http://localhost:9010
Console: http://localhost:9011
```

##### VS Code widgets

The Jupyter-widget download popup concerned interactive rendering support, not model training or MLflow integrity. Static output continued to work.

##### MLflow UI

Windows Application Control blocked `.venv\Scripts\mlflow.exe`. Start MLflow from the project root with:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

The command keeps the terminal occupied because the server remains active. Open `http://127.0.0.1:5000`. Runs may take a few seconds to appear.

---

#### 8. Phase 6 concepts

##### Five engine-level folds

Each Optuna trial:

- Trains on 80 engines and validates on 20.
- Changes the validation engines between five fixed folds.
- Keeps all rows of an engine on one side.
- Calculates metrics in every fold.
- Logs the mean and standard deviation across folds.

The folds do not create separate Optuna studies. One trial tests one hyperparameter configuration across all folds.

##### Operational cost

```text
missed_failure_cost = 1000
false_alarm_cost = 400
maintenance_intervention_cost = 300
```

The implemented cost is:

```text
400  × premature first alerts
+ 1000 × (late first alerts + engines without an alert)
+ 300  × engines receiving an intervention
```

Maintenance intervention cost represents inspection, technician time, downtime, parts, or scheduled maintenance even when the alert is justified. Optuna minimizes mean operational cost across folds rather than MAE alone.

##### MLflow concepts clarified

- A run stores parameters, metrics, tags, tables, figures, and artifacts.
- The parent run is the Optuna study; child runs are trials.
- `infer_signature()` records input/output schema; it does not train.
- Logging stores a model artifact in a run.
- Registration creates a named version pointing to the source run.
- Metrics and figures remain on the source run, not primarily on the Registry page.
- `source_run_id` is a string; `client.get_run(source_run_id)` returns the full run object.
- Tracking URI stores runs/experiments; registry URI stores model versions. Both use the same SQLite database here.

---

#### 9. Phase 6: `07_mlflow_tracking.ipynb`

Notebook 07:

- Loads the selected Gold run.
- Stores its run ID in MLflow tags.
- Applies the 20-cycle rule.
- Creates engine folds.
- Learns constants from training engines per fold.
- Reorders validation/test columns to training order.
- Logs feature lists, folds, metrics, tables, and plots.

Selected Gold run:

```text
20260902T120041719003Z
```

The evaluation function returns:

```text
metrics
engine_metrics
unstable_engine
predicted_validation_rows
```

`engine_metrics` includes `unit_number`, MAE, RMSE, signed error, missed critical rows, and warnings. Event counts are reindexed with zero so engines without events remain present.

##### Optuna result

```text
Experiment:          cmapss_phase_6_random_forest_operational_cost_folds
Tuning parent run:   b692bdfb2e5e43d7abcc1e3d243a4241
Best child run:      9ec1ce8c389c40a5877e8c36a1778ea8
Best trial:          0
Best mean cost:      8560.0
n_estimators:        153
min_samples_leaf:    2
max_features:        None
random_state:        42
```

Logging `best_*` values does not retrain the model. The notebook separately retrieves the completed tuning run after a restart and reconstructs the parameters instead of rerunning Optuna.

##### Final training and evaluation

The selected Random Forest is trained once on all development engines. Folds are not used again during final fitting. The final run logs:

- Dataset identity and Gold run.
- Hyperparameters and costs.
- Training duration.
- Feature list and removed constants.
- Complete metrics and engine table.
- Engine plots.
- Model signature and artifact.
- Official endpoint metrics using one final row per test engine.

Endpoint metric names:

```text
official_test_endpoint_mae
official_test_endpoint_rmse
official_test_endpoint_r2
```

Read their exact values from MLflow rather than guessing from notebook output.

##### Registered model and runs

```text
Registered model:   cmapss_fd001_rul_random_forest
Version:            1
Alias:              candidate
Final source run:   360c5414c7d841a38bcf9480521a7b59
Reference run:      08f9cb42ee224fcf8298ce2d82589c5f
```

The source run was created in the Default experiment. Runs cannot be moved between experiments, so a reference run was created in the folds experiment for comparison. That reference copies parameters, metrics, and tags. The registered model still points to the real source run.

Phase 6 was declared complete. Git commit logging was deliberately deferred. A separate final justification document was considered optional because the rationale is already captured in Notebook 06 and MLflow.

---

#### 10. Phase 7 scope and reused work

Phase 7 answers which inputs matter, why critical predictions occur, where errors occur, and whether the model is optimistic.

The following were reused instead of repeated:

- Error by engine.
- Error by healthy/warning/critical range.
- Prediction stability.
- Actual-versus-predicted engine curves.
- Missed critical rows and false alarms.

Phase 7 added native importance, permutation importance, SHAP, local explanations, and residual plots.

---

#### 11. Phase 7 reload: `08_explainability_and_error_analysis.ipynb`

The notebook loads the registered model rather than retraining:

```text
Model:          cmapss_fd001_rul_random_forest
Version:        1
Source run:     360c5414c7d841a38bcf9480521a7b59
Gold run:       20260902T120041719003Z
Model class:    RandomForestRegressor
Test engines:   100
Test rows:      11,196
Model features: 137
```

It downloads `metadata/final_training_features.json` to recover exact feature names/order and removed constants. `final_model.n_features_in_` provides only the number of inputs, not the complete feature contract.

It selects the exact test objects by Gold run ID, merges terminal labels, reconstructs per-cycle RUL, applies the 20-cycle rule, reorders features, and validates missing/non-finite values and alignment.

---

#### 12. Native Random Forest importance

Native importance uses `final_model.feature_importances_` indexed by `X_test.columns`.

Engineered versions are grouped under their original sensor. The agreed simple method initializes groups from current sensor columns, removes `current_`, matches all engineered columns beginning with that sensor name, and sums them.

Verification:

```text
individual importance total = 1.0
grouped importance total    = 1.0
```

Top grouped native results:

| Input | Importance |
|---|---:|
| Sensor 17 | 0.228569 |
| Sensor 3 | 0.196791 |
| Time in cycles | 0.166264 |
| Sensor 4 | 0.072934 |
| Sensor 2 | 0.044472 |
| Sensor 21 | 0.037210 |
| Sensor 14 | 0.036604 |
| Sensor 11 | 0.035325 |

Native importance shows split usage/effectiveness, not whether a feature raises or lowers RUL.

---

#### 13. Permutation importance

Individual permutation importance uses:

```text
scoring = neg_mean_absolute_error
n_repeats = 10
random_state = 42
n_jobs = -1
```

The fitted model already contains its trees and parameters. Permutation importance does not retrain. It shuffles inputs, predicts again, and measures how MAE changes.

Grouped permutation moves all engineered columns of one sensor with the same random row order, repeats ten times, and records mean/std MAE increase. This avoids treating correlated engineered versions as unrelated.

Baseline row-level official-test MAE used here:

```text
31.512585272930266
```

Top grouped permutation results:

| Input | Mean MAE increase | Std |
|---|---:|---:|
| Time in cycles | 7.447435 | 0.118068 |
| Sensor 14 | 1.271350 | 0.069611 |
| Sensor 6 | 0.965200 | 0.038532 |
| Sensor 21 | 0.901833 | 0.050628 |
| Sensor 11 | 0.738064 | 0.073837 |
| Sensor 9 | 0.704659 | 0.025442 |
| Sensor 4 | 0.652030 | 0.026474 |
| Sensor 13 | 0.651241 | 0.029052 |

Negative importance for groups such as Sensors 7 and 8 means shuffling did not worsen MAE and sometimes improved it. It does not mean a physically negative effect on RUL. Native and permutation rankings answer different questions and need not match.

---

#### 14. SHAP analysis

SHAP explains:

```text
prediction = baseline + sum of SHAP contributions
```

A feature value is the observed input. A SHAP value is that input's contribution in predicted RUL cycles.

The explainer is:

```python
shap.TreeExplainer(
    final_model,
    feature_perturbation="tree_path_dependent",
)
```

Saved baseline:

```text
98.84317319546116 RUL cycles
```

##### Samples

```text
Engine-balanced sample: 1,000 rows, 10 per engine
Additional healthy:     300
Additional warning:     250
All actual critical:    17
All predicted critical: 5
Unique combined rows:   1,523
Features:               137
SHAP shape:             (1523, 137)
```

The complete dataset was avoided because SHAP was slow. The fitted Random Forest gives the same prediction for the same input, so recomputing predictions is safe.

##### Global bar and beeswarm

The bar uses mean absolute SHAP values so positive and negative effects do not cancel. Top combined-sample features include:

| Feature | Mean absolute SHAP |
|---|---:|
| `time_in_cycles` | 20.3798 |
| `sensor_measurement_3_ema_20` | 10.2819 |
| `sensor_measurement_17_ema_20` | 9.5416 |
| `sensor_measurement_21_mean_20` | 4.2277 |
| `sensor_measurement_4_ema_20` | 3.0406 |
| `sensor_measurement_2_ema_20` | 2.8068 |
| `sensor_measurement_11_sensor_slope_20` | 2.6282 |
| `sensor_measurement_7_min_20` | 2.0781 |

Current sampling caveat: the manual SHAP table/bar uses the combined 1,523 rows, which oversamples chosen RUL ranges. The global beeswarm uses only the engine-balanced 1,000 rows. State this distinction when interpreting them.

In a beeswarm:

- Each dot is one observation/feature.
- X is SHAP contribution in RUL cycles.
- Negative pushes predicted RUL down; positive pushes it up.
- Red is a high feature value; blue is a low value.
- Features are ordered by mean absolute effect.

##### Critical predictions

Critical predictions use `predicted_rul <= 10`. The pandas Boolean Series is converted to a NumPy mask to positionally filter the SHAP `Explanation`, which keeps SHAP values, data, baselines, and feature names aligned.

A critical bar and beeswarm were created. They describe only five predicted-critical rows, so do not generalize strongly from them. Leading features include Sensor 3 EMA, Sensor 17 EMA, time, Sensor 4 EMA, and Sensor 2 EMA.

---

#### 15. Local SHAP explanations

| Case | Engine | Cycle | Actual RUL | Predicted RUL | Status |
|---|---:|---:|---:|---:|---|
| Correct critical | 34 | 203 | 7.0 | 8.2389 | Critical |
| Missed critical | 31 | 194 | 10.0 | 12.9382 | Warning |
| False critical alarm | 76 | 204 | 11.0 | 9.9011 | Critical |
| Correct healthy | 1 | 20 | 123.0 | 169.3954 | Healthy |

These explain one engine at one cycle. Earlier line plots explain full engine trajectories.

`candidate_rows.iloc[0]` returns a Series. Its `.name` is the original index label. `observations_to_explain.index.get_loc(label)` converts that label to the numerical position needed for `shap_values[position]`. Labels and positions are not necessarily equal.

Waterfalls display ten rows, use `show=False`, retrieve the figure through `plt.gcf()`, and resize to 12×6.

For Engine 34, cycle 203:

```text
Baseline:          98.843
Prediction:         8.239
Actual:             7.0
Sensor 17 EMA:    -21.15 cycles
Sensor 3 EMA:     -20.33 cycles
Time in cycles:   -12.94 cycles
Sensor 4 EMA:      -6.00 cycles
Sensor 11 slope:   -5.75 cycles
```

Grey y-axis numbers are observed feature values. Bar labels are SHAP contributions. Blue lowers predicted RUL; red raises it. `E[f(X)]` is the baseline and `f(x)` is the final prediction.

---

#### 16. Residual analysis

```text
residual = actual RUL - predicted RUL
```

- Zero: perfect.
- Positive: prediction too low, pessimistic/early.
- Negative: prediction too high, optimistic/late.

The histogram uses all 11,196 rows and 40 equal-width bins across the observed residual range. It does not create bins 40 cycles wide.

The residual-versus-actual plot uses:

```text
X: actual RUL
Y: residual
horizontal line: residual 0
vertical lines: actual RUL 10 and 30
alpha: 0.3
point size s: 20
```

The dangerous region is actual RUL `<= 10` with residual `< 0`: truly critical equipment whose remaining life is overestimated. Engine grouping was not repeated because Phase 6 already recorded signed error by engine.

---

#### 17. Phase 7 MLflow logging

Phase 7 was logged into the real source run:

```text
360c5414c7d841a38bcf9480521a7b59
```

Logging into the existing run does not retrain or register another model.

Logged metrics:

- Mean and median residual.
- Optimistic-prediction percentage.
- SHAP observation count.
- Predicted-critical SHAP count.

Logged tables:

- Native and grouped native importance.
- Individual and grouped permutation importance.
- SHAP and critical SHAP importance.
- Local explanation records.

Logged figures:

- Global SHAP bar/beeswarm.
- Critical SHAP bar/beeswarm.
- Four local waterfalls.
- Residual histogram.
- Residual-versus-actual plot.

The tag `phase7_analysis_complete=true` was set. Output confirms successful logging.

The final cell also logged the notebook itself. This was later judged optional because all useful metrics, tables, and figures are already separate. The saved notebook still contains and executed that block, so the artifact currently exists. Future runs may remove it without affecting analysis results.

##### Artifact-location warning

Tracking uses the root database, but the experiment's artifact location is relative. Since Notebook 08 ran with `notebooks` as its working directory, final-run artifacts physically exist under:

```text
C:\Users\Badr\Desktop\Python_3.0\notebooks\mlruns\0\360c5414c7d841a38bcf9480521a7b59\artifacts
```

Do not delete `notebooks/mlruns`. Root `mlruns` also contains other runs. A future cleanup must first make artifact locations absolute or migrate them carefully.

---

#### 18. Identities to preserve

```text
Tracking DB:
C:\Users\Badr\Desktop\Python_3.0\mlflow.db

Experiment:
cmapss_phase_6_random_forest_operational_cost_folds

Tuning parent:
b692bdfb2e5e43d7abcc1e3d243a4241

Best child:
9ec1ce8c389c40a5877e8c36a1778ea8

Final source run:
360c5414c7d841a38bcf9480521a7b59

Comparison reference run:
08f9cb42ee224fcf8298ce2d82589c5f

Registered model:
cmapss_fd001_rul_random_forest

Version / alias:
1 / candidate

Gold pipeline run:
20260902T120041719003Z
```

---

#### 19. Completion and known limitations

##### Complete

- Phases 0–4: contract, ingestion, data lake, EDA, and feature engineering.
- Phase 5: four baselines, common evaluation, and Random Forest recommendation.
- Phase 6: folds, Optuna, cost selection, final training, test evaluation, logging, and registration.
- Phase 7: native/permutation importance, SHAP, local explanations, residual analysis, and logging.

##### Deferred or limited

- Git setup and commit logging.
- Cleanup of the old database and mixed relative artifact directories.
- Optional removal of notebook artifact logging.
- Critical SHAP has only five predicted-critical rows.
- SHAP bar and beeswarm samples use different weighting.
- The official test set has now been evaluated. Do not tune further on it without a new validation strategy.

---

#### 20. Next phase: FastAPI

Original Phase 8 endpoints:

```text
GET  /health
GET  /model-info
POST /predict
POST /predict/batch
```

The service should load registered model version 1 or alias `candidate`, validate inputs, preserve feature order, return predicted RUL/status/model version, provide structured errors, and later add tests and Docker support.

##### First required design decision

The model expects 137 engineered features, while an API consumer probably has raw sensor cycles. Choose the contract before coding:

1. Accept one already engineered 137-feature row. Easier, but less realistic.
2. Accept the latest 20 raw cycles and apply the exact training feature engineering. Better for a portfolio system, but reusable feature code must first move from notebooks into `src`.

The second design is preferred, implemented incrementally. Do not paste an entire notebook into an API route.

Recommended sequence:

1. Read both progress documents and inspect notebooks 07–08.
2. Confirm registered model version 1 loads from the root database.
3. Agree on request schema.
4. Implement `GET /health`.
5. Implement `GET /model-info`.
6. Create a reusable model loader.
7. Implement one validated `/predict` request.
8. Convert prediction to Healthy/Warning/Critical using 30/10.
9. Add `/predict/batch` after single prediction works.
10. Add focused API tests.

---

#### 21. Facts not to rediscover

- Random Forest was selected using accuracy and operational behavior together.
- Final settings: 153 trees, `min_samples_leaf=2`, `max_features=None`, `random_state=42`.
- The registered model already exists; Phase 8 should not retrain it.
- Use version 1 or alias `candidate` unless deployment policy changes.
- Recover exact features from `metadata/final_training_features.json`.
- `unit_number` and `remaining_useful_life` are not model inputs.
- The model expects 137 ordered features.
- Raw-cycle serving needs at least 20 ordered observations per engine.
- Use the root `mlflow.db` and start UI with `python -m mlflow`.
- Preserve the MinIO volume and both relevant artifact directories.
- Do not tune again on the official test set.

This is the state from which the next chat should continue.

---
