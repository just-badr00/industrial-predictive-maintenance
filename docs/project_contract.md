# Industrial Predictive Maintenance MLOps Platform

## Project contract

**Status:** Implemented local educational scope through integrated deployment

**Version:** 2.0

**Updated:** 2026-11-10 (UTC)

**Dataset:** NASA C-MAPSS FD001

## Business problem and objective

Estimate an engine's Remaining Useful Life (RUL), measured in operating
cycles, from operational settings and sensor history. Present the estimate,
a maintenance status and model explanations to maintenance/reliability
engineers and fleet operators. The project demonstrates the path from
validated industrial data to a monitored, locally deployed ML system.

## Dataset and grain

FD001 represents one main operating condition and one degradation mode.
Each source row describes one engine at one cycle, with an engine ID,
cycle number, three operational settings and 21 sensor measurements.

- Training histories continue until failure. Training RUL is final cycle
  minus current cycle.
- Test histories stop before failure. The RUL file supplies the label for
  each test engine's final observed cycle.
- Per-cycle test RUL used for analysis is terminal RUL plus final observed
  cycle minus current cycle. Labels never become model inputs.
- Train and test engine IDs belong to separate datasets, even when their
  numeric values coincide.

Raw stores parsed, typed Parquet representations of the text inputs rather
than byte-identical source files. Silver removes exact duplicates, rejects
conflicting records and sorts/validates histories. Gold holds features and
run-scoped prediction/health snapshots. ETL manifests identify exact objects
and lineage in MinIO.

## Prediction interface

The stateless FastAPI service accepts one engine through `POST /predict`
or several through `POST /predict/batch`. Requests contain complete raw
histories from cycle 1 to the requested current cycle, ordered and
consecutive, with at least 20 observations. Every observation supplies its
cycle, three settings and all 21 finite sensor measurements. Engine IDs
are positive integers. Unexpected fields and invalid histories are rejected.

Features are calculated within each engine using current values, trailing
20-cycle statistics/slopes, differences, percentage changes and recursive
EMA. The first complete rolling window ends at cycle 20. Complete history
is required because EMA retains earlier information. Inference selects the
saved 137 input columns by name and training order; engine ID and RUL are
not model inputs. Missing or non-finite selected inputs are rejected.

Responses contain:

| Field | Meaning |
|---|---|
| `engine_id`, `cycle` | Engine and latest observed cycle |
| `predicted_rul` | Estimated nonnegative remaining cycles |
| `status` | `healthy`, `warning` or `critical` |
| `model_version` | Serving model version |
| `top_factors` | Three largest-magnitude signed SHAP contributions, with feature names and values |

Status rules are critical for RUL <= 10, warning for 10 < RUL <= 30,
and healthy for RUL > 30. SHAP describes contributions relative to the
model baseline; it is not a physical diagnosis. Batch output preserves
request order and fails as a whole for invalid input.

`GET /health` checks loaded model availability. `GET /model-info` reports
model identity and training/Gold lineage. API requests do not automatically
persist incoming histories or their results to MinIO.

## Model selection and evaluation

Mean RUL, Linear Regression, Random Forest and XGBoost were compared using
engine-level validation. Random Forest advanced based on MAE, stability
and near-failure behavior, followed by five engine-level folds and Optuna
selection using operational cost. The final registered model is
`cmapss_fd001_rul_random_forest`, version 1, alias `candidate`: 153 trees,
`min_samples_leaf=2`, `max_features=None`, `random_state=42`.

Engine-level splits keep every engine's rows on one side. Constant-column
selection and learned preprocessing use training engines only. Features
must not use future observations. The official test set was evaluated after
model/configuration selection; further tuning on it requires a new
validation strategy.

Evaluation includes MAE, RMSE, R², error by engine and RUL range, signed
residuals, warning lead time, missed-critical/false-alarm observations and
engine-level first-alert behavior. Saved final-model results are:

| Scope | Observations | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Official final observed cycle | 100, one per test engine | 19.3974 | 26.0757 | 0.6063 |
| All eligible official-test cycles | 11,196 | 31.5126 | 41.7455 | 0.4436 |

These are distinct evaluation scopes, not interchangeable scores. SHAP
analysis used a sample; only five predicted-critical rows were explained
in the saved critical analysis.

## Implemented architecture and operation

- pandas processes FD001 data; MinIO stores Raw/Silver/Gold, manifests and
  monitoring JSON. PySpark is not used.
- Notebooks contain exploration, training and model analysis. Reusable
  operational pipeline, serving and monitoring code lives under `src/`.
- MLflow tracks experiments and model versions. The container server uses
  a prepared SQLite copy and mounted historical artifacts. PostgreSQL
  stores Airflow metadata only.
- FastAPI loads the exported model and SHAP explainer once at startup.
  Streamlit requests predictions over HTTP and reads saved fleet snapshots
  and monitoring reports from MinIO.
- Airflow runs one ten-task, manually triggered data/prediction DAG. It
  does not retrain, register or promote models. Raw-quality failures gate
  downstream processing; descriptive drift and accuracy findings can be
  published without failing the DAG.
- Monitoring records Raw quality, feature-distribution comparisons,
  prediction statistics, optional labeled endpoint accuracy, batch HTTP
  timing and an MAE alert. The FD001 DAG enables benchmark labels. Default
  mean-shift/spread rules and the 25-cycle MAE limit are demonstration
  thresholds, not operationally calibrated safety limits.
- Overview and Maintenance Queue show saved Airflow snapshots or three
  demo engines. Engine Explorer remains a three-engine demonstration.
  Dashboard reruns fetch saved state; there is no sensor stream or automatic
  timer-based refresh.
- Docker Compose runs the integrated local services. GitHub Actions checks
  code, tests, DAG loading, four image builds and Compose configuration.
  It does not automatically deploy images.

## Acceptance and limitations

The agreed scope is demonstrated by a successful ten-task FD001 run,
published fleet/monitoring outputs, working prediction and dashboard views,
model registry/artifact access, passing CI and restartable Compose services.
See [README](../README.md) for resources, commands and URLs.

Deployment requires supplied datasets, model files, historical MLflow state
and persistent volumes; a fresh clone alone cannot recreate them. Existing
preparation utilities use the original Windows paths. Direct-package pins
and mutable base images are not a complete immutable dependency lock.
Manifest publication is not a multi-object transaction, and out-of-band
concurrent writers are not coordinated.

Explicit sensor-range rules, reference-fleet new-engine detection and
cross-run abnormal row-count alerts remain outside the completed monitoring
scope. Benchmark replays create new run partitions, not new training
information or evidence of a changing live fleet.

C-MAPSS is simulated data. The project is for education and portfolio
presentation, not real aircraft-maintenance or safety decisions. Cloud
hosting, streaming, Kubernetes, Kafka, deep learning, production
certification and automatic retraining/promotion are outside scope.
