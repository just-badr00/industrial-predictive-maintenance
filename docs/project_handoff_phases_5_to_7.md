# Industrial Predictive Maintenance MLOps Platform

## Detailed handoff after Phases 5, 6, and 7

**Last updated:** 2026-10-06  
**Project root:** `C:\Users\Badr\Desktop\Python_3.0`  
**Dataset:** NASA C-MAPSS FD001  
**Current status:** Phases 0 through 7 are considered complete. Phase 8, the FastAPI service, is next.

This document continues `docs/project_progress_summary_detailed.md`, which covers the project origin through Phase 4. Read that file first for the contract, MinIO lake, Raw/Silver/Gold processing, EDA, and feature engineering. This file records the later modeling, MLflow, explainability, troubleshooting, and handoff decisions.

---

## 1. Collaboration preferences

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

## 2. Current important files

### Project history

- `docs/project_contract.md`
- `docs/project_progress_summary_detailed.md`

### Modeling notebooks

- `notebooks/06_baseline_models.ipynb`
- `notebooks/07_mlflow_tracking.ipynb`
- `notebooks/08_explainability_and_error_analysis.ipynb`

### Environment

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

## 3. Modeling contract

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

## 4. Verification before baseline modeling

### Notebook 02

Raw-to-Silver validation was strengthened to:

- Raise an error for null values.
- Verify required columns.
- Reject NaN and positive or negative infinity in numeric data with `np.isfinite()`.
- Keep IDs and cycles numeric, integer-like, positive, and structurally consistent.

`np.where(~finite_mask)` identifies the row and column positions of invalid numeric values so the exception can show examples.

### Notebook 03

The Silver-to-Gold notebook was corrected so it replaces only the `rul_training_features` metadata/object information it owns instead of removing unrelated Gold metadata.

### Notebook 04

Training RUL appears correctly in the sensor-correlation heatmap. No further correction was needed for that point.

---

## 5. Difference between Phases 5 and 6

Phase 5 compared model families under one common baseline setup. It answered which family should advance.

Phase 6 selected the exact configuration, checked it across different engine folds, logged the complete run, evaluated once on the official test set, and registered the final model.

The mean RUL predictor is a model baseline, not a metric. It always predicts the mean training RUL. MAE, RMSE, and R² then measure that constant model.

---

## 6. Phase 5: `06_baseline_models.ipynb`

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

### Saved baseline results

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

### Phase 5 conclusion

Random Forest advanced because it gave the best overall balance:

- Lowest learned-model MAE.
- Lowest mean absolute cycle-to-cycle change.
- Strong critical and warning errors.
- 100% engine-level early identification in that split.
- No late first alerts.

Linear Regression had better RMSE/R² and fewer missed critical rows, so it remained a useful comparator. The decision was not based on one metric. The result used one 20-engine validation split, motivating five engine-level folds in Phase 6.

---

## 7. Environment and service problems solved

### Pandas

A broken/inconsistent pandas installation caused a partially initialized module error involving `_pandas_datetime_CAPI`. The environment was repaired, and later notebooks imported pandas successfully. If it returns, inspect the active kernel and pandas/NumPy installation before changing notebook code.

### Slow first cells

First import cells sometimes appeared to hang. This was a kernel/environment issue rather than a modeling issue. Restarting the intended `.venv` kernel restored normal execution.

### MinIO

Metadata-loading cells failed when MinIO was stopped. Data remained in the persistent Docker volume. Expected endpoints:

```text
API:     http://localhost:9010
Console: http://localhost:9011
```

### VS Code widgets

The Jupyter-widget download popup concerned interactive rendering support, not model training or MLflow integrity. Static output continued to work.

### MLflow UI

Windows Application Control blocked `.venv\Scripts\mlflow.exe`. Start MLflow from the project root with:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

The command keeps the terminal occupied because the server remains active. Open `http://127.0.0.1:5000`. Runs may take a few seconds to appear.

---

## 8. Phase 6 concepts

### Five engine-level folds

Each Optuna trial:

- Trains on 80 engines and validates on 20.
- Changes the validation engines between five fixed folds.
- Keeps all rows of an engine on one side.
- Calculates metrics in every fold.
- Logs the mean and standard deviation across folds.

The folds do not create separate Optuna studies. One trial tests one hyperparameter configuration across all folds.

### Operational cost

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

### MLflow concepts clarified

- A run stores parameters, metrics, tags, tables, figures, and artifacts.
- The parent run is the Optuna study; child runs are trials.
- `infer_signature()` records input/output schema; it does not train.
- Logging stores a model artifact in a run.
- Registration creates a named version pointing to the source run.
- Metrics and figures remain on the source run, not primarily on the Registry page.
- `source_run_id` is a string; `client.get_run(source_run_id)` returns the full run object.
- Tracking URI stores runs/experiments; registry URI stores model versions. Both use the same SQLite database here.

---

## 9. Phase 6: `07_mlflow_tracking.ipynb`

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

### Optuna result

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

### Final training and evaluation

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

### Registered model and runs

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

## 10. Phase 7 scope and reused work

Phase 7 answers which inputs matter, why critical predictions occur, where errors occur, and whether the model is optimistic.

The following were reused instead of repeated:

- Error by engine.
- Error by healthy/warning/critical range.
- Prediction stability.
- Actual-versus-predicted engine curves.
- Missed critical rows and false alarms.

Phase 7 added native importance, permutation importance, SHAP, local explanations, and residual plots.

---

## 11. Phase 7 reload: `08_explainability_and_error_analysis.ipynb`

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

## 12. Native Random Forest importance

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

## 13. Permutation importance

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

## 14. SHAP analysis

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

### Samples

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

### Global bar and beeswarm

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

### Critical predictions

Critical predictions use `predicted_rul <= 10`. The pandas Boolean Series is converted to a NumPy mask to positionally filter the SHAP `Explanation`, which keeps SHAP values, data, baselines, and feature names aligned.

A critical bar and beeswarm were created. They describe only five predicted-critical rows, so do not generalize strongly from them. Leading features include Sensor 3 EMA, Sensor 17 EMA, time, Sensor 4 EMA, and Sensor 2 EMA.

---

## 15. Local SHAP explanations

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

## 16. Residual analysis

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

## 17. Phase 7 MLflow logging

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

### Artifact-location warning

Tracking uses the root database, but the experiment's artifact location is relative. Since Notebook 08 ran with `notebooks` as its working directory, final-run artifacts physically exist under:

```text
C:\Users\Badr\Desktop\Python_3.0\notebooks\mlruns\0\360c5414c7d841a38bcf9480521a7b59\artifacts
```

Do not delete `notebooks/mlruns`. Root `mlruns` also contains other runs. A future cleanup must first make artifact locations absolute or migrate them carefully.

---

## 18. Identities to preserve

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

## 19. Completion and known limitations

### Complete

- Phases 0–4: contract, ingestion, data lake, EDA, and feature engineering.
- Phase 5: four baselines, common evaluation, and Random Forest recommendation.
- Phase 6: folds, Optuna, cost selection, final training, test evaluation, logging, and registration.
- Phase 7: native/permutation importance, SHAP, local explanations, residual analysis, and logging.

### Deferred or limited

- Git setup and commit logging.
- Cleanup of the old database and mixed relative artifact directories.
- Optional removal of notebook artifact logging.
- Critical SHAP has only five predicted-critical rows.
- SHAP bar and beeswarm samples use different weighting.
- The official test set has now been evaluated. Do not tune further on it without a new validation strategy.

---

## 20. Next phase: FastAPI

Original Phase 8 endpoints:

```text
GET  /health
GET  /model-info
POST /predict
POST /predict/batch
```

The service should load registered model version 1 or alias `candidate`, validate inputs, preserve feature order, return predicted RUL/status/model version, provide structured errors, and later add tests and Docker support.

### First required design decision

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

## 21. Facts not to rediscover

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
