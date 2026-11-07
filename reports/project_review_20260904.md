# Predictive Maintenance Project Review

**Review date:** 2026-09-04  
**Scope:** Original roadmap task, two implementation/progress tasks, project contract, progress summaries, source data, all six notebooks, and the live MinIO datasets.

## Overall assessment

The project has a sound educational data pipeline from local FD001 files through Raw, Silver, and Gold feature tables. The implemented feature calculations are reproducible, causal, isolated by engine, and consistent between training and test data. The current MinIO objects match fresh notebook calculations.

The project is ready to begin Phase 5 as an exploratory baseline-modeling stage after making one modeling-data decision: use only rows with a complete 20-cycle history. It is not yet the modular, production-oriented MLOps platform described in the original roadmap. Most implemented logic remains in notebooks, while `src/`, `tests/`, `api/`, `airflow/`, and `dashboard/` contain no implementation files.

## Evidence gathered

- Every code cell in every notebook parsed and executed successfully with Python 3.14.3, pandas 3.0.5, and NumPy 2.5.2.
- All upload calls during review were redirected to memory. No MinIO object was changed.
- The review read back all nine datasets represented in the live ETL manifest.
- All live manifest row counts match the corresponding objects.
- Raw and Silver content matches the local FD001 files after removing Silver metadata columns.
- Live `train_features` and `test_features` match fresh feature recomputation exactly.
- All generated DataFrames survived a Parquet write/read round trip without changing.
- The EDA notebook generated 63 figures during headless execution.

Machine-readable evidence is in `reports/notebook_review_evidence_20260904.json`. The repeatable review harness is `reports/review_notebooks_20260904.py`.

## Roadmap status

| Roadmap area | Status | Review |
|---|---|---|
| Phase 0: project contract | Substantially complete | The contract defines the problem, target, users, metrics, leakage rules, architecture, limitations, and success criteria. Prediction frequency is not explicit. |
| Phase 1: understanding and ingestion | Partially complete | Dataset reading, column definitions, data dictionary, Parquet upload, run ID, and metadata exist. The notebook does not perform the validation checks that the detailed summary attributes to it, and Raw is a parsed Parquet representation rather than an untouched source copy. This deviation is documented. No modular ingestion script or saved validation report exists. |
| Phase 2: Raw/Silver/Gold layers | Substantially complete as a notebook prototype | Live Raw, Silver, and Gold objects exist with lineage metadata and correct row counts. Silver sorting and duplicate handling work. Schema, type, range, and null enforcement are incomplete. Gold metadata replacement in Notebook 03 is unsafe when Notebook 03 is rerun after Notebook 05. |
| Phase 3: EDA | Substantially complete | Distribution, outlier, trajectory, correlation, early/late-life, train/test, and RUL analyses run. One intended training sensor-to-RUL heatmap omits RUL from its correlation column list. Conclusions are plausible but are mostly qualitative rather than backed by saved tables or statistical comparisons. |
| Phase 4: feature engineering | Calculation and export complete | Train/test logic matches, full-history rows are complete, live exports match recomputation, and leakage probes pass. The incomplete-window policy remains to be applied for modeling. |
| Phase 5: baseline models | Not started | No mean, linear, random-forest, or XGBoost baseline exists yet. No engine-level split or evaluation report exists. |
| Phases 6–13 | Not started | MLflow, final training, explainability, FastAPI, Streamlit, Airflow, monitoring, tests, CI/CD, Docker Compose, and final documentation remain future work. |

## Notebook verification

### 01 — Dataset understanding and Raw ingestion

Verified behavior:

- Loads FD001 train `(20,631, 26)`, test `(13,096, 26)`, and RUL `(100, 2)`.
- Finds 100 training engines and 100 test engines.
- Local source values contain no null or infinite values.
- Assigns the expected 26 column names and creates test-engine IDs for the RUL table.
- Writes run-scoped Raw object keys and a manifest with three records.

Issues:

- The notebook contains no actual missing-value, duplicate, cycle-order, engine-order, RUL-consistency, impossible-value, or schema assertions. Those checks live in Notebook 02. Section 6 of the detailed progress summary therefore overstates what Notebook 01 itself includes.
- Local source paths are absolute Windows paths, which prevents portable execution.
- Raw storage is parsed and typed Parquet. The original roadmap called for untouched Raw source files. The project has consciously accepted this change, but it should remain recorded as an architecture decision.
- The three objects are uploaded before the manifest is written. A mid-cell failure can leave unreferenced partial objects. This does not corrupt a completed manifest, but cleanup is not implemented.

### 02 — Raw to Silver

Verified behavior:

- Selects the newest run containing successful Raw records for train, test, and RUL.
- Reads exact object keys from metadata.
- Removes exact duplicates, rejects conflicting engine-cycle duplicates, sorts train and test, verifies complete cycle sequences, checks RUL cardinality/order, and rejects negative RUL.
- Writes the three Silver tables with processing metadata.

Validation probes against deliberately damaged input produced these results:

| Damage introduced | Result |
|---|---|
| Exact duplicate row | Accepted after removing the duplicate, as intended |
| Conflicting duplicate engine-cycle | Rejected |
| Missing engine cycle | Rejected |
| Negative RUL | Rejected |
| Sensor null | Accepted |
| Sensor infinity | Accepted |
| Negative engine ID | Accepted |
| Missing sensor column | Accepted |

The null cells print counts but do not fail validation. The detailed progress summary currently says the notebook validates missing values; it should say it reports them. There is no explicit expected-schema check, numeric/finite-value check, valid-ID check, or general impossible-value policy.

### 03 — Silver to Gold

Verified behavior:

- Calculates `remaining_useful_life = last_cycle - time_in_cycles` correctly.
- Produces `(20,631, 27)` with terminal RUL equal to zero for every training engine.
- Its target matches Notebook 05 exactly.

Important issue:

Notebook 03 deletes every Gold metadata row for the selected run before adding `rul_training_features`. An in-memory rerun after Notebook 05 changed the Gold metadata list from:

```text
rul_training_features, train_features, test_features
```

to:

```text
rul_training_features
```

The feature objects remain in storage, but their manifest records disappear. Notebook 03 should replace only its own `rul_training_features` metadata record, just as Notebook 05 replaces only its own two records.

The final Markdown note in this notebook is stale: the implemented object key already includes `run_id`, so rerunning the same source run replaces the exact object, while different runs are preserved.

### 04 — EDA and sensor analysis

Verified behavior:

- Executes all analysis and plotting cells successfully.
- Correctly converts terminal test labels into per-cycle descriptive RUL. The final observed RUL for every engine equals the supplied label.
- Preserves source columns; its temporary selection lists do not remove columns from the DataFrames.

Issues:

- The intended training correlation heatmap does not include `remaining_useful_life`. The target is calculated in `temp_train_df`, but `temp` contains only selected sensors plus `time_in_cycles`. Consequently, the heatmap titled `Sensor correlation analysis` cannot support a training sensor-to-RUL conclusion.
- Early-life selection uses the global minimum cycle. This works for FD001 because every engine starts at cycle 1, but an engine-relative expression is clearer and reusable.
- The notebook is exploratory and verbose. It does not save figures or summary tables into `reports/`, so its conclusions are difficult to reproduce outside the notebook.

### 05 — Earlier feature-engineering notebook

Verified behavior:

- The training feature result matches the final notebook's training inputs exactly.

Issue:

- It does not engineer test features; its test output remains `(13,096, 26)`. It is correctly described as an earlier non-final version, but its similar name makes accidental use possible. The final notebook should be treated as the only active Phase 4 artifact.

### 05 — Final feature-engineering notebook

Verified behavior:

- Produces training shape `(20,631, 195)` and test shape `(13,096, 194)`.
- Produces 194 identical input columns in train and test; training has only the additional RUL target.
- Has unique column names, 100 engines per table, and no duplicate engine-cycle keys.
- Correctly computes rolling mean, standard deviation, minimum, maximum, one-cycle difference, percentage change, 20-cycle slope, and EMA for all 21 sensors.
- Sorts inside the preparation function.
- Preserves prefix results when later observations are added or modified, which confirms no future-cycle leakage.
- Produces identical features for an engine whether it is processed alone or beside another engine, which confirms engine isolation.
- Recovers a slope of exactly 3 for a synthetic linear sensor.
- Converts a percentage change after a zero denominator to `NaN` rather than infinity.
- Contains no infinite values.
- The approved metadata filter is present and preserves Notebook 03's metadata record.
- Live feature objects match recomputation exactly.

Incomplete-window result:

- Exactly 1,900 rows in each table have null rolling/slope values: the first 19 cycles of each of 100 engines.
- Difference and percentage change are null on the first cycle of each engine.
- Every row at cycle 20 or later is completely non-null.
- This leaves 18,731 complete-window training rows and 11,196 complete-window test rows.
- The shortest test history is 31 cycles, so every test engine has at least one valid 20-cycle row.

For Phase 5, filter to `time_in_cycles >= 20` before splitting and modeling. This follows the roadmap's strict 20-cycle representation and avoids imputing values that mean “history does not exist.” Keep the full exported Gold tables as the auditable feature layer; apply the filter in the modeling pipeline.

## Cross-project issues to resolve before calling the pipeline production-oriented

1. **Core logic is still notebook-only.** The original roadmap explicitly called for modular code under `src/` and notebooks only for exploration and validation. Current `src/` subdirectories contain no code.
2. **There are no automated tests.** The verification in this review proves the current computations, but it is not yet part of the project's normal development workflow.
3. **Repository scaffolding is incomplete.** No `README.md`, `pyproject.toml`, `.env.example`, `.gitignore`, `docker-compose.yml`, or `Makefile` exists, and the directory is not a Git repository.
4. **The implemented stack is narrower than the roadmap.** Current transformations use pandas, not PySpark. PostgreSQL and Docker Compose have not been introduced. This is acceptable for the current dataset size, but it is a roadmap deviation that should be deliberate.
5. **Metadata update is read-modify-write without concurrency control.** Parallel stages could overwrite each other's manifest changes. This is acceptable during manual single-user notebook development and must be redesigned before orchestration.
6. **Dependency management is phase-specific.** `requirements-phase1.txt` covers the current data work but does not yet declare the Phase 5 modeling stack such as scikit-learn and XGBoost.
7. **The concise progress summary contains old notebook names and obsolete object-key behavior.** The detailed summary is the better current reference, but it also needs the corrections identified above.

## Recommended immediate sequence

1. Fix Notebook 03's metadata filter and update its stale object-key note.
2. Fix the training RUL correlation selection in Notebook 04.
3. Add fail-fast Silver schema, null, finite-value, and ID validations.
4. Update the detailed progress summary to distinguish reported checks from enforced checks and to record that live feature export/read-back verification is now complete.
5. Begin Phase 5 with a reproducible engine-level train/validation split and rows at cycle 20 or later.
6. Establish the four roadmap baselines: training-target mean, Linear Regression, Random Forest, and XGBoost.
7. Evaluate MAE, RMSE, R², engine-level error, RUL-range error, near-failure error, missed critical states, and false alarms. Evaluate the held-out C-MAPSS test labels at each engine's final observed cycle.
8. After the baseline logic is understood, move reusable ingestion, validation, feature, and modeling code into `src/` and cover it with focused tests. That transition is required to fulfill the original project goal of an MLOps platform rather than a notebook collection.

## Current conclusion

Phase 4's feature calculation and MinIO export are complete and verified. Phase 4's production hardening is not complete because feature logic remains inside a notebook and has no maintained automated tests. For the learning sequence, the correct next stage is baseline modeling after the three targeted notebook fixes and the complete-window policy described above.
