# Industrial Predictive Maintenance MLOps Platform

## Detailed project history and current status

**Date of update:** 2026-09-04  
**Dataset:** NASA C-MAPSS FD001  
**Main objective:** Predict the Remaining Useful Life (RUL) of an industrial engine from its operating history and sensor measurements.

This document summarizes the project direction, implementation decisions, problems discussed, work completed, and remaining work from the beginning of the project until the current feature-engineering stage.

---

## 1. Project idea and business objective

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

## 2. Dataset and prediction contract

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

## 3. Planned project roadmap

The original roadmap was designed as a complete MLOps project, but implementation is being done one stage at a time.

### Phase 0 — Project contract

Define the business problem, data, target, users, metrics, architecture, limitations, and definition of success.

### Phase 1 — Dataset understanding and ingestion

Understand the source files, create a data dictionary, validate the data, and upload the parsed Raw representation to MinIO.

### Phase 2 — Data lake layers

Create Raw, Silver, and initial Gold datasets with validation, lineage, and reproducible object paths.

### Phase 3 — Exploratory analysis

Study sensor distributions, correlation, sensor trends, degradation behavior, RUL relationships, outliers, stable sensors, and operating-condition behavior.

### Phase 4 — Feature engineering

Create leakage-safe time-series features such as rolling statistics, differences, percentage changes, EMA, slopes, engine age, and volatility.

### Phase 5 — Baseline models

Start with a mean RUL predictor, Linear Regression, Random Forest, and XGBoost. Evaluate with MAE, RMSE, R², error by RUL range, and business-oriented metrics.

### Later phases

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

## 4. Project environment and MinIO setup

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

## 5. Phase 0 completion: project contract

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

## 6. Phase 1: dataset understanding and Raw ingestion

The current Phase 1 notebook is:

~~~text
notebooks/01_dataset_understanding_and_raw_ingestion.ipynb
~~~

### Source-file handling

The original C-MAPSS files are space-separated text files without usable column headers. The notebook:

- Reads the training file.
- Reads the test file.
- Reads the RUL file.
- Assigns meaningful column names.
- Converts the values to numeric types.
- Adds unit_number to the RUL data so each RUL value can be associated with its test engine.

### Validation and documentation

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

### Raw-layer design decision

We discussed whether Raw should be an exact byte-for-byte copy of the original text files.

The current Raw layer is not the original text file itself. Before upload, the notebook:

- Parses the text into DataFrames.
- Assigns column names.
- Converts values to numeric types.
- Adds unit_number to the RUL table.

This is acceptable for this educational project because the Raw layer is an immutable parsed representation of the source data. The important rule is that after upload, Raw is not modified by later processing.

The distinction is documented conceptually: Raw is not an untouched byte copy, but it is the first immutable structured representation used by the pipeline.

### Raw object versioning

The Raw upload was changed to use a generated run_id and a date-partitioned object path:

~~~text
raw/train_data/load_date=YYYY-MM-DD/run_id=<run_id>/train.parquet
raw/test_data/load_date=YYYY-MM-DD/run_id=<run_id>/test.parquet
raw/RUL_data/load_date=YYYY-MM-DD/run_id=<run_id>/RUL.parquet
~~~

This prevents repeated pipeline executions from being confused with one another and gives later stages a precise source version.

---

## 7. The rerun problem and the run_id solution

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

## 8. ETL metadata in MinIO

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

### Selecting the correct input

When Phase 2 reads Raw data, it:

1. Reads all metadata manifests.
2. Keeps successful Raw records.
3. Groups them by run_id.
4. Finds a run containing train, test, and RUL data.
5. Selects the latest valid run.
6. Reads the exact object_key recorded for that run.

The same principle is used later when reading Silver data.

This is why read_parquet_object is used after metadata selection: metadata has already identified one exact file. read_parquet_prefix is used when intentionally discovering and combining multiple Parquet files under a prefix.

### Manifest updates

When a later stage writes new datasets, it:

- Reads the existing manifest.
- Removes only metadata records that the current stage is replacing.
- Adds the new records.
- Writes the updated manifest back to the same manifest key.

This makes reruns idempotent at the metadata level.

---

## 9. Phase 2: Raw to Silver

The current Phase 2 notebook is:

~~~text
notebooks/02_raw_to_silver_layer.ipynb
~~~

Its responsibility is to read the selected Raw run from MinIO, validate it, create Silver DataFrames, and write Silver data back to MinIO.

### Silver behavior

Silver DataFrames are created as copies of the MinIO-loaded Raw DataFrames so that Raw remains unchanged.

Silver contains:

- The original standardized columns.
- processing_timestamp.
- data_quality_status.

### Validation and cleaning

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

### Silver object organization

Silver objects are written under run-scoped paths such as:

~~~text
silver/train_data/load_date=YYYY-MM-DD/run_id=<run_id>/train.parquet
silver/test_data/load_date=YYYY-MM-DD/run_id=<run_id>/test.parquet
silver/RUL_data/load_date=YYYY-MM-DD/run_id=<run_id>/RUL.parquet
~~~

The Silver records are added to the ETL metadata manifest.

---

## 10. Phase 3: Silver to Gold

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

### Decision about Notebook 03 and Notebook 05

We discussed merging the Silver-to-Gold and feature-engineering notebooks. The current decision is to keep Notebook 03 independent for now.

Notebook 05 reads directly from Silver, performs feature engineering, calculates the training target, and writes the final feature tables. This creates some duplicated RUL-calculation logic with Notebook 03, but it preserves the existing notebook structure and avoids changing Notebook 03 at this stage.

Later, the architecture can be improved by making Notebook 03 the creator of a reusable Gold base dataset and making Notebook 05 consume that base dataset. That is a future refactoring, not part of the current change.

---

## 11. Phase 3: exploratory analysis

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

### Main EDA observations

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

### Correct RUL interpretation in EDA

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

## 12. Phase 4: feature-engineering concepts established

Before implementing features, the following concepts were clarified.

### Current sensor value

The current sensor value is the original sensor measurement at the current engine cycle. It is retained as a feature, for example:

~~~text
current_sensor_measurement_7
~~~

### A 20-cycle window

A row at cycle 30 uses the trailing history ending at cycle 30:

~~~text
cycles 11 through 30
~~~

It does not use cycles 31 or later.

For strict 20-cycle rolling features, the first 19 rows of each engine do not have enough history and therefore contain NaN.

### Time-series features

Time-series features describe how a sensor changes over time rather than looking only at one isolated measurement.

### Sensor slope

Sensor slope is numeric, not a text value such as increasing or decreasing.

It is the slope of a fitted straight line through the sensor values in the recent window:

~~~text
slope > 0  → sensor is generally increasing
slope < 0  → sensor is generally decreasing
slope ≈ 0  → sensor is relatively stable
~~~

Unlike a one-cycle difference, slope describes the overall direction across the window and is less sensitive to one individual jump.

### Recent volatility

Recent volatility means how much a sensor has fluctuated recently. Rolling standard deviation is being used as the volatility measure.

### EMA

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

### Wide feature table decision

The idea of storing each sensor as a separate table was considered. The current approach keeps one wide table with:

- Original/current sensor values.
- Derived features for each sensor.
- Engine and cycle identifiers.
- The training target where appropriate.

This is simpler for baseline model training because one row still represents one engine-cycle observation and all required model columns are together.

---

## 13. Phase 4 implementation in Notebook 05

The current final notebook is:

~~~text
notebooks/05_Feature engineering_Final.ipynb
~~~

It reads the selected Silver run through ETL metadata and exact object keys.

### Input preparation function

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

### Feature function

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

### Applying the function to train and test

The same function is called for both datasets:

~~~python
train_df_minio = add_time_series_features(train_df_minio, sensors)
test_df_minio = add_time_series_features(test_df_minio, sensors)
~~~

The same sensor list and parameters are used for both, preventing train/test feature mismatch.

### Percentage-change safety

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

### Training RUL target

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

## 14. Exporting engineered features to MinIO

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

## 15. Current notebook inventory

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

## 16. Important issues identified and their current resolution

### Rerunning notebooks can duplicate or confuse data

Resolved conceptually through run-scoped object keys and ETL metadata. Final execution and read-back verification are still required.

### Raw is a parsed representation rather than a byte-for-byte source copy

Accepted and documented. The current Raw layer is an immutable structured representation of the original files.

### Silver initially validated more than it transformed

Improved by adding duplicate removal, conflicting-duplicate rejection, sorting, and index resetting while preserving Raw.

### Validation coverage was stronger for test than train

Reviewed and strengthened so train and test receive appropriate structural validation.

### Gold contained Silver metadata columns

Corrected by removing processing_timestamp and data_quality_status before Gold and EDA use.

### Notebook names were unclear

The active notebook names now communicate their responsibilities more clearly for the first three stages. The feature-engineering final notebook was created separately.

### Portability and Git

These remain intentionally deferred until the notebook work is finished. The workspace was previously found without an initialized Git repository.

### Gold metadata replacement scope

The replacement must be limited to train_features and test_features. Removing every Gold record would unnecessarily remove Notebook 03’s independent Gold dataset.

---

## 17. Current completion status

### Completed or substantially completed

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

### Still pending or requiring verification

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

## 18. Recommended next sequence

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

