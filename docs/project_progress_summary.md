# Predictive Maintenance Project — Progress Summary

## 1. Project foundation

The project is an industrial predictive-maintenance platform based on the NASA C-MAPSS FD001 dataset.

The dataset contains:

- Training data: engine histories until failure
- Test data: engine histories stopping before failure
- RUL data: one remaining-useful-life value for each test engine
- 26 original columns:
  - `unit_number`
  - `time_in_cycles`
  - 3 operational settings
  - 21 sensor measurements

The project roadmap and initial ideas were established using the “Recall machine learning ideas chat”.

The one-page project contract is located at:

`docs/project_contract.md`

## 2. Project environment

A project-specific virtual environment was created at:

`C:\Users\Badr\Desktop\Python_3.0\.venv`

One environment is used for the whole project rather than creating a separate environment for every phase.

The selected Jupyter kernel is:

`Python 3.14 - Predictive Maintenance (.venv)`

The main installed requirements include:

- pandas
- numpy
- matplotlib
- seaborn
- jupyter
- pytest
- python-dotenv
- boto3
- pyarrow
- fastparquet

`fastparquet` was added because PyArrow’s native DLLs were blocked by Windows Application Control. Parquet reading and writing therefore uses:

~~~python
engine="fastparquet"
~~~

## 3. Phase 1 — Dataset understanding

Notebook:

`notebooks/01_dataset_understanding.ipynb`

The notebook contains:

- Source-file loading for train, test, and RUL files
- Column-name creation for the C-MAPSS files
- Test-engine ID creation for the RUL table
- Preserved `.head()` cells
- Numeric type conversion
- The project data dictionary
- MinIO Raw-layer upload code

The following checks were completed during Phase 1:

- Missing-value behavior
- Duplicate behavior
- Engine order between test and RUL data
- Cycle ordering
- RUL consistency
- Numeric data types

The arbitrary RUL minimum/maximum check was removed because it was not useful for the current stage.

### Data dictionary

The data dictionary explains each column’s:

- Meaning
- Unit
- Data type
- Category
- Modeling role
- Notes

It distinguishes:

- Identifier: `unit_number`
- Time variable: `time_in_cycles`
- Input features: operational settings and sensor measurements
- Target: `remaining_useful_life`

The 21 sensor meanings and engineering units are based on the C-MAPSS documentation.

### Engine-level summary

An engine-level summary was discussed as a diagnostic table with one row per engine. It can describe:

- Engine lifetime
- Final cycle
- Number of observations
- Possible degradation behavior

It is useful for understanding differences between engines but is not automatically a model input.

## 4. MinIO data lake setup

MinIO is used as the project’s object-storage data lake.

The main components are:

- Docker container: runs the MinIO service
- Docker volume: stores MinIO data persistently
- Host ports: provide access from the computer
- `.env`: stores project configuration and credentials

The project-specific MinIO volume is:

`minio_cmapss_data`

The project container is:

`minio-cmapss`

Another MinIO container from a different project uses ports `9000` and `9001`. Therefore, this project uses:

- API: `http://localhost:9010`
- Console: `http://localhost:9011`

The project `.env` uses:

~~~env
MINIO_ENDPOINT=http://localhost:9010
~~~

Credential values are intentionally not recorded in this document.

### MinIO credential behavior

MinIO stores its root credentials inside the data volume during initialization. Reusing a volume also reuses the credentials associated with that volume.

Stopping or deleting a container does not delete its volume or the data stored inside it.

### Port conflict

The initial attempt to create the project container on ports `9000` and `9001` failed because those ports were already used by the other MinIO container.

The project container was then mapped to ports `9010` and `9011` while keeping MinIO’s internal ports at `9000` and `9001`.

The first login issue happened because the project endpoint still pointed to `localhost:9000`, which accessed the other project’s MinIO instance. The endpoint was corrected to `localhost:9010`.

## 5. Reading Parquet data from MinIO

The project uses a `read_parquet_prefix` function.

The function:

1. Lists objects under a MinIO prefix.
2. Keeps only `.parquet` objects.
3. Downloads each object into memory.
4. Reads each object with pandas.
5. Combines the DataFrames.

MinIO prefixes behave like logical folders, so a prefix such as `raw/train_data/` cannot be opened directly as one file. The objects under the prefix must be listed and read individually.

The verified dataset sizes were:

- Training: `(20631, 26)`
- Test: `(13096, 26)`
- RUL: `(100, 2)`

## 6. Phase 2 — Data lake layers

Notebook:

`notebooks/02_data_lake_layers.ipynb`

The final workflow is:

1. Phase 1 reads and validates the local source files.
2. Phase 1 writes unchanged data to the MinIO Raw layer.
3. Phase 2 reads the Raw layer from MinIO.
4. Phase 2 validates the MinIO-loaded data.
5. Phase 2 creates Silver DataFrames.
6. Phase 2 writes the Silver datasets to MinIO.

Phase 2 contains:

- MinIO connection setup
- `read_parquet_prefix`
- Raw-layer reads from MinIO
- Column-name normalization
- Validation checks copied from Phase 1
- Silver DataFrame creation
- Processing timestamp
- Data-quality status
- Silver-layer upload

The original validation code uses these aliases:

~~~python
raw_df_train = train_df_minio
raw_df_test = test_df_minio
raw_df_RUL = rul_df_minio
~~~

This allows the original validation code to operate on the MinIO DataFrames without rewriting the copied validation cells.

Numeric type conversion was not repeated because Parquet preserves the already-validated types.

Sorting and duplicate removal were not repeated because Phase 1 established that the data was already correctly ordered and contained no duplicates.

## 7. Silver layer

Silver DataFrames are created as copies of the MinIO-loaded Raw DataFrames:

~~~python
silver_train_df = train_df_minio.copy()
silver_test_df = test_df_minio.copy()
silver_rul_df = rul_df_minio.copy()
~~~

The original Raw DataFrames are not modified.

The Silver DataFrames preserve the original columns and add:

- `processing_timestamp`
- `data_quality_status`

The current status value is:

`passed_phase_1_validation`

The Silver datasets are written under:

~~~text
silver/train_data/
silver/test_data/
silver/RUL_data/
~~~

## 8. Phase 3 — Exploratory analysis

Notebook:

`notebooks/04_eda_and_sensor_analysis.ipynb`

The EDA notebook contains analysis of:

- Sensor distributions
- Constant sensors
- Sensor correlations
- Sensor trends over engine life
- Early-life versus near-failure behavior
- Training/test similarities
- Engine-level behavior
- Potential degradation patterns
- RUL prediction difficulty

The main observations were:

Sensors that generally increase over engine lifetime:

- `sensor_measurement_2`
- `sensor_measurement_3`
- `sensor_measurement_4`
- `sensor_measurement_8`
- `sensor_measurement_11`
- `sensor_measurement_13`
- `sensor_measurement_15`
- `sensor_measurement_17`

Sensors that generally decrease:

- `sensor_measurement_7`
- `sensor_measurement_12`
- `sensor_measurement_20`
- `sensor_measurement_21`

`sensor_measurement_9` and `sensor_measurement_14` vary more from engine to engine.

Constant sensors appear non-informative in FD001, but no sensor was permanently removed.

Operational settings remain available because they may explain differences in sensor behavior.

Some apparent outliers may represent genuine degradation signals rather than simple noise. This requires further investigation.

RUL prediction is difficult because engines may enter the observed dataset at different degradation states. An engine with a long apparent operating history is not necessarily healthy at the beginning of the observed data.

## 9. Gold layer

Notebook:

`notebooks/03_gold_layer.ipynb`

The first Gold dataset was created:

~~~text
gold/rul_training_features/
~~~

Its grain is:

~~~text
one row = one engine at one operating cycle
~~~

The dataset is created from the Silver training data.

The RUL calculation is:

~~~text
remaining_useful_life =
engine_final_training_cycle - current_time_in_cycles
~~~

The intermediate `last_cycle` column is removed after the calculation.

The dataset is written using:

~~~text
gold/rul_training_features/load_date=YYYY-MM-DD/rul.parquet
~~~

The shortened filename avoids the previous object-key length problem.

### Planned Gold datasets

The following datasets are documented but not populated yet:

~~~text
gold/current_engine_health/
gold/maintenance_predictions/
gold/model_performance/
~~~

Their intended grains are:

- `current_engine_health`: one row per engine at its latest observed cycle
- `maintenance_predictions`: one row per engine prediction event
- `model_performance`: one row per metric from one model evaluation run

These datasets require later feature engineering, model training, prediction, and evaluation. Artificial or empty records should not be created before those processes exist.

### Gold object-key behavior

The current object key uses the short filename `rul.parquet`. If the Gold notebook is rerun on the same date, it will replace the object at that same key.

Preserving multiple Gold runs can be added later using a run identifier or timestamp in the filename.

## 10. Notebook organization

The current notebook organization is:

~~~text
01_dataset_understanding.ipynb
    Read and understand source files
    Validate the dataset
    Upload unchanged Raw data

02_data_lake_layers.ipynb
    Read Raw data from MinIO
    Validate it
    Build and upload Silver data

03_gold_layer.ipynb
    Read Silver data from MinIO
    Build rul_training_features
    Upload Gold data

04_eda_and_sensor_analysis.ipynb
    Explore sensors, RUL, trends, and degradation
~~~

## 11. Correction made during the project

An earlier notebook reorganization moved cells too aggressively and caused duplicated prerequisite code and removal of the intended MinIO read-back structure from Phase 2.

That was corrected by restoring the logical baseline:

- Phase 1 keeps source-file reading, dataset understanding, data dictionary, and Raw upload.
- Phase 2 reads Raw data from MinIO and builds Silver.
- Phase 3 Gold builds the first training-feature dataset.
- The EDA notebook remains separate.

The validation cells were then copied into Phase 2 and removed from Phase 1 as requested. The copied validation code was preserved without rewriting its logic.

## 12. Current status and remaining work

Completed:

- Project contract
- Project virtual environment
- Requirements setup
- Dataset loading and understanding
- Data dictionary
- Dataset validation
- MinIO setup
- Raw-layer ingestion
- MinIO Parquet reading
- Silver-layer creation
- First Gold dataset
- EDA notebook and conclusions

Remaining:

- Complete exploratory analysis where needed
- Create time-series and degradation features
- Establish a modeling baseline
- Train and evaluate an RUL model
- Create `current_engine_health`
- Create `maintenance_predictions`
- Create `model_performance`
- Add production-oriented orchestration and monitoring later

The next major technical stage is feature engineering and baseline modeling. The original columns should remain available while derived features are added as new columns.


