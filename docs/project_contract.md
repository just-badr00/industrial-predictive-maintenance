# Industrial Predictive Maintenance MLOps Platform

## Phase 0 - Project Specification Contract

**Status:** Approved initial scope  
**Version:** 1.0  
**Dataset:** NASA C-MAPSS FD001

### 1. Business problem

Unexpected machine failure can create safety risks, production downtime, and expensive emergency maintenance. This project will build a predictive-maintenance platform that uses historical engine sensor data to estimate when an engine may fail, allowing maintenance teams to plan interventions earlier.

### 2. Project objective

Build an end-to-end, reproducible MLOps system that predicts the **Remaining Useful Life (RUL)** of a simulated aircraft engine. RUL is the estimated number of operating cycles remaining before failure.

The initial version will focus on understanding the data, creating leakage-safe features, training a baseline model, and making predictions available through a simple operational workflow.

### 3. Dataset and scope

The first release will use **FD001** only because it contains one operating condition and one main degradation mode. Each row represents one engine during one operating cycle.

- Training data: engine histories that continue until failure.
- Test data: engine histories that stop before failure.
- RUL files: the true remaining cycles for the final observed row of each test engine.
- Inputs: engine ID, cycle, three operational settings, and 21 sensor measurements.

The other datasets (FD002, FD003, and FD004) are future extensions after the FD001 pipeline is understood and validated.

### 4. Prediction contract

**Input:** the recent operating history of one engine, initially represented by its cycle, operational settings, and sensor measurements. Later versions may use a rolling window of the most recent 20 cycles.

**Output:**

```text
predicted_rul       estimated remaining cycles
health_status       healthy, warning, or critical
model_version       version of the model used
```

Health-status rules:

- Healthy: RUL > 30 cycles
- Warning: 10 < RUL <= 30 cycles
- Critical: RUL <= 10 cycles

### 5. Intended users

The intended users are maintenance engineers, reliability engineers, fleet operators, and operations managers. The system must communicate both the numerical prediction and an understandable maintenance status.

### 6. Evaluation criteria

Model performance will be assessed using MAE, RMSE, R², error by RUL range, and error near failure. We will also report early-warning performance, missed critical engines, false alarms, and warning lead time. Late predictions will be treated as more serious than slightly early predictions because failing to warn before a failure is operationally costly.

Data must be split by engine ID. Rows from the same engine must not be randomly divided between training and validation/test sets. Scalers, feature transformations, and other learned preprocessing steps must be fitted only on the training engines.

### 7. Planned system architecture

```text
C-MAPSS files
      -> Raw storage
      -> Schema and data-quality validation
      -> Clean Silver data
      -> RUL and time-series features
      -> Model training and evaluation
      -> MLflow tracking and model registry
      -> FastAPI prediction service
      -> Streamlit monitoring dashboard
      -> Airflow orchestration
```

### 8. Initial deliverables

The project will produce a documented FD001 ingestion process, raw and processed datasets, data-quality checks, leakage-safe feature generation, baseline models, evaluation reports, an MLflow-tracked model, a prediction API, a monitoring dashboard, automated workflow tasks, tests, and setup documentation.

### 9. Limitations and exclusions

C-MAPSS is simulated benchmark data and is not real aircraft-maintenance data. The resulting model is for education and portfolio demonstration only and must not be used for real safety decisions. The initial scope excludes Kubernetes, Kafka, cloud deployment, real-time streaming, deep learning, and production certification. These may be considered as later extensions.

### Definition of success

The initial project is successful when a new FD001 engine history can be validated, transformed into features without data leakage, scored by a reproducibly trained model, and presented with a RUL estimate and health status through the planned platform components.
