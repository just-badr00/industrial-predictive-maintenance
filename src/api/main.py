from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from src.api.model_loader import load_registered_model, load_exported_model
from src.api.schemas import PredictionRequest,BatchPredictionRequest, PredictionResponse
from src.transformations.feature_engineering import prepare_model_input,InvalidEngineHistoryError
from src.modeling.predict import health_status
from src.modeling.explain import explain_predictions
from src.logging_config import configure_logging
from uuid import uuid4
import time
import os
import pandas as pd


logger = configure_logging()

# Like `with open(...)`: manage preparation before use and cleanup afterward.
@asynccontextmanager
# FastAPI's lifespan mechanism requires an async context manager.
async def lifespan(app: FastAPI):
    # STARTUP: prepare the application before it starts handling requests.
    # An environment variable is a setting supplied before starting the API.
    serving_directory = os.getenv("MODEL_DIRECTORY")

    # Keep the loaded model dictionary on the app, shared across requests.
    # Load once per startup; a development reload starts the app again.
    if serving_directory is not None:
        if not serving_directory.strip():
            raise ValueError("MODEL_DIRECTORY must contain a folder path.")
        logger.info("Loading the exported model")
        app.state.model_data = load_exported_model(serving_directory)
    else:
        logger.info("Loading the registered model")
        app.state.model_data = load_registered_model()

    logger.info(
        "Model loaded successfully",
        extra={
            "details": {
                "model_name": app.state.model_data["model_name"],
                "model_version": app.state.model_data["model_version"],
            }
        },
    )
    # Pause here while the app handles requests; resume during shutdown.
    # This separates startup from shutdown; it does not send an HTTP response.
    yield

    # SHUTDOWN: runs when stopping the service, including a reload.
    # Release the app's reference; saved model files remain on disk.
    app.state.model_data = None
    logger.info("Application stopped")

# Give FastAPI the function so it can manage startup and shutdown.
app = FastAPI(lifespan=lifespan)
@app.middleware("http")
async def log_requests(request: Request, call_next):

    # request is our parameter’s name 
    # Request is the class describing the HTTP request object.

    # request holds information like : 
        #method → GET
        #URL    → http://127.0.0.1:8000/health
        #headers
        #access to the submitted body, if there is one

    #call_next : It is a function supplied by FastAPI that means : “Pass this request to the rest of the application for processing.”

    start_time = time.perf_counter()

    request_id = uuid4().hex

    model_data = getattr(app.state, "model_data", None)
    model_version = model_data["model_version"] if model_data else None

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.exception(
            "Request failed unexpectedly",
            extra={
                "details": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "duration_ms": round(duration_ms, 2),
                    "request_id": request_id,
                    "model_version": model_version,
                }
            },
        )

        raise

    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Request completed",
        extra={
            "details": {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "request_id": request_id,
                "model_version": model_version,
            }
        },
    )

    response.headers["X-Request-ID"] = request_id
    return response


def get_model_data():
    model_data = getattr(app.state, "model_data", None) #this means Retrieve app.state.model_data. If that attribute doesn’t exist, use None

    if model_data is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is unavailable.",
        )

    return model_data


@app.get("/health")
def health():
    model_data = get_model_data()
    return {
        "status": "ok",
        "message": "Predictive maintenance API is running",
        "model_loaded": True,
    }

@app.get("/model-info")
def model_info():
    
    model_data = get_model_data()
    return{
        "model_name": model_data["model_name"],
        "model_version": model_data["model_version"],
        "source_run_id": model_data["source_run_id"],
        "gold_run_id": model_data["gold_run_id"],
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    model_data = get_model_data()
    try:
        model_input = prepare_model_input(
            request,
            model_data["feature_columns"],
        )
    except InvalidEngineHistoryError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    predictions = model_data["model"].predict(model_input)
    predicted_rul = float(predictions[0])
    all_factors = explain_predictions(
        model_data["explainer"],
        model_input,
    )
    return {
        "engine_id": request.engine_id,
        "cycle": request.history[-1].time_in_cycles,
        "predicted_rul": predicted_rul,
        "status": health_status(predicted_rul),
        "top_factors": all_factors[0],
        "model_version": model_data["model_version"],
    }

@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(request: BatchPredictionRequest):
    model_data = get_model_data()

    all_model_input = pd.DataFrame()
    for element in request.engines:
        try:
            model_input = prepare_model_input(
                element,
                model_data["feature_columns"],
            )
            all_model_input = pd.concat([all_model_input,model_input],ignore_index=True)
        except InvalidEngineHistoryError as error:
            raise HTTPException(
                status_code=422,
                detail=str(error),
            ) from error

    predictions = model_data["model"].predict(all_model_input)
    all_factors = explain_predictions(
        model_data["explainer"],
        all_model_input,
    )
    responses = []

    for engine_request, prediction, factors  in zip(request.engines, predictions,all_factors):
        predicted_rul = float(prediction)

        responses.append({
            "engine_id": engine_request.engine_id,
            "cycle": engine_request.history[-1].time_in_cycles,
            "predicted_rul": predicted_rul,
            "status": health_status(predicted_rul),
            "top_factors": factors,
            "model_version": model_data["model_version"],
        })

    return responses
