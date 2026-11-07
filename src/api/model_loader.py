from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import json
import shap


def load_exported_model(serving_directory):
    # This folder contains the three items created by export_model.py.
    serving_directory = Path(serving_directory)

    with open(serving_directory / "model_info.json", "r", encoding="utf-8") as file:
        model_info = json.load(file)

    with open(serving_directory / "final_training_features.json", "r", encoding="utf-8") as file:
        feature_metadata = json.load(file)

    # Load the saved model directly from the exported model folder.
    final_model = mlflow.sklearn.load_model(
        str(serving_directory / "model")
    )
    feature_columns = feature_metadata["feature_columns"]

    # Keep the same feature checks used when loading from the registry.
    if len(feature_columns) != final_model.n_features_in_:
        raise ValueError(
            "The saved feature count does not match the model."
        )

    if feature_columns != final_model.feature_names_in_.tolist():
        raise ValueError(
            "The saved feature names or order do not match the model."
        )

    shap_explainer = shap.TreeExplainer(
        final_model,
        feature_perturbation="tree_path_dependent",
    )

    # Return the same structure that our API already knows how to use.
    return {
        "model": final_model,
        "model_name": model_info["model_name"],
        "model_version": model_info["model_version"],
        "source_run_id": model_info["source_run_id"],
        "gold_run_id": model_info["gold_run_id"],
        "feature_columns": feature_columns,
        "explainer": shap_explainer,
    }


def load_registered_model():

    # ============================================================
    # 1. Configure the project and MLflow
    # ============================================================

    project_root = Path(
        r"C:\Users\Badr\Desktop\Python_3.0"
    )
    mlflow_database = (
        project_root / "mlflow.db"
    ).resolve().as_posix()

    tracking_uri = f"sqlite:///{mlflow_database}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = MlflowClient()


    # ============================================================
    # 2. Find and load the registered final model
    # ============================================================

    registered_model_name = (
        "cmapss_fd001_rul_random_forest"
    )

    model_version = 1

    model_version_details = client.get_model_version(
        name=registered_model_name,
        version=model_version,
    )

    source_run_id = model_version_details.run_id

    if source_run_id is None:
        raise ValueError(
            "The registered model version has no source run."
        )

    source_run = client.get_run(source_run_id)

    pipeline_run_id = source_run.data.tags.get(
        "gold_run_id"
    )

    if pipeline_run_id is None:
        raise ValueError(
            "The model's source run does not contain "
            "the Gold dataset run ID."
        )

    model_uri = (
        f"models:/{registered_model_name}/{model_version}"
    )

    final_model = mlflow.sklearn.load_model(
        model_uri
    )

    feature_metadata_path = mlflow.artifacts.download_artifacts(
    run_id=source_run_id,
    artifact_path="metadata/final_training_features.json",
)

    with open(feature_metadata_path, "r", encoding="utf-8") as file:
        feature_metadata = json.load(file)

    feature_columns = feature_metadata["feature_columns"]

    if len(feature_columns) != final_model.n_features_in_:
        raise ValueError(
            "The saved feature count does not match the model."
        )

    if feature_columns != final_model.feature_names_in_.tolist():
        raise ValueError(
            "The saved feature names or order do not match the model."
        )

    shap_explainer = shap.TreeExplainer(
        final_model,
        feature_perturbation="tree_path_dependent",
    )



    return {
        "model": final_model,
        "model_name": registered_model_name,
        "model_version": str(model_version_details.version),
        "source_run_id": model_version_details.run_id,
        "gold_run_id": pipeline_run_id,
        "feature_columns": feature_columns,
        "explainer": shap_explainer,
    }
