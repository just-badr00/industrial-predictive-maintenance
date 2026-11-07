"""Copy a registered model and its metadata into a portable serving folder."""

import json
import shutil
from pathlib import Path

import mlflow
from mlflow import MlflowClient


def export_registered_model():
    # 1. Configure MLflow, using the same setup as model_loader.py.
    project_root = Path(
        r"C:\Users\Badr\Desktop\Python_3.0"
    )
    mlflow_database = (
        project_root / "mlflow.db"
    ).resolve().as_posix()

    if not Path(mlflow_database).is_file():
        raise FileNotFoundError(
            f"Tracking database not found: {mlflow_database}"
        )

    tracking_uri = f"sqlite:///{mlflow_database}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    client = MlflowClient()

    # 2. Find the registered model and its source run.
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

    # 3. Locate the saved model files and feature metadata.
    model_uri = (
        f"models:/{registered_model_name}/{model_version}"
    )
    model_directory = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
    )
    feature_metadata_path = mlflow.artifacts.download_artifacts(
        run_id=source_run_id,
        artifact_path="metadata/final_training_features.json",
    )

    # 4. Create the export folder and copy the existing files into it.
    destination = project_root / "output" / "serving_model"
    if destination.exists():
        raise FileExistsError(f"Export folder already exists: {destination}")

    destination.mkdir(parents=True)
    shutil.copytree(model_directory, destination / "model")
    shutil.copy2(feature_metadata_path, destination / "final_training_features.json")

    # 5. Write a JSON file identifying the exported model.
    model_info = {
        "model_name": registered_model_name,
        "model_version": str(model_version_details.version),
        "source_run_id": source_run_id,
        "gold_run_id": pipeline_run_id,
    }
    with open(destination / "model_info.json", "w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=2)

    print(f"Exported model version {model_version_details.version} to: {destination}")


if __name__ == "__main__":
    export_registered_model()
