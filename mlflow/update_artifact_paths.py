from pathlib import Path
import sqlite3

project_root = Path(r"C:\Users\Badr\Desktop\Python_3.0")
docker_database = project_root / "output" / "docker_mlflow" / "mlflow.db"

old_prefix = "file:c:/Users/Badr/Desktop/Python_3.0/"
new_prefix = "file:///opt/cmapss/"

# These database columns contain artifact locations.
locations = [
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
    ("model_versions", "source"),
    ("model_versions", "storage_location"),
]

# Open the existing Docker copy; fail if it is missing.
connection = sqlite3.connect(
    docker_database.as_uri() + "?mode=rw",
    uri=True,
)

# Commit all changes together, or roll them back if an update fails.
try:
    with connection:
        for table, column in locations:
            connection.execute(
                f"UPDATE {table} "
                f"SET {column} = REPLACE({column}, ?, ?)",
                (old_prefix, new_prefix),
            )
finally:
    connection.close()

print("Artifact addresses updated in the Docker database.")