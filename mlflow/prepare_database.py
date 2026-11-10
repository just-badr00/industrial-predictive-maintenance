from pathlib import Path
import sqlite3

project_root = Path(r"C:\Users\Badr\Desktop\Python_3.0")

source_database = project_root / "mlflow.db"
docker_directory = project_root / "output" / "docker_mlflow"
docker_database = docker_directory / "mlflow.db"

# Refuse to replace an existing Docker database.
if docker_database.exists():
    raise FileExistsError("The Docker database already exists.")

docker_directory.mkdir(parents=True, exist_ok=True)

# Open the original database with read-only access.
source_connection = sqlite3.connect(
    source_database.as_uri() + "?mode=ro",
    uri=True,
)

destination_connection = sqlite3.connect(docker_database)

# Copy the database using SQLite's backup operation.
source_connection.backup(destination_connection)

destination_connection.close()
source_connection.close()

print(f"Database copied to: {docker_database}")