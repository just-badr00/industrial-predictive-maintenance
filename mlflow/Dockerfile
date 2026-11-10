# Start with Linux and Python.
FROM python:3.14-slim

# Use /app as the working folder.
WORKDIR /app

# Copy the package list and install MLflow.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Start MLflow using the database copy we will prepare for Docker.
CMD ["python", "-m", "mlflow", "server", "--backend-store-uri", "sqlite:////opt/cmapss/output/docker_mlflow/mlflow.db", "--host", "0.0.0.0", "--port", "5000", "--allowed-hosts", "localhost:*,127.0.0.1:*,host.docker.internal:5001,mlflow:5000"]