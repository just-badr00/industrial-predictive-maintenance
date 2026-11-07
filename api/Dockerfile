# Start from an existing image that contains Linux and Python 3.14.
FROM python:3.14-slim

# Use /app as the working directory inside the image/container.
WORKDIR /app

# Copy the package list from the project into /app inside the image.
COPY api/requirements.txt /app/requirements.txt

# Install these packages into the image's Python environment.
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy the project's Python source code into /app/src inside the image.
COPY src /app/src

# At startup, load the exported folder that we will mount at /model.
ENV MODEL_DIRECTORY=/model

# Start the API when a container runs, accepting connections on its port 8000.
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
