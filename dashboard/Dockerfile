# Start with Linux and Python.
FROM python:3.14-slim

# Use /app as the working folder inside the container.
WORKDIR /app

# Copy the package list and install its packages.
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Copy the dashboard code.
COPY streamlit_app.py /app/dashboard/streamlit_app.py

# Tell the dashboard where we will make project files available.
ENV PROJECT_ROOT=/opt/cmapss

# Start Streamlit and accept connections from outside the container.
CMD ["python", "-m", "streamlit", "run", "dashboard/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]