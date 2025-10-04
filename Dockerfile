# Use a GPU-compatible base image with necessary libraries
# This is a standard, robust image for PyTorch applications.
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Set environment variables and expose port 7860 (Hugging Face default)
ENV PYTHONUNBUFFERED 1
ENV APP_PORT 7860
EXPOSE 7860

# Install system dependencies needed for building Python wheels (if any)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# 1. Copy and Install Requirements (From Repository Root)
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# 2. Copy all application files (code and ML models)
# This copies your local 'my_app' folder (code, assets) to /app/my_app in the container
COPY my_app /app/my_app

# --- Set Entry Point ---
# The CMD runs Gunicorn, using the full Python import path for your application.
# Path: my_app (folder/package) -> app (subfolder/module) -> main (file) -> app (FastAPI instance)
CMD exec gunicorn my_app.app.main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$APP_PORT
