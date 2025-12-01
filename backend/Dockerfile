# Multi-arch Python image; works on Raspberry Pi (arm/v7) and arm64
FROM python:3.11-slim

WORKDIR /app

# Install minimal build deps (for any packages that may need compilation)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Use a single worker for resource constrained devices like Raspberry Pi
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
