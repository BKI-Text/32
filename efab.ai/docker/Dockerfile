# Multi-stage Docker build for Beverly Knits AI Supply Chain Planner
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-cov black flake8 mypy

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command for development
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/data/models /app/logs && \
    chown -R appuser:appuser /app/data /app/logs

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command for production
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# ML Model Training stage
FROM base as ml-training

# Install additional ML dependencies
RUN pip install \
    jupyter \
    mlflow \
    tensorboard \
    optuna

# Copy training scripts
COPY src/ src/
COPY train_*.py ./
COPY test_*.py ./

# Create model output directory
RUN mkdir -p /app/models/trained

# Default command for training
CMD ["python", "train_enhanced_ml.py"]

# ML Inference stage
FROM base as ml-inference

# Copy only necessary files for inference
COPY src/ src/
COPY api/ api/
COPY requirements.txt .

# Copy pre-trained models (if available)
COPY --from=ml-training /app/models/trained/ /app/models/trained/

# Create non-root user
RUN groupadd -r mluser && useradd -r -g mluser mluser
RUN mkdir -p /app/data/models /app/logs && \
    chown -R mluser:mluser /app/data /app/logs

# Switch to non-root user
USER mluser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command for ML inference
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]