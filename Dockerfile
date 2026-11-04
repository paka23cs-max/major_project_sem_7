
# ==========================================
# STAGE 1: BUILDER
# Install dependencies and prepare environment
# ==========================================

FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Create isolated Python environment
RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and wheel
RUN pip install --no-cache-dir --upgrade pip wheel

# Install CPU-only PyTorch and Torchvision
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining production dependencies
RUN pip install --no-cache-dir -r requirements.txt


# ==========================================
# STAGE 2: RUNTIME
# Minimal production environment
# ==========================================

FROM python:3.11-slim AS runtime

WORKDIR /app

# Create non-root user for security
ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} appgroup \
    && useradd -m -u ${UID} -g appgroup -s /bin/bash appuser

# Install runtime system dependencies
# libgl1 replaces the unavailable libgl1-mesa-glx
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed Python environment from builder
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# Copy application files
COPY --chown=appuser:appgroup config.py .
COPY --chown=appuser:appgroup settings.py .
COPY --chown=appuser:appgroup models/ ./models/
COPY --chown=appuser:appgroup data/ ./data/
COPY --chown=appuser:appgroup evaluation/ ./evaluation/
COPY --chown=appuser:appgroup utils/ ./utils/
COPY --chown=appuser:appgroup app.py .

# Create required directories
RUN mkdir -p outputs static temp logs \
    && chown -R appuser:appgroup /app

# Copy trained model and output files
COPY --chown=appuser:appgroup outputs/ ./outputs/

# Run application as non-root user
USER appuser

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_WORKERS=1 \
    APP_LOG_FORMAT=json

# Expose application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start production application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


# ==========================================
# STAGE 3: DEVELOPMENT
# Additional development and testing tools
# ==========================================

FROM runtime AS development

USER root

# Install development system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install development Python dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    httpx \
    black \
    flake8 \
    mypy \
    onnx

# Switch back to non-root user
USER appuser

# Development server with automatic reload
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
