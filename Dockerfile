# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.13-slim

# -----------------------------------------------------------------------------
# Stage 1: Build virtual environment, generate dataset, train & verify ML model
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies in a separate layer for fast rebuild caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy source and train/verify natively inside Linux so binary artifacts match runtime
COPY car_catalog.py generate_dataset.py train_model.py app.py test.py test_app.py ./
COPY static/ ./static/

RUN python generate_dataset.py \
    && python train_model.py \
    && python test.py \
    && python -m unittest test_app.py -v

# -----------------------------------------------------------------------------
# Stage 2: Minimal non-root production runtime image
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION} AS runtime

LABEL org.opencontainers.image.title="AutoValuate AI" \
      org.opencontainers.image.description="Second-Hand Car Price Prediction ML Web App"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=5000 \
    WEB_CONCURRENCY=2 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Run as an unprivileged system user
RUN groupadd --system appuser \
    && useradd --system --gid appuser --home-dir /app --shell /bin/false appuser

WORKDIR /app

# Copy isolated Python environment and runtime application artifacts only
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /build/car_catalog.py /build/generate_dataset.py /build/train_model.py /build/app.py ./
COPY --from=builder --chown=appuser:appuser /build/static ./static
COPY --from=builder --chown=appuser:appuser /build/data ./data
COPY --from=builder --chown=appuser:appuser /build/model ./model

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; port = os.environ.get('PORT', '5000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=3)"]

# Use exec so gunicorn runs as PID 1 and handles SIGTERM gracefully
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers ${WEB_CONCURRENCY:-2} --threads 4 --timeout 60 --access-logfile - --error-logfile - app:app"]
