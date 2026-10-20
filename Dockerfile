# syntax=docker/dockerfile:1

# ---------- Stage 1: install dependencies into a virtual environment ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements-serve.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements-serve.txt

# ---------- Stage 2: slim runtime image ----------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="heart-disease-api" \
      org.opencontainers.image.description="Heart disease risk prediction API (FastAPI + scikit-learn)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/models/model.joblib \
    PREDICTION_THRESHOLD=0.5

# Run as an unprivileged user rather than root.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY models/model.joblib models/model_meta.json ./models/

USER appuser
EXPOSE 8000

# Docker marks the container unhealthy if /health stops returning 200.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
