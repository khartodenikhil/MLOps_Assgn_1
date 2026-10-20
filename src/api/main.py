"""Heart disease prediction API.

Run locally (from the project root):
    uvicorn src.api.main:app --reload
Then open http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response

from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ModelInfoResponse,
    PatientFeatures,
    PredictionResponse,
)
from src.predict import MODEL_PATH, load_model, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("heart_api")

# Loaded once at startup and shared by all requests.
STATE: dict[str, Any] = {"model": None, "meta": {}}


def read_meta() -> dict:
    path = MODEL_PATH.with_name("model_meta.json")
    return json.loads(path.read_text()) if path.exists() else {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE["model"] = load_model(str(MODEL_PATH))
        STATE["meta"] = read_meta()
        logger.info("Model ready: %s", STATE["meta"].get("model_name", "unknown"))
    except FileNotFoundError:
        # Start anyway so /health can report the problem (and Kubernetes can see it).
        logger.exception("Model could not be loaded from %s", MODEL_PATH)
        STATE["model"], STATE["meta"] = None, {}
    yield
    STATE["model"], STATE["meta"] = None, {}


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts heart disease risk from 13 clinical features "
                "(UCI Heart Disease, Cleveland). For coursework, not medical use.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_model():
    model = STATE.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return model


def model_name() -> str:
    return STATE.get("meta", {}).get("model_name", "unknown")


# --------------------------------------------------------------------------- #
# Service endpoints
# --------------------------------------------------------------------------- #
@app.get("/", tags=["service"])
def root() -> dict:
    return {
        "service": app.title,
        "version": app.version,
        "docs": "/docs",
        "endpoints": ["/health", "/model/info", "/predict", "/predict/batch"],
    }


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(response: Response) -> HealthResponse:
    """200 when the model is loaded, 503 otherwise (used by Docker and Kubernetes probes)."""
    loaded = STATE.get("model") is not None
    if not loaded:
        response.status_code = 503
    return HealthResponse(
        status="ok" if loaded else "unavailable",
        model_loaded=loaded,
        model_name=model_name() if loaded else None,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["service"])
def model_info() -> ModelInfoResponse:
    get_model()
    meta = STATE["meta"]
    return ModelInfoResponse(**{k: meta.get(k) for k in ModelInfoResponse.model_fields})


# --------------------------------------------------------------------------- #
# Prediction endpoints
# --------------------------------------------------------------------------- #
@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict_one(patient: PatientFeatures) -> PredictionResponse:
    """Predict for a single patient."""
    result = predict(patient.model_dump(), model=get_model())[0]
    logger.info("prediction=%s probability=%.3f", result["prediction"],
                result["probability_disease"])
    return PredictionResponse(**result, model_name=model_name())


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(request: BatchRequest) -> BatchResponse:
    """Predict for up to 100 patients in one call."""
    records = [p.model_dump() for p in request.patients]
    results = predict(records, model=get_model())
    name = model_name()
    return BatchResponse(
        count=len(results),
        predictions=[PredictionResponse(**r, model_name=name) for r in results],
    )
