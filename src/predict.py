"""Load the trained pipeline and make predictions.

This module is the single inference entry point: the CLI below, the unit tests
and the FastAPI service (Step 6) all call `predict()`.

Usage (run from the project root):
    python -m src.predict --input samples/patients.json
    python -m src.predict --input samples/patients.json --source registry
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from src.data import PROJECT_ROOT
from src.features import FEATURES

logger = logging.getLogger(__name__)

# Overridable by environment variables, which is how Docker/Kubernetes will configure it.
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "model.joblib")))
THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
LABELS = {0: "No disease", 1: "Disease"}

Record = dict[str, Any]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def load_model(path: str = str(MODEL_PATH)) -> Pipeline:
    """Load the saved pipeline once and cache it."""
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run `python -m src.train` first.")

    meta_path = model_path.with_name("model_meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        trained_with = meta.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            warnings.warn(
                f"Model trained with scikit-learn {trained_with} but {sklearn.__version__} "
                "is installed; predictions may differ. Reinstall from requirements.txt.",
                stacklevel=2,
            )
    logger.info("Loaded model from %s", model_path)
    return joblib.load(model_path)


def load_from_registry(alias: str = "champion") -> Pipeline:
    """Load the model currently holding `alias` in the MLflow Model Registry."""
    import mlflow.sklearn  # imported lazily: only this path needs MLflow

    from src.train import REGISTERED_MODEL_NAME, TRACKING_URI

    mlflow.set_tracking_uri(TRACKING_URI)
    uri = f"models:/{REGISTERED_MODEL_NAME}@{alias}"
    logger.info("Loading model from registry: %s", uri)
    return mlflow.sklearn.load_model(uri)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def to_frame(records: Record | list[Record]) -> pd.DataFrame:
    """Turn one record or a list of records into a model-ready DataFrame.

    Unknown fields raise an error (they usually mean a typo). Missing fields become
    NaN and are filled by the pipeline's imputers.
    """
    if isinstance(records, dict):
        records = [records]
    if not records:
        raise ValueError("No records to predict")

    df = pd.DataFrame(records)
    unknown = set(df.columns) - set(FEATURES)
    if unknown:
        raise ValueError(f"Unknown fields: {sorted(unknown)}. Expected: {FEATURES}")

    df = df.reindex(columns=FEATURES)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df


def predict(
    records: Record | list[Record],
    model: Pipeline | None = None,
    threshold: float = THRESHOLD,
) -> list[dict[str, Any]]:
    """Return one result per record: class, label, disease probability, confidence."""
    model = model if model is not None else load_model()
    X = to_frame(records)
    probabilities = model.predict_proba(X)[:, 1]

    results = []
    for p in probabilities:
        p = float(p)
        cls = int(p >= threshold)
        results.append({
            "prediction": cls,
            "label": LABELS[cls],
            "probability_disease": round(p, 4),
            "confidence": round(max(p, 1 - p), 4),
        })
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Predict heart disease risk")
    parser.add_argument("--input", required=True, type=Path,
                        help="JSON file with one patient object or a list of them")
    parser.add_argument("--source", choices=["file", "registry"], default="file",
                        help="load models/model.joblib or the MLflow 'champion' model")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    records = json.loads(args.input.read_text())
    model = load_from_registry() if args.source == "registry" else load_model()
    results = predict(records, model=model, threshold=args.threshold)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
