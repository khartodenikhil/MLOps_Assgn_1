"""Request and response models.

Validation rules mirror src/data.py (VALID_RANGES / ALLOWED_VALUES), so the API
rejects the same impossible values the training data checks reject.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_PATIENT = {
    "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0,
    "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2,
    "ca": 3, "thal": 3,
}


class PatientFeatures(BaseModel):
    """The 13 clinical features. `ca` and `thal` are optional (imputed if missing)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_PATIENT]})

    age: int = Field(ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        description="Chest pain: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic")
    trestbps: float = Field(ge=50, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(
        description="Resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy")
    thalach: float = Field(ge=50, le=250, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(description="Exercise-induced angina")
    oldpeak: float = Field(ge=0, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(description="Peak exercise ST slope: 1 up, 2 flat, 3 down")
    ca: Literal[0, 1, 2, 3] | None = Field(
        default=None, description="Major vessels coloured by fluoroscopy (optional)")
    thal: Literal[3, 6, 7] | None = Field(
        default=None, description="Thalassemia: 3 normal, 6 fixed, 7 reversible (optional)")


class _Response(BaseModel):
    # Allow field names starting with "model_" (Pydantic reserves them by default).
    model_config = ConfigDict(protected_namespaces=())


class PredictionResponse(_Response):
    prediction: int = Field(description="1 = heart disease predicted, 0 = not")
    label: str
    probability_disease: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0.5, le=1, description="Probability of the predicted class")
    model_name: str


class BatchRequest(BaseModel):
    patients: list[PatientFeatures] = Field(min_length=1, max_length=100)


class BatchResponse(_Response):
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(_Response):
    status: str
    model_loaded: bool
    model_name: str | None = None


class ModelInfoResponse(_Response):
    model_name: str | None = None
    features: list[str] | None = None
    threshold: float | None = None
    trained_at: str | None = None
    sklearn_version: str | None = None
    git_commit: str | None = None
    test_metrics: dict[str, Any] | None = None
