"""Shared fixtures. Tests use synthetic data, so they never touch the network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data import RAW_COLUMNS, clean
from src.features import build_preprocessor, split_xy


def make_raw(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Random rows in the raw UCI format, with a learnable link to the target."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(95, 200, n),
        "chol": rng.integers(130, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(90, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 5, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3, 6, 7], n).astype(float),
    })
    risk = (
        (df["cp"] == 4).astype(int)
        + df["exang"]
        + (df["thalach"] < 140).astype(int)
        + (df["ca"] > 0).astype(int)
    )
    df["num"] = np.where(risk >= 2, rng.integers(1, 5, n), 0)
    return df[RAW_COLUMNS]


@pytest.fixture
def raw_factory():
    """Call as raw_factory(n, seed) to get a fresh raw DataFrame."""
    return make_raw


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def clean_df() -> pd.DataFrame:
    return clean(make_raw())


@pytest.fixture(scope="session")
def trained_pipeline() -> Pipeline:
    """A small fitted pipeline, built once per test session."""
    X, y = split_xy(clean(make_raw(200, seed=1)))
    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    return pipeline.fit(X, y)


@pytest.fixture
def patient() -> dict:
    return {
        "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0,
        "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2,
        "ca": 3, "thal": 3,
    }


@pytest.fixture
def patients(patient) -> list[dict]:
    low_risk = {
        "age": 41, "sex": 0, "cp": 2, "trestbps": 130, "chol": 204, "fbs": 0,
        "restecg": 2, "thalach": 172, "exang": 0, "oldpeak": 1.4, "slope": 1,
        "ca": 0, "thal": 3,
    }
    return [patient, low_risk]
