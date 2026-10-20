"""Tests for model training helpers (MLflow logging is exercised by the CI train job)."""
from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data import clean
from src.evaluate import METRIC_NAMES
from src.features import split_xy
from src.train import comparison_table, strip_prefix, train_model


@pytest.fixture
def split(raw_factory):
    X, y = split_xy(clean(raw_factory(200, seed=2)))
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


@pytest.fixture
def result(split):
    X_train, X_test, y_train, y_test = split
    return train_model(
        "logistic_regression",
        LogisticRegression(max_iter=1000),
        {"model__C": [0.1, 1.0]},
        X_train, y_train, X_test, y_test,
        cv_folds=3,
    )


def test_train_model_returns_all_metrics(result, split):
    _, X_test, _, _ = split
    assert set(result.best_params) == {"C"}
    assert len(result.y_prob) == len(X_test)
    for m in METRIC_NAMES:
        assert 0.0 <= result.test_metrics[m] <= 1.0
        assert f"cv_{m}_mean" in result.cv_metrics
        assert f"cv_{m}_std" in result.cv_metrics


def test_trained_model_beats_chance(result):
    assert result.test_metrics["roc_auc"] > 0.6


def test_best_pipeline_can_predict(result, split):
    _, X_test, _, _ = split
    assert set(result.pipeline.predict(X_test)) <= {0, 1}


def test_comparison_table(result):
    table = comparison_table([result])
    assert list(table.index) == ["logistic_regression"]
    for m in METRIC_NAMES:
        assert f"cv_{m}" in table.columns
        assert f"test_{m}" in table.columns


def test_strip_prefix():
    assert strip_prefix({"model__C": 1.0, "model__max_depth": 3}) == {"C": 1.0, "max_depth": 3}
