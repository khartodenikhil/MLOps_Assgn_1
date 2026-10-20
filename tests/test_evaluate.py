"""Tests for metrics and evaluation plots."""
from __future__ import annotations

import pytest

from src.evaluate import (
    METRIC_NAMES,
    compute_metrics,
    feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
)
from src.features import split_xy


def test_perfect_predictions_score_one():
    metrics = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics == {m: pytest.approx(1.0) for m in METRIC_NAMES}


def test_metrics_are_in_range():
    metrics = compute_metrics([0, 1, 1, 0, 1], [0, 1, 0, 1, 1], [0.2, 0.7, 0.4, 0.6, 0.9])
    assert set(metrics) == set(METRIC_NAMES)
    assert all(0.0 <= v <= 1.0 for v in metrics.values())


def test_plots_are_written(tmp_path, trained_pipeline, clean_df):
    X, y = split_xy(clean_df)
    y_prob = trained_pipeline.predict_proba(X)[:, 1]
    y_pred = trained_pipeline.predict(X)

    paths = [
        plot_confusion_matrix(y, y_pred, "cm", tmp_path / "cm.png"),
        plot_roc_curves({"lr": (y, y_prob)}, tmp_path / "roc.png"),
        plot_feature_importance(trained_pipeline, "fi", tmp_path / "fi.png"),
    ]
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_feature_importance_covers_every_encoded_feature(trained_pipeline):
    importance = feature_importance(trained_pipeline)
    names = trained_pipeline.named_steps["preprocess"].get_feature_names_out()
    assert len(importance) == len(names)
    assert (importance >= 0).all()
