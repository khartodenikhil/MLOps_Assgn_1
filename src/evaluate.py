"""Metrics and evaluation plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed: works in CI and Docker

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402

CLASS_NAMES = ["No disease", "Disease"]
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def compute_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=CLASS_NAMES, cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curves(curves: dict[str, tuple], path: Path) -> Path:
    """curves maps model name -> (y_true, y_prob)."""
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, (y_true, y_prob) in curves.items():
        RocCurveDisplay.from_predictions(y_true, y_prob, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def feature_importance(pipeline: Pipeline) -> pd.Series:
    """Importance per encoded feature: tree importances or |LR coefficient|."""
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        raise TypeError(f"{type(model).__name__} exposes no importances")
    return pd.Series(values, index=names).sort_values(ascending=False)


def plot_feature_importance(pipeline: Pipeline, title: str, path: Path, top_n: int = 15) -> Path:
    importance = feature_importance(pipeline).head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    importance.plot.barh(color="#4C72B0", ax=ax)
    ax.set(title=title, xlabel="Importance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
