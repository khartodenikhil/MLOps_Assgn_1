"""Train, tune and compare classifiers with MLflow tracking; save the best pipeline.

Usage (run from the project root):
    python -m src.train
    python -m src.train --select-metric recall
    python -m src.train --log-candidates     # also log every grid-search combination
    python -m src.train --cv 3               # faster, e.g. for CI

View results:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.data import CLEAN_FILE, PROJECT_ROOT, TARGET
from src.evaluate import (
    METRIC_NAMES,
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
)
from src.features import build_preprocessor, split_xy

logger = logging.getLogger(__name__)

SEED = 42
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures" / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"

# MLflow: a local SQLite store supports the Model Registry; artifacts go to ./mlruns.
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
EXPERIMENT_NAME = "heart-disease-classification"
REGISTERED_MODEL_NAME = "heart-disease-classifier"

# Each model: (estimator, hyperparameter grid). "model__" targets the pipeline step.
MODEL_SPECS: dict[str, tuple[BaseEstimator, dict]] = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED),
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 4, 8],
            "model__min_samples_leaf": [1, 3, 5],
            "model__class_weight": [None, "balanced"],
        },
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
        },
    ),
}


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    best_params: dict
    cv_metrics: dict[str, float]
    test_metrics: dict[str, float]
    cv_results: dict = field(repr=False, default_factory=dict)
    y_pred: list = field(repr=False, default_factory=list)
    y_prob: list = field(repr=False, default_factory=list)
    run_id: str | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_data(path: Path = CLEAN_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data` first.")
    return pd.read_csv(path)


def strip_prefix(params: dict) -> dict:
    return {k.removeprefix("model__"): v for k, v in params.items()}


def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - git missing or not a repo
        return "unknown"


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=(PROJECT_ROOT / "mlruns").as_uri()
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s | experiment: %s", TRACKING_URI, EXPERIMENT_NAME)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_model(
    name: str,
    estimator: BaseEstimator,
    grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    refit_metric: str = "roc_auc",
    cv_folds: int = 5,
) -> ModelResult:
    """Grid-search one model with stratified CV, then score it on the test set."""
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", clone(estimator))])
    search = GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring={m: m for m in METRIC_NAMES},
        refit=refit_metric,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED),
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    i = search.best_index_
    cv_metrics = {}
    for m in METRIC_NAMES:
        cv_metrics[f"cv_{m}_mean"] = float(search.cv_results_[f"mean_test_{m}"][i])
        cv_metrics[f"cv_{m}_std"] = float(search.cv_results_[f"std_test_{m}"][i])

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_pred, y_prob)

    params = strip_prefix(search.best_params_)
    logger.info("%s | best params %s | CV %s=%.3f | test %s=%.3f",
                name, params, refit_metric, cv_metrics[f"cv_{refit_metric}_mean"],
                refit_metric, test_metrics[refit_metric])
    return ModelResult(name, best, params, cv_metrics, test_metrics,
                       dict(search.cv_results_), list(y_pred), list(y_prob))


# --------------------------------------------------------------------------- #
# MLflow logging
# --------------------------------------------------------------------------- #
def log_candidates(result: ModelResult) -> None:
    """One nested run per grid combination, for MLflow's parallel-coordinates view."""
    cv = result.cv_results
    for i, params in enumerate(cv["params"]):
        with mlflow.start_run(run_name=f"{result.name}_candidate_{i:02d}", nested=True):
            mlflow.set_tags({"model_type": result.name, "run_level": "candidate"})
            mlflow.log_params(strip_prefix(params))
            mlflow.log_metrics({f"cv_{m}_mean": float(cv[f"mean_test_{m}"][i])
                                for m in METRIC_NAMES})


def log_model_run(
    result: ModelResult,
    grid: dict,
    X_train: pd.DataFrame,
    plots: list[Path],
    with_candidates: bool,
) -> str:
    """One nested run per model: best params, CV + test metrics, plots and the model."""
    with mlflow.start_run(run_name=result.name, nested=True) as run:
        mlflow.set_tags({"model_type": result.name, "run_level": "model"})
        mlflow.log_params(result.best_params)
        mlflow.log_param("param_grid", json.dumps(strip_prefix(grid), default=str))
        mlflow.log_metrics(result.cv_metrics)
        mlflow.log_metrics({f"test_{k}": v for k, v in result.test_metrics.items()})

        for plot in plots:
            mlflow.log_artifact(str(plot), artifact_path="plots")

        signature = infer_signature(X_train, result.pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            result.pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
        )

        if with_candidates:
            log_candidates(result)
        return run.info.run_id


def register_best(run_id: str) -> None:
    """Register the selected model and point the 'champion' alias at it."""
    version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    MlflowClient().set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version.version)
    logger.info("Registered %s version %s as 'champion'", REGISTERED_MODEL_NAME, version.version)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def comparison_table(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"model": r.name}
        row.update({f"cv_{m}": r.cv_metrics[f"cv_{m}_mean"] for m in METRIC_NAMES})
        row.update({f"test_{m}": r.test_metrics[m] for m in METRIC_NAMES})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model").round(4)


def run(
    select_metric: str = "roc_auc",
    cv_folds: int = 5,
    test_size: float = 0.2,
    with_candidates: bool = False,
    register: bool = True,
) -> ModelResult:
    df = load_data()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    logger.info("Train %d rows | test %d rows", len(X_train), len(X_test))

    setup_mlflow()
    with mlflow.start_run(run_name=f"model_selection_{select_metric}"):
        mlflow.set_tags({
            "run_level": "parent",
            "git_commit": git_commit(),
            "data_md5": file_md5(CLEAN_FILE),
        })
        mlflow.log_params({
            "seed": SEED,
            "cv_folds": cv_folds,
            "test_size": test_size,
            "select_metric": select_metric,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": X.shape[1],
        })
        try:
            dataset = mlflow.data.from_pandas(df, source=str(CLEAN_FILE),
                                              name="heart_clean", targets=TARGET)
            mlflow.log_input(dataset, context="training")
        except Exception as exc:  # noqa: BLE001 - dataset logging is a nice-to-have
            logger.warning("Could not log dataset metadata: %s", exc)

        results = []
        for name, (estimator, grid) in MODEL_SPECS.items():
            result = train_model(name, estimator, grid, X_train, y_train, X_test, y_test,
                                 refit_metric=select_metric, cv_folds=cv_folds)
            plots = [
                plot_confusion_matrix(y_test, result.y_pred, f"Confusion matrix: {name}",
                                      FIG_DIR / f"confusion_matrix_{name}.png"),
                plot_feature_importance(result.pipeline, f"Feature importance: {name}",
                                        FIG_DIR / f"feature_importance_{name}.png"),
            ]
            result.run_id = log_model_run(result, grid, X_train, plots, with_candidates)
            results.append(result)

        roc_path = plot_roc_curves({r.name: (y_test, r.y_prob) for r in results},
                                   FIG_DIR / "roc_curves.png")
        table = comparison_table(results)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        table_path = REPORTS_DIR / "model_comparison.csv"
        table.to_csv(table_path)

        # Select on the CV score, never the test score: picking by test performance
        # would turn the test set into a second validation set and inflate results.
        best = max(results, key=lambda r: r.cv_metrics[f"cv_{select_metric}_mean"])

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best.pipeline, MODEL_PATH)

        summary = {
            "selected_model": best.name,
            "selection_metric": select_metric,
            "mlflow_run_id": best.run_id,
            "best_params": best.best_params,
            "cv_metrics": best.cv_metrics,
            "test_metrics": best.test_metrics,
            "seed": SEED,
            "cv_folds": cv_folds,
            "test_size": test_size,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        metrics_path = REPORTS_DIR / "metrics.json"
        metrics_path.write_text(json.dumps(summary, indent=2, default=str))

        mlflow.set_tag("selected_model", best.name)
        mlflow.log_metrics({f"best_test_{k}": v for k, v in best.test_metrics.items()})
        for path in (roc_path, table_path, metrics_path):
            mlflow.log_artifact(str(path))

        if register:
            register_best(best.run_id)

    print("\nModel comparison:\n")
    print(table.to_string())
    print(f"\nSelected: {best.name} (highest CV {select_metric})")
    print(f"Saved pipeline to {MODEL_PATH}")
    print("View runs: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare heart disease models")
    parser.add_argument("--select-metric", choices=METRIC_NAMES, default="roc_auc")
    parser.add_argument("--cv", type=int, default=5, help="number of CV folds")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--log-candidates", action="store_true",
                        help="log every grid-search combination as a nested run")
    parser.add_argument("--no-register", action="store_true",
                        help="skip the MLflow Model Registry step")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(select_metric=args.select_metric, cv_folds=args.cv, test_size=args.test_size,
        with_candidates=args.log_candidates, register=not args.no_register)


if __name__ == "__main__":
    main()
