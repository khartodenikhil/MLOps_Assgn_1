"""Tests for the reproducibility checker in scripts/compare_metrics.py."""
from __future__ import annotations

import copy
import importlib.util
import json

from src.data import PROJECT_ROOT

_spec = importlib.util.spec_from_file_location(
    "compare_metrics", PROJECT_ROOT / "scripts" / "compare_metrics.py"
)
compare_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare_metrics)

BASE = {
    "selected_model": "random_forest",
    "best_params": {"max_depth": 4},
    "cv_metrics": {"cv_roc_auc_mean": 0.90},
    "test_metrics": {"roc_auc": 0.91},
    "mlflow_run_id": "run-a",
}


def write(tmp_path, name, content) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(content))
    return str(path)


def test_identical_runs_pass_even_with_different_run_ids(tmp_path):
    other = {**BASE, "mlflow_run_id": "run-b"}
    assert compare_metrics.main(write(tmp_path, "a.json", BASE),
                                write(tmp_path, "b.json", other)) == 0


def test_changed_metric_fails(tmp_path):
    other = copy.deepcopy(BASE)
    other["test_metrics"]["roc_auc"] = 0.85
    assert compare_metrics.main(write(tmp_path, "a.json", BASE),
                                write(tmp_path, "b.json", other)) == 1


def test_changed_model_fails(tmp_path):
    other = {**BASE, "selected_model": "gradient_boosting"}
    assert compare_metrics.main(write(tmp_path, "a.json", BASE),
                                write(tmp_path, "b.json", other)) == 1
