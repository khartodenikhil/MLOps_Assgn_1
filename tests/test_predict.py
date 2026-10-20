"""Tests for model loading and the predict() entry point."""
from __future__ import annotations

import json

import joblib
import pytest

from src.features import FEATURES
from src.predict import LABELS, load_model, predict, to_frame

RESULT_KEYS = {"prediction", "label", "probability_disease", "confidence"}


# --------------------------------------------------------------------------- #
# predict()
# --------------------------------------------------------------------------- #
def test_predict_returns_valid_results(trained_pipeline, patients):
    results = predict(patients, model=trained_pipeline)

    assert len(results) == len(patients)
    for r in results:
        assert set(r) == RESULT_KEYS
        assert r["prediction"] in (0, 1)
        assert r["label"] == LABELS[r["prediction"]]
        assert 0.0 <= r["probability_disease"] <= 1.0
        assert 0.5 <= r["confidence"] <= 1.0


def test_predict_accepts_a_single_record(trained_pipeline, patient):
    assert len(predict(patient, model=trained_pipeline)) == 1


def test_predict_is_deterministic(trained_pipeline, patients):
    assert predict(patients, model=trained_pipeline) == predict(patients, model=trained_pipeline)


def test_threshold_controls_the_class(trained_pipeline, patients):
    always_yes = predict(patients, model=trained_pipeline, threshold=0.0)
    always_no = predict(patients, model=trained_pipeline, threshold=1.01)
    assert all(r["prediction"] == 1 for r in always_yes)
    assert all(r["prediction"] == 0 for r in always_no)


def test_missing_field_is_imputed(trained_pipeline, patient):
    record = dict(patient)
    del record["ca"]
    assert len(predict(record, model=trained_pipeline)) == 1


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def test_to_frame_orders_columns(patient):
    reversed_record = dict(reversed(list(patient.items())))
    assert list(to_frame(reversed_record).columns) == FEATURES


def test_unknown_field_is_rejected(patient):
    with pytest.raises(ValueError, match="Unknown fields"):
        to_frame({**patient, "weight": 80})


def test_non_numeric_value_is_rejected(patient):
    with pytest.raises(ValueError):
        to_frame({**patient, "age": "old"})


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="No records"):
        to_frame([])


# --------------------------------------------------------------------------- #
# load_model()
# --------------------------------------------------------------------------- #
def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "missing.joblib"))


def test_load_model_round_trip(tmp_path, trained_pipeline, patients):
    path = tmp_path / "model.joblib"
    joblib.dump(trained_pipeline, path)

    loaded = load_model(str(path))

    assert predict(patients, model=loaded) == predict(patients, model=trained_pipeline)


def test_load_model_warns_on_sklearn_version_mismatch(tmp_path, trained_pipeline):
    path = tmp_path / "model.joblib"
    joblib.dump(trained_pipeline, path)
    (tmp_path / "model_meta.json").write_text(json.dumps({"sklearn_version": "0.0.0"}))

    with pytest.warns(UserWarning, match="scikit-learn"):
        load_model(str(path))
