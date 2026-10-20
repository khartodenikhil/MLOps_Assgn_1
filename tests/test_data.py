"""Tests for data download, loading, cleaning and validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.data as data
from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, clean, load_raw, validate

# Two rows in the real file format; the second has a missing `thal` ("?").
RAW_TEXT = (
    "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n"
    "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,3.0,?,2\n"
)


class FakeResponse:
    """Stands in for the object urllib.request.urlopen returns."""

    def __init__(self, content: bytes):
        self._content = content

    def read(self) -> bytes:
        return self._content

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def test_load_raw_reads_question_marks_as_nan(tmp_path):
    path = tmp_path / "raw.data"
    path.write_text(RAW_TEXT)

    df = load_raw(path)

    assert df.shape == (2, 14)
    assert df["thal"].isna().sum() == 1
    assert df["ca"].isna().sum() == 0


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def test_clean_creates_binary_target(raw_df):
    df = clean(raw_df)
    assert TARGET in df.columns
    assert "num" not in df.columns
    assert set(df[TARGET].unique()) <= {0, 1}


def test_clean_maps_any_severity_to_disease(raw_factory):
    raw = raw_factory(5)
    raw["num"] = [0, 1, 2, 3, 4]
    assert clean(raw)[TARGET].tolist() == [0, 1, 1, 1, 1]


def test_drop_strategy_removes_rows_with_missing_values(raw_df):
    raw = raw_df.copy()
    raw.loc[0, "ca"] = np.nan
    raw.loc[1, "thal"] = np.nan

    df = clean(raw, missing_strategy="drop")

    assert len(df) == len(raw) - 2
    assert not df.isna().any().any()


def test_mode_strategy_keeps_all_rows(raw_df):
    raw = raw_df.copy()
    raw.loc[0, "ca"] = np.nan
    raw.loc[1, "chol"] = np.nan

    df = clean(raw, missing_strategy="mode")

    assert len(df) == len(raw)
    assert not df.isna().any().any()


def test_clean_removes_duplicates(raw_df):
    raw = pd.concat([raw_df, raw_df.head(3)], ignore_index=True)
    assert len(clean(raw)) == len(raw_df)


def test_clean_does_not_modify_input(raw_df):
    before = raw_df.copy()
    clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_unknown_missing_strategy_raises(raw_df):
    with pytest.raises(ValueError, match="missing_strategy"):
        clean(raw_df, missing_strategy="magic")


def test_cleaned_column_types(clean_df):
    for col in CATEGORICAL_FEATURES:
        assert pd.api.types.is_integer_dtype(clean_df[col]), col
    for col in NUMERIC_FEATURES:
        assert pd.api.types.is_numeric_dtype(clean_df[col]), col


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validate_accepts_clean_data(clean_df):
    validate(clean_df)  # should not raise


@pytest.mark.parametrize(
    "column, value",
    [("age", 200), ("chol", 5), ("trestbps", 400), ("cp", 9), ("thal", 5), ("sex", 2)],
)
def test_validate_rejects_invalid_values(clean_df, column, value):
    df = clean_df.copy()
    df.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        validate(df)


def test_validate_rejects_missing_column(clean_df):
    with pytest.raises(ValueError, match="Missing columns"):
        validate(clean_df.drop(columns="chol"))


def test_validate_rejects_nan(clean_df):
    df = clean_df.copy()
    df["chol"] = df["chol"].astype(float)
    df.loc[0, "chol"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate(df)


# --------------------------------------------------------------------------- #
# Download (network calls are replaced with fakes)
# --------------------------------------------------------------------------- #
@pytest.fixture
def tmp_raw_file(tmp_path, monkeypatch):
    """Redirect the module's raw-data paths into a temporary folder."""
    raw_dir = tmp_path / "raw"
    raw_file = raw_dir / "processed.cleveland.data"
    monkeypatch.setattr(data, "RAW_DIR", raw_dir)
    monkeypatch.setattr(data, "RAW_FILE", raw_file)
    return raw_file


def test_download_writes_file(tmp_raw_file, monkeypatch):
    monkeypatch.setattr(data.urllib.request, "urlopen",
                        lambda url, timeout: FakeResponse(RAW_TEXT.encode()))

    path = data.download_raw()

    assert path == tmp_raw_file
    assert path.read_text() == RAW_TEXT


def test_download_skips_existing_file(tmp_raw_file, monkeypatch):
    tmp_raw_file.parent.mkdir(parents=True)
    tmp_raw_file.write_text(RAW_TEXT)

    def fail(*args, **kwargs):
        raise AssertionError("network should not be used when the file exists")

    monkeypatch.setattr(data.urllib.request, "urlopen", fail)
    assert data.download_raw() == tmp_raw_file


def test_download_rejects_empty_response(tmp_raw_file, monkeypatch):
    monkeypatch.setattr(data.urllib.request, "urlopen",
                        lambda url, timeout: FakeResponse(b"   "))
    with pytest.raises(RuntimeError, match="empty"):
        data.download_raw()
