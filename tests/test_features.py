"""Tests for the preprocessing transformer."""
from __future__ import annotations

import numpy as np

from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.features import FEATURES, build_preprocessor, split_xy


def test_split_xy_columns_and_lengths(clean_df):
    X, y = split_xy(clean_df)
    assert list(X.columns) == FEATURES
    assert y.name == TARGET
    assert len(X) == len(y) == len(clean_df)


def test_output_shape_and_no_nans(clean_df):
    X, _ = split_xy(clean_df)
    out = build_preprocessor().fit_transform(X)

    n_onehot = sum(X[col].nunique() for col in CATEGORICAL_FEATURES)
    assert out.shape == (len(X), len(NUMERIC_FEATURES) + n_onehot)
    assert not np.isnan(out).any()


def test_numeric_features_are_standardised(clean_df):
    X, _ = split_xy(clean_df)
    out = build_preprocessor().fit_transform(X)[:, : len(NUMERIC_FEATURES)]

    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(out.std(axis=0), 1, atol=1e-7)


def test_one_hot_columns_are_binary(clean_df):
    X, _ = split_xy(clean_df)
    onehot = build_preprocessor().fit_transform(X)[:, len(NUMERIC_FEATURES):]
    assert set(np.unique(onehot)) <= {0.0, 1.0}


def test_missing_values_are_imputed(clean_df):
    X, _ = split_xy(clean_df)
    preprocessor = build_preprocessor().fit(X)

    X_missing = X.head(5).astype(float)
    X_missing.iloc[0, X_missing.columns.get_loc("chol")] = np.nan
    X_missing.iloc[1, X_missing.columns.get_loc("ca")] = np.nan

    assert not np.isnan(preprocessor.transform(X_missing)).any()


def test_unknown_category_does_not_crash(clean_df):
    X, _ = split_xy(clean_df)
    preprocessor = build_preprocessor().fit(X)
    expected_width = preprocessor.transform(X.head(1)).shape[1]

    X_new = X.head(1).copy()
    X_new["cp"] = 99

    assert preprocessor.transform(X_new).shape == (1, expected_width)
