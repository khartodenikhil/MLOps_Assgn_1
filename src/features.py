"""Feature engineering: a single preprocessing transformer shared by every model.

Keeping preprocessing inside the sklearn Pipeline means the exact same
transformations run during training, cross-validation and API inference.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Impute, then scale numerics and one-hot encode categoricals.

    - Imputers: the training data has no NaNs (rows were dropped in Step 1),
      but API requests may omit a field, so the pipeline must handle it.
    - StandardScaler: needed by Logistic Regression; harmless for tree models.
    - OneHotEncoder: cp, restecg, slope, ca and thal are nominal codes, so
      treating "4" as greater than "1" would be meaningless.
      handle_unknown="ignore" stops unseen codes from crashing inference.
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model inputs from the target, in a fixed column order."""
    return df[FEATURES].copy(), df[TARGET].copy()
