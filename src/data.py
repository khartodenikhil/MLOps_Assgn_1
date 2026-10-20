"""Data acquisition, cleaning and validation for the UCI Heart Disease (Cleveland) dataset.

Usage (run from the project root):
    python -m src.data                      # download if missing, clean, save
    python -m src.data --force              # re-download the raw file
    python -m src.data --missing mode       # impute instead of dropping rows
"""
from __future__ import annotations

import argparse
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_FILE = RAW_DIR / "processed.cleveland.data"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"

PRIMARY_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
FALLBACK_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

# The raw file has no header row, so column names come from the UCI documentation.
RAW_COLUMNS = [
    "age",       # years
    "sex",       # 1 = male, 0 = female
    "cp",        # chest pain type: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic
    "trestbps",  # resting blood pressure (mm Hg)
    "chol",      # serum cholesterol (mg/dl)
    "fbs",       # fasting blood sugar > 120 mg/dl (1 = true)
    "restecg",   # resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy
    "thalach",   # max heart rate achieved
    "exang",     # exercise-induced angina (1 = yes)
    "oldpeak",   # ST depression induced by exercise
    "slope",     # slope of peak exercise ST segment: 1 up, 2 flat, 3 down
    "ca",        # number of major vessels coloured by fluoroscopy (0-3)
    "thal",      # 3 normal, 6 fixed defect, 7 reversible defect
    "num",       # diagnosis: 0 = no disease, 1-4 = disease severity
]

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
TARGET = "target"

# Plausibility checks: a failure here means the data is corrupt, not just unusual.
VALID_RANGES = {
    "age": (1, 120),
    "trestbps": (50, 250),
    "chol": (50, 700),
    "thalach": (50, 250),
    "oldpeak": (0, 10),
}
ALLOWED_VALUES = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_raw(force: bool = False, timeout: int = 30) -> Path:
    """Download the raw Cleveland file into data/raw/. Skips if already present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists() and not force:
        logger.info("Raw file already exists at %s (use --force to re-download)", RAW_FILE)
        return RAW_FILE

    try:
        logger.info("Downloading %s", PRIMARY_URL)
        with urllib.request.urlopen(PRIMARY_URL, timeout=timeout) as resp:
            content = resp.read()
    except Exception as exc:  # noqa: BLE001 - any failure triggers the fallback
        logger.warning("Primary download failed (%s); trying zip archive", exc)
        with urllib.request.urlopen(FALLBACK_ZIP_URL, timeout=timeout) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        member = next(n for n in archive.namelist() if n.endswith("processed.cleveland.data"))
        content = archive.read(member)

    if not content.strip():
        raise RuntimeError("Downloaded file is empty")

    RAW_FILE.write_bytes(content)
    logger.info("Saved raw data to %s (%d bytes)", RAW_FILE, len(content))
    return RAW_FILE


# --------------------------------------------------------------------------- #
# Load, clean, validate
# --------------------------------------------------------------------------- #
def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Read the raw file. '?' marks missing values in this dataset."""
    return pd.read_csv(path, header=None, names=RAW_COLUMNS, na_values="?")


def clean(df: pd.DataFrame, missing_strategy: str = "drop") -> pd.DataFrame:
    """Return a cleaned copy with a binary `target` column.

    missing_strategy:
        "drop" - drop rows with missing values (only ~6 rows in Cleveland)
        "mode" - impute categoricals with the mode, numerics with the median
    """
    df = df.copy().apply(pd.to_numeric, errors="coerce")

    missing = df.isna().sum()
    if missing.any():
        logger.info("Missing values before cleaning: %s", missing[missing > 0].to_dict())

    n_before = len(df)
    df = df.drop_duplicates()
    if len(df) < n_before:
        logger.info("Dropped %d duplicate rows", n_before - len(df))

    if missing_strategy == "drop":
        n_before = len(df)
        df = df.dropna()
        logger.info("Dropped %d rows with missing values", n_before - len(df))
    elif missing_strategy == "mode":
        for col in df.columns[df.isna().any()]:
            fill = df[col].mode()[0] if col in CATEGORICAL_FEATURES else df[col].median()
            df[col] = df[col].fillna(fill)
            logger.info("Imputed %s with %s", col, fill)
    else:
        raise ValueError(f"Unknown missing_strategy: {missing_strategy!r}")

    # Collapse severity levels 1-4 into a single "disease present" class.
    df[TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(int)

    validate(df)
    return df.reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError if the cleaned frame breaks any schema or range rule."""
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    absent = expected - set(df.columns)
    if absent:
        raise ValueError(f"Missing columns: {sorted(absent)}")

    problems = []
    if df.isna().any().any():
        problems.append("NaN values remain after cleaning")
    for col, (lo, hi) in VALID_RANGES.items():
        bad = ~df[col].between(lo, hi)
        if bad.any():
            problems.append(f"{col}: {int(bad.sum())} values outside [{lo}, {hi}]")
    for col, allowed in ALLOWED_VALUES.items():
        bad = ~df[col].isin(allowed)
        if bad.any():
            problems.append(f"{col}: unexpected values {sorted(df.loc[bad, col].unique())}")
    if not set(df[TARGET].unique()) <= {0, 1}:
        problems.append("target is not binary")

    if problems:
        raise ValueError("Data validation failed:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Download and clean the heart disease data")
    parser.add_argument("--force", action="store_true", help="re-download the raw file")
    parser.add_argument("--missing", choices=["drop", "mode"], default="drop")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw = load_raw(download_raw(force=args.force))
    logger.info("Raw shape: %s", raw.shape)

    df = clean(raw, missing_strategy=args.missing)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_FILE, index=False)

    balance = df[TARGET].value_counts().sort_index().to_dict()
    logger.info("Clean shape: %s | class balance: %s", df.shape, balance)
    logger.info("Saved cleaned data to %s", CLEAN_FILE)


if __name__ == "__main__":
    main()
