# Heart Disease Risk Prediction - MLOps Pipeline

![CI](https://github.com/khartodenikhil/MLOps_Assgn_1/actions/workflows/ci.yml/badge.svg)

End-to-end MLOps project for BITS Pilani AIMLCZG523 (Assignment 01): predicting heart disease risk from patient health data using the UCI Heart Disease (Cleveland) dataset.

## Project structure

```
heart-disease-mlops/
├── data/
│   ├── raw/              # downloaded file (git-ignored, reproducible)
│   └── processed/        # heart_clean.csv (committed)
├── notebooks/            # EDA and experiments
├── src/                  # reusable Python modules
├── tests/                # unit tests
├── reports/figures/      # saved plots
└── screenshots/          # evidence for the report
```

## Setup

Requires Python 3.11.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Data

Source: [UCI Machine Learning Repository - Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), Cleveland subset (303 patients, 13 features).

Download and clean (run from the project root):

```bash
python -m src.data
```

This downloads the raw file to `data/raw/`, handles missing values (marked `?` in the source), converts the 0-4 `num` diagnosis into a binary `target`, validates ranges, and writes `data/processed/heart_clean.csv`.

| Column | Meaning |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (1-4) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG result (0-2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment (1-3) |
| ca | Major vessels coloured by fluoroscopy (0-3) |
| thal | Thalassemia (3 normal, 6 fixed, 7 reversible) |
| target | 1 = heart disease present, 0 = absent |

## EDA

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Figures are saved to `reports/figures/`.

## Model training

```bash
python -m src.train                        # select by CV ROC-AUC (default)
python -m src.train --select-metric recall # prioritise catching disease cases
```

Three models are tuned with `GridSearchCV` and stratified 5-fold cross-validation on an 80/20 stratified split (seed 42):

- Logistic Regression
- Random Forest
- Gradient Boosting

Preprocessing (imputation, scaling, one-hot encoding) lives inside the sklearn `Pipeline`, so the saved model takes raw feature values.

Outputs:

| Path | Contents |
|---|---|
| `models/model.joblib` | Best full pipeline (preprocessing + model) |
| `reports/model_comparison.csv` | CV and test metrics for every model |
| `reports/metrics.json` | Selected model, parameters and metrics |
| `reports/figures/models/` | Confusion matrices, ROC curves, feature importances |

## Experiment tracking (MLflow)

Every training run is logged to MLflow (local SQLite store `mlflow.db`, artifacts in `mlruns/`):

```
model_selection_<metric>          parent run: config, data hash, git commit, ROC plot, comparison table
├── logistic_regression           best params, CV mean/std + test metrics, plots, model
├── random_forest
└── gradient_boosting
```

The selected model is registered as `heart-disease-classifier` with the alias `champion`.

```bash
python -m src.train --log-candidates                  # also log every grid combination
mlflow ui --backend-store-uri sqlite:///mlflow.db     # open http://127.0.0.1:5000
```

## Inference

The saved artifact is a full sklearn `Pipeline`, so it accepts raw feature values; no separate preprocessing step is needed.

```bash
python -m src.predict --input samples/patients.json                     # models/model.joblib
python -m src.predict --input samples/patients.json --source registry   # MLflow 'champion'
```

Example output:

```json
[{"prediction": 1, "label": "Disease", "probability_disease": 0.91, "confidence": 0.91}]
```

Missing fields are imputed by the pipeline (the third sample patient omits `ca`); unknown fields are rejected. `MODEL_PATH` and `PREDICTION_THRESHOLD` can be set as environment variables.

| File | Purpose |
|---|---|
| `models/model.joblib` | Preprocessing + model pipeline |
| `models/model_meta.json` | Model name, features, library versions, git commit, data hash, metrics |

## Reproducibility

- All dependencies are pinned in `requirements.txt`.
- One seed (`42`) controls the train/test split, CV folds and every model.
- `models/model_meta.json` records the Python and scikit-learn versions used; `src/predict.py` warns if they differ at load time.
- Each MLflow run is tagged with the git commit and an MD5 hash of the training data.

To rebuild everything in a fresh environment and confirm the metrics match:

```bash
# Windows
powershell -ExecutionPolicy Bypass -File scripts\reproduce.ps1
# macOS/Linux
bash scripts/reproduce.sh
```

The script creates `.venv-repro`, installs from `requirements.txt`, re-downloads the data, retrains, predicts on the samples, and fails if any metric differs from the committed `reports/metrics.json`.

## Testing and linting

```bash
pytest                      # all tests
pytest --cov                # with coverage report (fails below 60%)
ruff check .                # lint
```

Tests use synthetic data and fake network responses, so they run offline in a few seconds.

| Test file | Covers |
|---|---|
| `test_data.py` | `?` parsing, binary target, missing-value strategies, duplicates, range/category validation, download |
| `test_features.py` | Output shape, scaling, one-hot encoding, imputation, unseen categories |
| `test_train.py` | Grid search returns all CV and test metrics; model beats chance |
| `test_evaluate.py` | Metric calculation, plot files |
| `test_predict.py` | Output structure, threshold, missing/unknown/invalid fields, model loading, version warning |
| `test_compare_metrics.py` | Reproducibility checker |

## CI/CD (GitHub Actions)

`.github/workflows/ci.yml` runs on every push to `main`, every pull request, and on demand:

```
lint (ruff) ──► test (Python 3.10 + 3.11, coverage) ──► train (download → train → predict → compare)
```

- Each stage only runs if the previous one passed, so a lint error or failing test stops the pipeline.
- Test results (JUnit XML, coverage) and the trained model, reports and MLflow runs are uploaded as workflow artifacts.
- The train job writes a model comparison table to the run's summary page.
