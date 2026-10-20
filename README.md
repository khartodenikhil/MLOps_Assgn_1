# Heart Disease Risk Prediction - MLOps Pipeline

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
