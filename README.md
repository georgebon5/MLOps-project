# MLOps Churn Prediction Pipeline

![CI](https://github.com/georgebon5/MLOps-project/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/mlflow-tracked-0194E2)
![DVC](https://img.shields.io/badge/dvc-versioned-945DD6)
![License](https://img.shields.io/badge/license-MIT-green)

End-to-end MLOps pipeline for bank customer churn prediction. Built to be reproducible, containerized, and ready to deploy — not just a notebook.

---

## What it does

Takes raw customer data, trains a classifier to predict churn, serves it via a REST API, and monitors production data for drift. Every step is versioned, tested, and automated.

The dataset is synthetic but reflects realistic class imbalance (~9% churn rate) and feature relationships found in real banking data.

---

## Stack

| Layer | Tool |
|---|---|
| Data & model versioning | DVC + Git |
| Experiment tracking | MLflow |
| Model serving | FastAPI + Uvicorn |
| Containerization | Docker (multi-stage) |
| CI/CD | GitHub Actions |
| Drift monitoring | Evidently AI |

---

## Project structure

```
├── notebooks/
│   └── 01_eda.ipynb   # exploratory data analysis
├── src/
│   ├── data/          # data generation and preprocessing
│   ├── models/        # training, evaluation
│   └── api/           # FastAPI app
├── tests/             # unit tests
├── monitoring/        # Evidently drift reports
├── .github/workflows/ # CI/CD pipeline
├── dvc.yaml           # reproducible pipeline stages
└── params.yaml        # all hyperparameters in one place
```

---

## Quickstart

```bash
git clone https://github.com/georgebon5/MLOps-project.git
cd MLOps-project

python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt

# Run the full pipeline
dvc repro

# Start the API
uvicorn src.api.main:app --reload
```

API docs available at `http://localhost:8000/docs`

---

## EDA

Before modeling, exploratory analysis covers class imbalance, feature distributions, correlation heatmap, and feature importance. Open the notebook:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## Training & Model Registry

Three models are compared per run — Random Forest, Gradient Boosting, and Logistic Regression. SMOTE handles class imbalance. The best model by F1 score is automatically registered in the MLflow Model Registry and promoted to `@production`.

```bash
python src/models/train.py
mlflow ui  # view experiments and registry
```

The API can load directly from the registry instead of a local file:

```bash
USE_REGISTRY=true uvicorn src.api.main:app --reload
```

---

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data/processed:/app/data/processed:ro \
  churn-api
```

---

## Monitoring

Generates HTML drift reports comparing training distribution vs production batches.

```bash
python monitoring/evidently_report.py
open monitoring/reports/data_drift_report.html
```

---

## CI/CD

Every push to `main` triggers:
1. Black + Flake8 + isort
2. pytest (10 tests)
3. Docker build → push to GitHub Container Registry

---

## Results

| Model | F1 | ROC-AUC |
|---|---|---|
| Random Forest | 0.265 | 0.730 |
| Gradient Boosting | 0.131 | 0.726 |
| Logistic Regression | 0.250 | 0.703 |

Low F1 is expected given the class imbalance — the focus here is the pipeline architecture, not squeezing out model performance.
