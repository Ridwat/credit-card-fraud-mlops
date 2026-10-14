# Credit Card Fraud Detection: MLOps Prototype

**Adetola Ridwat Odulaja** · Academic machine learning & API project

A prototype connecting an imbalanced-classification training workflow to a FastAPI prediction service.

**Focus:** Imbalanced data · Model serving · Containerization · CI

## Question

How can a fraud classifier be trained, serialized, and exposed through an API while preserving its preprocessing steps?

## What is implemented

The training script loads a credit-card dataset, standardizes features, creates stratified train/validation/test splits, applies SMOTE to training data, and fits an XGBoost classifier. It computes precision, recall, F1, and ROC-AUC and saves the model and scaler.

The API loads those artifacts and returns a class prediction, a fraud probability, and a readable result. Docker, Compose, basic API tests, and a CI workflow are included.

## Why it matters

Fraud detection requires attention to missed positives, false alerts, and imbalanced data. The prototype illustrates how training artifacts and preprocessing are carried into a service.

## Explore the work

- [Training script](src/train.py)
- [Prediction API](app/main.py)
- [API tests](tests/test_api.py)
- [CI workflow](.github/workflows/ci.yml)
- [Docker Compose](docker-compose.yml)

## Run

From the repository root:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`. The prediction request takes a `features` list matching the training feature order; the current smoke test supplies 30 numeric values.

To retrain, supply `data/creditcard.csv` with a `Class` target column, then run:

```bash
python src/train.py
pytest
```

The source dataset is not included. No independently reproduced model metrics are reported here.

## Limitations & next steps

The scaler is currently fitted before splitting, which leaks test information into preprocessing. Fit it on training data only and retrain before relying on evaluation. The validation subset is created but not used. Feature count and schema need explicit validation. Prediction and retraining modules under `src` are empty placeholders.

Tests check basic API responses rather than fraud-detection quality. This is a coursework prototype; source review does not establish production readiness or successful deployment.

[Explore my portfolio](https://github.com/Ridwat/Ethical-AI-Analysis/blob/main/PORTFOLIO.md)
