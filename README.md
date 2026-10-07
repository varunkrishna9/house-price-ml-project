# House Price Prediction — End-to-End Machine Learning System

## Overview
Production-style ML system for residential price prediction using the supplied 300-row dataset.

**Best model:** Gradient Boosting Regressor  
**Test R²:** 0.9850  
**Test MAE:** ₹993,300  
**5-fold CV R²:** 0.9894 ± 0.0026

## Workflow
Data validation → feature engineering → preprocessing → 3-model comparison → cross-validation → explainability → versioned model → Flask UI/API → validation/tests.

## Models Compared
| Model | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Linear Regression | ₹2,339,561 | ₹3,104,977 | 0.9323 | 14.04% |\n| Random Forest | ₹1,412,156 | ₹1,998,822 | 0.9719 | 6.07% |\n| Gradient Boosting | ₹993,300 | ₹1,459,545 | 0.9850 | 4.75% |\n
## Feature Engineering
- Total_Rooms
- Area_per_Bedroom
- Bathrooms_per_Bedroom
- Age_Squared
- Area_Log

## Structure
```text
house_price_ml_project/
├── app/                 # Flask UI + REST API
├── config/              # configuration
├── data/                # supplied dataset
├── docs/                # reports and documentation
├── models/              # versioned model + metadata
├── notebooks/           # EDA/model notebook
├── scripts/             # training and app launchers
├── src/                 # production ML code
├── tests/               # automated tests
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Tests
```bash
pytest -q
```

## Run Web App
```bash
python scripts/run_app.py
```
Open `http://127.0.0.1:5000`.

## Retrain
```bash
python scripts/train_model.py
```

## API
`POST /predict`
```json
{
  "Area": 1200,
  "Bedrooms": 3,
  "Bathrooms": 2,
  "Age": 5,
  "Location": "City Center",
  "Property_Type": "Apartment"
}
```

Response:
```json
{"predicted_price": 12450000.0, "currency": "INR"}
```

## Explainability
`docs/feature_importance.csv` contains permutation importance on the held-out test set. `docs/model_comparison.csv` contains all model metrics.

## Model Versioning
Current artifact: `models/house_price_model_v1.0.0.joblib` with metadata in `models/metadata_v1.0.0.json`.

## Limitations
This is an educational/project dataset with only 300 rows. Production valuation would need larger, more representative data, geographic and temporal coverage, market conditions, monitoring and recalibration.

## Production Extensions
MLflow, Docker, CI/CD, model registry, cloud storage, authentication, rate limiting, data/model drift monitoring and scheduled retraining.
