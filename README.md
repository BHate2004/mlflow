# Gurgaon Real Estate MLOps Pipeline

## Project Overview

This project builds a production-grade MLOps pipeline for **Gurgaon Property Price Prediction**. It mirrors the architecture and workflow of the NEPSE ML pipeline, adapted for a cross-sectional real estate regression problem instead of a time-series stock problem.

### What it predicts
The model predicts **property price in Crore INR** given features like:
- Property type, sector, number of bedrooms/bathrooms
- Built-up area, balcony count, age/possession status
- Furnishing type, luxury score, servant/store room presence

### Key differences from NEPSE pipeline
| Aspect | NEPSE Pipeline | This Pipeline |
|---|---|---|
| Domain | Stock prices (NABIL) | Real estate (Gurgaon) |
| Problem type | Time-series regression | Cross-sectional regression |
| Split strategy | Chronological 80/20 | Shuffled 80/20 |
| Target encoding | `next_close` (shift-1) | `price` (direct) |
| High-cardinality column | `symbol` (target-encoded) | `sector` (target-encoded) |
| Airflow DAGs | daily prediction + weekly retrain | daily monitoring + weekly retrain |

---

## Architecture

```
[gurgaon_properties.csv]
         │  (Daily Airflow Cron)
         ▼
[MariaDB raw_properties table]
         │  (Feature Engineering)
         ▼
[Redis Cache] ──▶ [MariaDB processed_properties table]
         │
         ▼  (Weekly Airflow Cron)
[Train: Baseline / LinearRegression / Ridge / RandomForest]
         │
         ├─▶ [MLflow Registry (Metrics & Artifacts)]
         │
         ▼
[FastAPI Serving Layer] ◀── /predict POST
         │
         ▼  (Daily Airflow Cron)
[Evidently AI (Data Drift HTML Reports)]
```

---

## Quick Start

### 1. Docker (Full Stack)

```bash
git clone <repo>
cd real-estate-ml-pipeline
cp .env.example .env
docker compose up -d --build
```

URLs once running:
- **FastAPI Swagger**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **Airflow UI**: http://localhost:8080 (admin/admin)

### 2. Local (No Docker)

```bash
pip install -r requirements.txt
cp .env.example .env

# Step 1: Ingest raw CSV into MariaDB (or just reads CSV if no DB)
python -m src.ingestion.load_to_mariadb

# Step 2: Feature engineering → processed CSV + MariaDB + Redis
python -m src.preprocessing.feature_engineering

# Step 3: Train models, select best, save artifacts
python -m src.training.train_model

# Step 4: Start prediction API
uvicorn src.serving.main:app --host 0.0.0.0 --port 8000

# Step 5: Run data drift monitoring
python -m src.monitoring.evidently_report
```

---

## Pipeline Steps

### `src/ingestion/load_to_mariadb.py`
Reads the raw CSV (`data/raw/gurgaon_properties.csv`) and loads it into the `raw_properties` MariaDB table. Gracefully falls back to CSV-only if DB is unavailable.

### `src/preprocessing/feature_engineering.py`
- Drops unused columns (`floorNum`, `pooja room`, etc.)
- Converts numeric `furnishing_type` to labels (unfurnished / semiunfurnished / furnished)
- Engineers `price_per_sqft` and `bath_bed_ratio`
- Log-transforms price (`price_log = log1p(price)`)
- Saves to CSV + MariaDB + Redis cache

### `src/training/train_model.py`
- Loads processed data from Redis → MariaDB → CSV fallback
- Builds `ColumnTransformer` preprocessor:
  - `StandardScaler` for numeric columns
  - `OneHotEncoder` for `agePossession`
  - `OrdinalEncoder` for balcony, luxury_score, property_type, furnishing_type
  - `TargetEncoder` for `sector` (high-cardinality)
- Trains: Baseline, LinearRegression, Ridge, RandomForestRegressor
- Selects best model by RMSE on original price scale
- Saves `models/best_model.joblib` and `models/model_metadata.json`
- Logs all runs to MLflow (skips gracefully if unavailable)

### `src/serving/main.py`
FastAPI app with `/predict` POST endpoint. Accepts property features, returns predicted price in Crore INR.

### `src/monitoring/evidently_report.py`
Generates Evidently HTML reports for data drift and quality, saved to `reports/monitoring/`.

---

## Airflow DAGs

| DAG | Schedule | Tasks |
|---|---|---|
| `weekly_training_dag` | Every Sunday 18:00 | preprocess → train → model check |
| `daily_monitoring_dag` | Every day 19:00 | ingest → preprocess → drift report → summary |

---

## Example Prediction Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "property_type": "flat",
    "sector": "sector 36",
    "bedRoom": 3,
    "bathroom": 2,
    "balcony": "2",
    "agePossession": "New Property",
    "built_up_area": 1200,
    "servant_room": 0,
    "store_room": 0,
    "furnishing_type": "unfurnished",
    "luxury_score": "Low"
  }'
```

Response:
```json
{
  "predicted_price_cr": 1.24,
  "model_version": "RandomForest",
  "input_features": { ... }
}
```

---

## Troubleshooting

- **`/predict` fails**: Run `python -m src.training.train_model` to generate `models/best_model.joblib`.
- **MariaDB errors**: Pipeline gracefully falls back to CSV. Set correct credentials in `.env`.
- **Redis errors**: Non-fatal; caching is skipped silently.
- **MLflow not logging**: Non-fatal; ensure MLflow server is running at `MLFLOW_TRACKING_URI`.
