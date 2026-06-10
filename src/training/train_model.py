"""
Model Training Pipeline — Gurgaon Real Estate Price Prediction
==============================================================
Mirrors the NEPSE pipeline structure:
  - Redis → MariaDB → CSV fallback for data loading
  - Stratified 80/20 split (shuffled; this is not time-series)
  - Baseline + LinearRegression + RandomForest evaluated
  - Best model (lowest RMSE on original price scale) saved locally
  - MLflow tracking with graceful skip when server is unavailable
"""
import json

import category_encoders as ce
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

from src.config.settings import (
    MODELS_DIR,
    NUMERIC_COLS,
    ONEHOT_COLS,
    ORDINAL_COLS,
    PROCESSED_DATA_DIR,
    TARGET_COL,
    TARGET_ENCODE_COLS,
)
from src.registry.mlflow_utils import MLflowManager
from src.storage.mariadb_client import MariaDBClient
from src.storage.redis_client import RedisClient
from src.training.evaluate_model import evaluate_predictions, save_evaluation_report
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data loading — Redis → MariaDB → CSV
# ---------------------------------------------------------------------------

def fetch_processed_data() -> pd.DataFrame:
    # 1. Redis
    redis_client = RedisClient()
    df = redis_client.get_cached_dataframe("processed_properties")
    if df is not None and not df.empty:
        logger.info("Loaded processed data from Redis cache.")
        return df

    # 2. MariaDB
    logger.info("Cache miss. Trying MariaDB...")
    db_client = MariaDBClient()
    try:
        with db_client.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM processed_properties")
            rows = cursor.fetchall()
            if rows:
                columns = [desc[0] for desc in cursor.description]
                df = pd.DataFrame(rows, columns=columns)
                df = df.rename(columns={
                    "servant_room": "servant room",
                    "store_room": "store room",
                    "furnishing_type_label": "furnishing_type",
                })
                logger.info(f"Loaded {len(df)} rows from MariaDB.")
                return df
    except Exception as e:
        logger.warning(f"MariaDB load failed: {e}")

    # 3. CSV fallback
    csv_path = PROCESSED_DATA_DIR / "properties_features.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV fallback.")
        return df

    logger.error("Could not load processed data from any source.")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Constructs the column transformer matching model.ipynb."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), ONEHOT_COLS),
        ("order", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ORDINAL_COLS),
        ("target", ce.TargetEncoder(), TARGET_ENCODE_COLS),
    ], remainder="passthrough")


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def split_data(df: pd.DataFrame, features: list, target_log: str):
    """Shuffled 80/20 split (property data is cross-sectional, not time-series)."""
    X = df[features]
    y = df[target_log]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def scorer(model_name: str, model, X, y_log, preprocessor) -> dict:
    """K-fold CV score + held-out MAE on original price scale."""
    pipeline = Pipeline([("preprocessor", preprocessor), (model_name, model)])
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y_log, cv=kfold, scoring="r2")

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred_log = pipeline.predict(X_test)

    # Back-transform from log1p
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    metrics = evaluate_predictions(y_true.values, y_pred)
    metrics["cv_r2_mean"] = float(cv_scores.mean())
    metrics["cv_r2_std"] = float(cv_scores.std())
    return metrics, pipeline


def train_and_evaluate():
    df = fetch_processed_data()
    if df.empty:
        logger.error("No data available to train.")
        return

    # Log-transformed target (already created in preprocessing)
    target_log = "price_log"

    # All feature columns (exclude target variants, ids, derived targets)
    exclude_cols = {"id", TARGET_COL, "price_log", "price_per_sqft", "bath_bed_ratio"}
    features = [c for c in df.columns if c not in exclude_cols]
    # Keep only the columns the preprocessor actually knows about
    keep = NUMERIC_COLS + ONEHOT_COLS + ORDINAL_COLS + TARGET_ENCODE_COLS
    features = [f for f in features if f in keep]

    df = df.dropna(subset=features + [target_log]).reset_index(drop=True)

    X = df[features]
    y_log = df[target_log]

    logger.info(f"Training on {len(df)} rows with {len(features)} features: {features}")

    preprocessor = build_preprocessor()
    mlflow_manager = MLflowManager()

    results = {}

    # 1. Baseline — median price
    logger.info("--- Baseline Model (median price) ---")
    X_train, X_test, y_train, y_test = split_data(df, features, target_log)
    median_log = y_train.median()
    y_pred_baseline = np.expm1(np.full(len(y_test), median_log))
    y_true_baseline = np.expm1(y_test.values)
    baseline_metrics = evaluate_predictions(y_true_baseline, y_pred_baseline)
    logger.info(f"Baseline Metrics: {baseline_metrics}")
    save_evaluation_report("Baseline", baseline_metrics)
    results["Baseline"] = (None, baseline_metrics)

    # 2. Linear Regression
    logger.info("--- Linear Regression ---")
    lr_metrics, lr_pipeline = scorer("linear_regression", LinearRegression(), X, y_log, build_preprocessor())
    logger.info(f"LR Metrics: {lr_metrics}")
    save_evaluation_report("LinearRegression", lr_metrics)
    mlflow_manager.log_run("Linear_Regression", lr_pipeline, {"fit_intercept": True}, lr_metrics, features)
    results["LinearRegression"] = (lr_pipeline, lr_metrics)

    # 3. Ridge Regression
    logger.info("--- Ridge Regression ---")
    ridge_params = {"alpha": 1.0}
    ridge_metrics, ridge_pipeline = scorer("ridge", Ridge(**ridge_params), X, y_log, build_preprocessor())
    logger.info(f"Ridge Metrics: {ridge_metrics}")
    save_evaluation_report("Ridge", ridge_metrics)
    mlflow_manager.log_run("Ridge", ridge_pipeline, ridge_params, ridge_metrics, features)
    results["Ridge"] = (ridge_pipeline, ridge_metrics)

    # 4. Random Forest
    logger.info("--- Random Forest Regressor ---")
    rf_params = {"n_estimators": 100, "random_state": 42, "max_depth": 10, "n_jobs": -1}
    rf_metrics, rf_pipeline = scorer("random_forest", RandomForestRegressor(**rf_params), X, y_log, build_preprocessor())
    logger.info(f"RF Metrics: {rf_metrics}")
    save_evaluation_report("RandomForest", rf_metrics)
    mlflow_manager.log_run("Random_Forest", rf_pipeline, rf_params, rf_metrics, features)
    results["RandomForest"] = (rf_pipeline, rf_metrics)

    # Model Selection — lowest RMSE on held-out original price scale
    best_name, best_pipeline, best_rmse = "", None, float("inf")
    for name, (pipeline, metrics) in results.items():
        if pipeline is not None and metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_pipeline = pipeline
            best_name = name

    logger.info(f"--- Best Model: {best_name} | RMSE: {best_rmse:.4f} ---")

    # Save model artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)
    logger.info(f"Saved best model to {model_path}")

    # Save metadata
    metadata = {
        "model_name": best_name,
        "features": features,
        "rmse": best_rmse,
        "target": "price (original scale, Crore INR)",
        "transform": "log1p applied during training; expm1 applied at inference",
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved model metadata to {metadata_path}")


if __name__ == "__main__":
    train_and_evaluate()
