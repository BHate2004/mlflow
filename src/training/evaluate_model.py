import json
import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config.settings import EVALUATION_REPORTS_DIR

logger = logging.getLogger(__name__)


def evaluate_predictions(y_true, y_pred) -> dict:
    """
    Calculates regression metrics on original (non-log) price scale.
    Inputs y_true / y_pred should already be expm1-transformed back.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")

    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": float(r2),
        "MAPE": mape,
    }


def save_evaluation_report(model_name: str, metrics: dict):
    """Saves metrics dict as JSON under reports/evaluation/."""
    EVALUATION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EVALUATION_REPORTS_DIR / f"{model_name}_metrics.json"
    try:
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved evaluation metrics to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation metrics: {e}")
