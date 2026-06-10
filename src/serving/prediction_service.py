import json
import logging
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self):
        self.model_path = MODELS_DIR / "best_model.joblib"
        self.metadata_path = MODELS_DIR / "model_metadata.json"
        self.model = None
        self.metadata = None
        self.features = []
        self.model_name = "unknown"
        self._load_model_artifacts()

    def _load_model_artifacts(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                "Model file missing. Run: python -m src.training.train_model"
            )
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                "Model metadata missing. Run: python -m src.training.train_model"
            )
        self.model = joblib.load(self.model_path)
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.features = self.metadata.get("features", [])
        self.model_name = self.metadata.get("model_name", "unknown")
        logger.info(f"Loaded model '{self.model_name}' with {len(self.features)} features.")

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts a dict of property features and returns predicted price.
        Price is predicted in log1p space then back-transformed to Crore INR.
        """
        try:
            row = pd.DataFrame([input_data])
            # Ensure feature order matches training
            for feat in self.features:
                if feat not in row.columns:
                    row[feat] = np.nan

            X = row[self.features]
            predicted_log = float(self.model.predict(X)[0])
            predicted_price = float(np.expm1(predicted_log))

            return {
                "predicted_price_cr": round(predicted_price, 4),
                "model_version": self.model_name,
                "input_features": input_data,
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
