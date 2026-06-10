import logging
import mlflow
import requests
from src.config.settings import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)

class MLflowManager:
    def __init__(self, experiment_name: str = "RealEstate_Price_Prediction"):
        self.tracking_uri = MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name
        self.is_available = self._check_availability()
        if self.is_available:
            mlflow.set_tracking_uri(self.tracking_uri)
            try:
                mlflow.set_experiment(self.experiment_name)
                logger.info(f"Connected to MLflow at {self.tracking_uri}. Experiment: {self.experiment_name}")
            except Exception as e:
                logger.warning(f"MLflow accessible but experiment setup failed: {e}")
                self.is_available = False

    def _check_availability(self) -> bool:
        try:
            response = requests.get(f"{self.tracking_uri}/health", timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            logger.warning(f"MLflow unreachable at {self.tracking_uri}. Logging will be skipped.")
            return False

    def log_run(self, run_name: str, model, params: dict, metrics: dict, features: list):
        if not self.is_available:
            logger.info(f"Skipping MLflow logging for '{run_name}' (unavailable).")
            return
        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_param("features", ", ".join(features))
                logger.info(f"Logged run '{run_name}' to MLflow.")
        except Exception as e:
            logger.error(f"Failed to log run to MLflow: {e}")
