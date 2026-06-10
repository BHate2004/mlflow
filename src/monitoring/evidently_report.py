"""
Data Drift Monitoring using Evidently AI
-----------------------------------------
Compares current processed data against a reference snapshot.
Saves HTML reports to reports/monitoring/.
"""
import pandas as pd

from src.config.settings import MONITORING_REPORTS_DIR, PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_monitoring():
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    except ImportError:
        logger.warning("Evidently not installed. Skipping drift monitoring.")
        return

    csv_path = PROCESSED_DATA_DIR / "properties_features.csv"
    if not csv_path.exists():
        logger.warning("Processed data not found. Run preprocessing first.")
        return

    df = pd.read_csv(csv_path)

    # Use first 70% as reference, remaining 30% as current
    split = int(len(df) * 0.7)
    reference_df = df.iloc[:split].reset_index(drop=True)
    current_df = df.iloc[split:].reset_index(drop=True)

    # Numeric columns only for Evidently
    num_cols = ["price", "bedRoom", "bathroom", "built_up_area",
                "servant room", "store room", "price_per_sqft", "bath_bed_ratio"]
    num_cols = [c for c in num_cols if c in df.columns]

    ref = reference_df[num_cols]
    curr = current_df[num_cols]

    MONITORING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Data drift report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=ref, current_data=curr)
    drift_path = MONITORING_REPORTS_DIR / "properties_data_drift_report.html"
    drift_report.save_html(str(drift_path))
    logger.info(f"Drift report saved to {drift_path}")

    # Data quality report
    quality_report = Report(metrics=[DataQualityPreset()])
    quality_report.run(reference_data=ref, current_data=curr)
    quality_path = MONITORING_REPORTS_DIR / "properties_data_quality_report.html"
    quality_report.save_html(str(quality_path))
    logger.info(f"Quality report saved to {quality_path}")


if __name__ == "__main__":
    run_monitoring()
