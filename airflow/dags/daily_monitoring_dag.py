import json
import logging
from datetime import date, datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "retries": 1,
}


def generate_monitoring_summary():
    """Writes a simple monitoring summary JSON."""
    from src.config.settings import MONITORING_REPORTS_DIR, PROCESSED_DATA_DIR
    import pandas as pd

    logger = logging.getLogger(__name__)
    MONITORING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = PROCESSED_DATA_DIR / "properties_features.csv"
    summary = {"run_date": str(date.today()), "status": "ok"}

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        summary["total_rows"] = len(df)
        summary["avg_price"] = float(df["price"].mean()) if "price" in df.columns else None
        summary["null_counts"] = df.isnull().sum().to_dict()

    out_path = MONITORING_REPORTS_DIR / "monitoring_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)
    logger.info(f"Monitoring summary saved to {out_path}")


with DAG(
    dag_id="daily_monitoring_dag",
    default_args=default_args,
    schedule="0 19 * * *",  # Every day at 19:00
    start_date=datetime(2024, 1, 1),
    catchup=False,
    description="Daily: Refresh features and run drift monitoring.",
) as dag:

    ingest_properties = BashOperator(
        task_id="ingest_properties",
        bash_command="cd /opt/airflow && python -m src.ingestion.load_to_mariadb",
    )

    preprocess_properties = BashOperator(
        task_id="preprocess_properties",
        bash_command="cd /opt/airflow && python -m src.preprocessing.feature_engineering",
    )

    run_monitoring = BashOperator(
        task_id="run_evidently_monitoring",
        bash_command="cd /opt/airflow && python -m src.monitoring.evidently_report",
    )

    write_summary = PythonOperator(
        task_id="write_monitoring_summary",
        python_callable=generate_monitoring_summary,
    )

    ingest_properties >> preprocess_properties >> run_monitoring >> write_summary
