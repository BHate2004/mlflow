import numpy as np
import pandas as pd

from src.config.settings import MARIADB_DATABASE, MARIADB_HOST, MARIADB_PASSWORD, MARIADB_PORT, MARIADB_USER, RAW_DATA_DIR
from src.storage.mariadb_client import MariaDBClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_CSV = RAW_DATA_DIR / "gurgaon_properties.csv"
RAW_TABLE = "raw_properties"


def load_raw_csv() -> pd.DataFrame:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    logger.info(f"Loaded raw CSV. Shape: {df.shape}")
    return df


def ingest_to_mariadb(df: pd.DataFrame):
    """Inserts raw properties data into MariaDB raw_properties table."""
    client = MariaDBClient()
    client.init_tables()

    # Clean for DB insert
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(pd.notnull(df), None)

    records = []
    for _, row in df.iterrows():
        records.append((
            row.get("property_type"),
            row.get("sector"),
            row.get("price"),
            row.get("bedRoom"),
            row.get("bathroom"),
            str(row.get("balcony")) if row.get("balcony") is not None else None,
            row.get("agePossession"),
            row.get("built_up_area"),
            row.get("servant room"),
            row.get("store room"),
            row.get("furnishing_type"),
            row.get("luxury_score"),
        ))

    insert_query = f"""
        INSERT INTO {RAW_TABLE} (
            property_type, sector, price, bedRoom, bathroom,
            balcony, agePossession, built_up_area,
            servant_room, store_room, furnishing_type, luxury_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    try:
        with client.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"TRUNCATE TABLE {RAW_TABLE}")
            cursor.executemany(insert_query, records)
            conn.commit()
            logger.info(f"Ingested {len(records)} rows into '{RAW_TABLE}'.")
    except Exception as e:
        logger.error(f"MariaDB ingestion failed: {e}")
        raise


def ingest():
    df = load_raw_csv()
    try:
        ingest_to_mariadb(df)
    except Exception as e:
        logger.warning(f"DB ingestion failed ({e}). Saving to CSV fallback only.")
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_DIR / "gurgaon_properties.csv", index=False)
    logger.info("Data ingestion completed.")


if __name__ == "__main__":
    ingest()
