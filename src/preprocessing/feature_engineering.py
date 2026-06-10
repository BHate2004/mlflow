import pandas as pd
import numpy as np

from src.config.settings import (
    DROP_COLS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    TARGET_COL,
)
from src.storage.mariadb_client import MariaDBClient
from src.storage.redis_client import RedisClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


def categorize_furnishing(x) -> str:
    """Maps numeric furnishing codes to labels."""
    if x == 0:
        return "unfurnished"
    elif x == 1:
        return "semiunfurnished"
    else:
        return "furnished"


def get_raw_data() -> pd.DataFrame:
    """Load raw data: MariaDB → CSV fallback."""
    client = MariaDBClient()
    try:
        with client.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM raw_properties")
            rows = cursor.fetchall()
            if rows:
                columns = [desc[0] for desc in cursor.description]
                df = pd.DataFrame(rows, columns=columns)
                # Rename DB columns back to original names
                df = df.rename(columns={
                    "servant_room": "servant room",
                    "store_room": "store room",
                })
                logger.info(f"Loaded {len(df)} rows from MariaDB.")
                return df
    except Exception as e:
        logger.warning(f"MariaDB load failed: {e}. Falling back to CSV.")

    csv_path = RAW_DATA_DIR / "gurgaon_properties.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV fallback.")
        return df

    raise FileNotFoundError("Could not load raw data from MariaDB or CSV.")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all feature engineering steps matching model.ipynb pipeline."""
    initial_rows = len(df)

    # 1. Drop unused columns
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_existing)

    # 2. Convert furnishing type to readable label
    if df["furnishing_type"].dtype != object:
        df["furnishing_type"] = df["furnishing_type"].apply(categorize_furnishing)

    # 3. Derived features
    df["price_per_sqft"] = df[TARGET_COL] / df["built_up_area"].replace(0, np.nan)
    df["bath_bed_ratio"] = df["bathroom"] / df["bedRoom"].replace(0, np.nan)

    # 4. Log-transform target to reduce skew
    df["price_log"] = np.log1p(df[TARGET_COL])

    # 5. Drop rows missing key fields
    key_cols = ["property_type", "sector", TARGET_COL, "bedRoom", "bathroom",
                "agePossession", "built_up_area", "furnishing_type", "luxury_score"]
    df = df.dropna(subset=key_cols).reset_index(drop=True)

    final_rows = len(df)
    logger.info(f"Feature engineering done. Rows before: {initial_rows} | after: {final_rows} | dropped: {initial_rows - final_rows}")
    return df


def save_to_mariadb(df: pd.DataFrame):
    client = MariaDBClient()
    records = []
    for _, row in df.iterrows():
        records.append((
            row.get("property_type"),
            row.get("sector"),
            float(row.get(TARGET_COL, 0)),
            float(row.get("price_log", 0)),
            row.get("bedRoom"),
            row.get("bathroom"),
            str(row.get("balcony")) if row.get("balcony") is not None else None,
            row.get("agePossession"),
            row.get("built_up_area"),
            row.get("servant room"),
            row.get("store room"),
            row.get("furnishing_type"),
            row.get("luxury_score"),
            row.get("price_per_sqft"),
            row.get("bath_bed_ratio"),
        ))

    query = """
        INSERT INTO processed_properties (
            property_type, sector, price, price_log, bedRoom, bathroom,
            balcony, agePossession, built_up_area, servant_room, store_room,
            furnishing_type_label, luxury_score, price_per_sqft, bath_bed_ratio
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        with client.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("TRUNCATE TABLE processed_properties")
            cursor.executemany(query, records)
            conn.commit()
            logger.info(f"Saved {len(records)} processed rows to MariaDB.")
    except Exception as e:
        logger.warning(f"MariaDB save failed (non-fatal): {e}")


def preprocess():
    df = get_raw_data()
    df = engineer_features(df)

    # Save to CSV
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROCESSED_DATA_DIR / "properties_features.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Processed data saved to {csv_path}")

    # Save to MariaDB
    try:
        save_to_mariadb(df)
    except Exception as e:
        logger.warning(f"MariaDB persistence skipped: {e}")

    # Cache in Redis
    try:
        redis_client = RedisClient()
        redis_client.cache_dataframe("processed_properties", df)
        logger.info("Processed data cached in Redis.")
    except Exception as e:
        logger.warning(f"Redis cache failed (non-fatal): {e}")


if __name__ == "__main__":
    preprocess()
