from contextlib import contextmanager

try:
    import mariadb
    MARIADB_AVAILABLE = True
except ImportError:
    MARIADB_AVAILABLE = False

from src.config.settings import (
    MARIADB_DATABASE, MARIADB_HOST, MARIADB_PASSWORD, MARIADB_PORT, MARIADB_USER,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MariaDBClient:
    def __init__(self):
        self.config = {
            "host": MARIADB_HOST,
            "port": MARIADB_PORT,
            "user": MARIADB_USER,
            "password": MARIADB_PASSWORD,
            "database": MARIADB_DATABASE,
        }

    @contextmanager
    def get_connection(self):
        if not MARIADB_AVAILABLE:
            raise RuntimeError("mariadb package not installed. Install with: pip install mariadb")
        conn = mariadb.connect(**self.config)
        try:
            yield conn
        finally:
            conn.close()

    def init_tables(self):
        if not MARIADB_AVAILABLE:
            logger.warning("mariadb not installed; table init skipped.")
            return
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_properties (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    property_type VARCHAR(50), sector VARCHAR(100), price FLOAT,
                    bedRoom FLOAT, bathroom FLOAT, balcony VARCHAR(20),
                    agePossession VARCHAR(50), built_up_area FLOAT,
                    servant_room FLOAT, store_room FLOAT,
                    furnishing_type FLOAT, luxury_score VARCHAR(20)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_properties (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    property_type VARCHAR(50), sector VARCHAR(100),
                    price FLOAT, price_log FLOAT, bedRoom FLOAT, bathroom FLOAT,
                    balcony VARCHAR(20), agePossession VARCHAR(50), built_up_area FLOAT,
                    servant_room FLOAT, store_room FLOAT,
                    furnishing_type_label VARCHAR(20), luxury_score VARCHAR(20),
                    price_per_sqft FLOAT, bath_bed_ratio FLOAT
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    prediction_date DATE NOT NULL UNIQUE,
                    sector VARCHAR(100), property_type VARCHAR(50),
                    predicted_price FLOAT, model_version VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("Tables initialized.")
