import io

import pandas as pd
import redis

from src.config.settings import REDIS_HOST, REDIS_PORT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RedisClient:
    def __init__(self):
        try:
            self.client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=False,
                socket_connect_timeout=2,
            )
            self.client.ping()
            self._available = True
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            self._available = False
            logger.warning(f"Redis unavailable: {e}. Caching will be skipped.")

    def cache_dataframe(self, key: str, df: pd.DataFrame, ttl: int = 86400):
        if not self._available:
            return
        try:
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            self.client.set(key, buf.getvalue(), ex=ttl)
            logger.info(f"Cached DataFrame '{key}' in Redis ({len(df)} rows).")
        except Exception as e:
            logger.warning(f"Redis cache write failed: {e}")

    def get_cached_dataframe(self, key: str):
        if not self._available:
            return None
        try:
            raw = self.client.get(key)
            if raw:
                df = pd.read_parquet(io.BytesIO(raw))
                logger.info(f"Cache hit for '{key}' — {len(df)} rows.")
                return df
        except Exception as e:
            logger.warning(f"Redis cache read failed: {e}")
        return None
