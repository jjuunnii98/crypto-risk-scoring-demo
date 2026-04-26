from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Binance REST API
    binance_base_url: str = "https://api.binance.com"
    binance_futures_url: str = "https://fapi.binance.com"

    # Target symbols for risk scoring
    symbols: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
    )

    # Kline (OHLCV) settings
    kline_interval: str = "1h"       # 1m 3m 5m 15m 30m 1h 4h 1d
    kline_lookback: int = 500        # number of candles to fetch

    # Order book depth
    order_book_limit: int = 20       # 5 | 10 | 20 | 50 | 100 | 500 | 1000

    # HTTP client
    request_timeout: float = 10.0
    max_retries: int = 3
    retry_wait_min: float = 1.0
    retry_wait_max: float = 8.0

    # Data storage
    raw_data_dir: str = "data/raw"


settings = Settings()
