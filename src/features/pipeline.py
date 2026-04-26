"""
Feature pipeline — combines technical and volatility features
and prepares the design matrix for the survival model.
"""

import pandas as pd
import structlog

from .technical import build_feature_matrix
from .volatility import build_volatility_features, add_funding_features

logger = structlog.get_logger(__name__)

FEATURE_COLS = [
    "log_return",
    "rsi_14",
    "bb_width_20",
    "bb_pct_20",
    "macd_hist",
    "atr_pct_14",
    "volume_ratio",
    "vwap_divergence",
    "buy_sell_ratio",
    "realized_vol_24h",
    "realized_vol_168h",
    "downside_dev_24h",
    "max_drawdown_24h",
    "max_drawdown_168h",
    "gk_vol_24h",
    "funding_rate_z",
]


def build_features(
    df_klines: pd.DataFrame,
    funding_rates: dict[str, list] | None = None,
) -> pd.DataFrame:
    """
    Full feature pipeline from raw klines to model-ready matrix.

    Args:
        df_klines:     Output of DataCollector.snapshots_to_kline_df().
        funding_rates: Dict of symbol -> list[FundingRateRecord dicts].

    Returns:
        DataFrame with all feature columns; rows with NaNs dropped.
    """
    results = []
    for symbol, group in df_klines.groupby("symbol"):
        group = group.sort_values("open_time").copy()
        group = build_feature_matrix(group)
        group = build_volatility_features(group)
        results.append(group)

    df = pd.concat(results, ignore_index=True)

    if funding_rates:
        df = add_funding_features(df, funding_rates)
    else:
        df["funding_rate_z"] = float("nan")

    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        logger.warning("features_missing", cols=list(missing))

    df = df.dropna(subset=available)
    logger.info("features_built", rows=len(df), features=len(available))
    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the feature columns used by the model."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    return df[cols]
