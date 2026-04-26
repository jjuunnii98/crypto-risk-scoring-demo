"""
Technical indicator features derived from OHLCV klines.

All functions accept a DataFrame with columns:
    open, high, low, close, volume, num_trades,
    taker_buy_base_volume, taker_buy_quote_volume
and return the same DataFrame with new feature columns appended.
"""

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["pct_return"] = df["close"].pct_change()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    ma = df["close"].rolling(period).mean()
    sigma = df["close"].rolling(period).std()
    df[f"bb_upper_{period}"] = ma + std * sigma
    df[f"bb_lower_{period}"] = ma - std * sigma
    df[f"bb_mid_{period}"] = ma
    df[f"bb_width_{period}"] = (std * 2 * sigma) / ma  # normalized width
    df[f"bb_pct_{period}"] = (df["close"] - df[f"bb_lower_{period}"]) / (
        df[f"bb_upper_{period}"] - df[f"bb_lower_{period}"]
    )
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range — baseline volatility/liquidity proxy."""
    df = df.copy()
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df[f"atr_{period}"] = tr.ewm(span=period, adjust=False).mean()
    df[f"atr_pct_{period}"] = df[f"atr_{period}"] / df["close"]
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-based features including buy/sell pressure and VWAP."""
    df = df.copy()
    df["taker_sell_volume"] = df["volume"] - df["taker_buy_base_volume"]
    df["buy_sell_ratio"] = df["taker_buy_base_volume"] / df["taker_sell_volume"].replace(0, np.nan)
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]

    # Approximate VWAP over rolling window
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap_20"] = (typical_price * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    df["vwap_divergence"] = (df["close"] - df["vwap_20"]) / df["vwap_20"]
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all technical indicators in sequence."""
    df = add_returns(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    df = add_macd(df)
    df = add_atr(df)
    df = add_volume_features(df)
    return df
