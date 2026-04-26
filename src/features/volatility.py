"""
Volatility and tail-risk features for the risk scoring model.

These features serve as covariates in the survival analysis model:
    - realized volatility (rolling log-return std)
    - downside deviation
    - max drawdown over window
    - funding rate z-score (external market stress signal)
"""

import numpy as np
import pandas as pd


def add_realized_vol(
    df: pd.DataFrame,
    windows: list[int] = [24, 72, 168],
) -> pd.DataFrame:
    """Annualized realized volatility at multiple lookback windows (hourly bars)."""
    df = df.copy()
    for w in windows:
        annualization = np.sqrt(8760 / w)
        df[f"realized_vol_{w}h"] = df["log_return"].rolling(w).std() * annualization
    return df


def add_downside_deviation(
    df: pd.DataFrame,
    windows: list[int] = [24, 72],
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Semi-deviation below threshold (Sortino ratio numerator input)."""
    df = df.copy()
    downside = df["log_return"].clip(upper=threshold)
    for w in windows:
        df[f"downside_dev_{w}h"] = downside.rolling(w).std() * np.sqrt(8760 / w)
    return df


def add_max_drawdown(df: pd.DataFrame, windows: list[int] = [24, 168]) -> pd.DataFrame:
    """Rolling max drawdown from peak within window."""
    df = df.copy()
    for w in windows:
        rolling_max = df["close"].rolling(w).max()
        df[f"max_drawdown_{w}h"] = (df["close"] - rolling_max) / rolling_max
    return df


def add_garman_klass_vol(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """
    Garman-Klass volatility estimator — more efficient than close-to-close
    because it incorporates high/low range information.
    """
    df = df.copy()
    log_hl = np.log(df["high"] / df["low"]) ** 2
    log_co = np.log(df["close"] / df["open"]) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    df[f"gk_vol_{window}h"] = gk.rolling(window).mean().apply(np.sqrt) * np.sqrt(8760)
    return df


def add_funding_features(
    df_klines: pd.DataFrame,
    funding_rates: dict[str, list],
) -> pd.DataFrame:
    """
    Merge funding rate statistics into the klines DataFrame.

    Args:
        df_klines:     Klines DataFrame with 'symbol' and 'open_time' columns.
        funding_rates: Dict of symbol -> list of FundingRateRecord dicts.
    """
    df = df_klines.copy()
    funding_rows = []
    for symbol, records in funding_rates.items():
        for rec in records:
            funding_rows.append(
                {"symbol": symbol, "funding_time": rec["funding_time"], "funding_rate": rec["funding_rate"]}
            )

    if not funding_rows:
        df["funding_rate_mean"] = np.nan
        df["funding_rate_z"] = np.nan
        return df

    df_fund = pd.DataFrame(funding_rows)
    df_fund["funding_time"] = pd.to_datetime(df_fund["funding_time"])
    df_fund = df_fund.sort_values("funding_time")

    # Rolling stats per symbol
    df_fund["funding_rate_mean"] = df_fund.groupby("symbol")["funding_rate"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )
    df_fund["funding_rate_std"] = df_fund.groupby("symbol")["funding_rate"].transform(
        lambda x: x.rolling(30, min_periods=1).std()
    )
    df_fund["funding_rate_z"] = (
        df_fund["funding_rate"] - df_fund["funding_rate_mean"]
    ) / df_fund["funding_rate_std"].replace(0, np.nan)

    # Merge-asof to align funding timestamps to kline open_time
    df["open_time"] = pd.to_datetime(df["open_time"])
    merged = pd.merge_asof(
        df.sort_values("open_time"),
        df_fund[["symbol", "funding_time", "funding_rate_mean", "funding_rate_z"]].sort_values("funding_time"),
        left_on="open_time",
        right_on="funding_time",
        by="symbol",
        direction="backward",
    )
    return merged.sort_values(["symbol", "open_time"]).reset_index(drop=True)


def build_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_realized_vol(df)
    df = add_downside_deviation(df)
    df = add_max_drawdown(df)
    df = add_garman_klass_vol(df)
    return df
