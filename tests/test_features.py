"""
Unit tests for the feature engineering pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.technical import (
    add_returns,
    add_rsi,
    add_bollinger_bands,
    add_macd,
    add_atr,
    add_volume_features,
    build_feature_matrix,
)
from src.features.volatility import (
    add_realized_vol,
    add_garman_klass_vol,
    add_max_drawdown,
)


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 40_000 + np.cumsum(rng.normal(0, 200, n))
    noise = rng.uniform(0.001, 0.005, n)
    volume = rng.uniform(100, 500, n)
    # taker_buy must be strictly less than total volume so sell_vol is always positive
    taker_buy = volume * rng.uniform(0.1, 0.9, n)
    return pd.DataFrame({
        "open": close * (1 - noise / 2),
        "high": close * (1 + noise),
        "low": close * (1 - noise),
        "close": close,
        "volume": volume,
        "taker_buy_base_volume": taker_buy,
        "taker_buy_quote_volume": taker_buy * close,
        "num_trades": rng.integers(500, 2000, n),
    })


class TestReturnFeatures:
    def test_log_return_first_row_nan(self):
        df = add_returns(_make_ohlcv(50))
        assert np.isnan(df["log_return"].iloc[0])

    def test_log_return_shape(self):
        df = add_returns(_make_ohlcv(50))
        assert len(df) == 50
        assert "log_return" in df.columns
        assert "pct_return" in df.columns

    def test_log_return_close_to_pct_for_small_changes(self):
        df = add_returns(_make_ohlcv(100))
        r = df.dropna()
        # For small returns: log_return ≈ pct_return
        assert (abs(r["log_return"] - r["pct_return"]) < 0.05).all()


class TestRSI:
    def test_rsi_range(self):
        df = add_rsi(_make_ohlcv(100))
        rsi = df["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_rsi_warmup(self):
        df = add_rsi(_make_ohlcv(100), period=14)
        assert df["rsi_14"].iloc[:14].isna().all()

    def test_rsi_constant_price(self):
        df = _make_ohlcv(50)
        df["close"] = 42000.0
        df = add_rsi(df)
        # Constant price → all gains or losses → RSI should be 100 or 50
        valid = df["rsi_14"].dropna()
        assert ((valid == 100) | (valid == 50) | valid.isna()).all()


class TestBollingerBands:
    def test_bb_upper_above_lower(self):
        df = add_bollinger_bands(_make_ohlcv(100))
        valid = df.dropna(subset=["bb_upper_20", "bb_lower_20"])
        assert (valid["bb_upper_20"] >= valid["bb_lower_20"]).all()

    def test_bb_width_positive(self):
        df = add_bollinger_bands(_make_ohlcv(100))
        valid = df["bb_width_20"].dropna()
        assert (valid >= 0).all()

    def test_bb_mid_equals_rolling_mean(self):
        df = add_bollinger_bands(_make_ohlcv(100))
        expected = df["close"].rolling(20).mean()
        pd.testing.assert_series_equal(df["bb_mid_20"], expected, check_names=False)


class TestMACD:
    def test_macd_columns_present(self):
        df = add_macd(_make_ohlcv(100))
        for col in ["macd", "macd_signal", "macd_hist"]:
            assert col in df.columns

    def test_macd_hist_equals_diff(self):
        df = add_macd(_make_ohlcv(100))
        valid = df.dropna(subset=["macd", "macd_signal"])
        diff = (valid["macd"] - valid["macd_signal"]).round(8)
        hist = valid["macd_hist"].round(8)
        pd.testing.assert_series_equal(diff, hist, check_names=False)


class TestATR:
    def test_atr_positive(self):
        df = add_atr(_make_ohlcv(100))
        assert (df["atr_14"].dropna() > 0).all()

    def test_atr_pct_reasonable_range(self):
        df = add_atr(_make_ohlcv(100))
        valid = df["atr_pct_14"].dropna()
        # ATR % for crypto typically 0.1%–5% hourly
        assert (valid > 0).all() and (valid < 0.5).all()


class TestVolumeFeatures:
    def test_buy_sell_ratio_positive(self):
        df = add_volume_features(_make_ohlcv(100))
        valid = df["buy_sell_ratio"].dropna()
        assert (valid > 0).all()

    def test_volume_ratio_columns_present(self):
        df = add_volume_features(_make_ohlcv(100))
        for col in ["volume_ratio", "vwap_20", "vwap_divergence"]:
            assert col in df.columns


class TestVolatilityFeatures:
    def test_realized_vol_non_negative(self):
        df = add_returns(_make_ohlcv(200))
        df = add_realized_vol(df)
        for col in ["realized_vol_24h", "realized_vol_72h", "realized_vol_168h"]:
            assert (df[col].dropna() >= 0).all()

    def test_gk_vol_non_negative(self):
        df = add_garman_klass_vol(_make_ohlcv(200))
        assert (df["gk_vol_24h"].dropna() >= 0).all()

    def test_max_drawdown_non_positive(self):
        df = add_max_drawdown(_make_ohlcv(200))
        assert (df["max_drawdown_24h"].dropna() <= 0).all()

    def test_max_drawdown_bounded(self):
        df = add_max_drawdown(_make_ohlcv(200))
        valid = df["max_drawdown_24h"].dropna()
        assert (valid >= -1).all()


class TestBuildFeatureMatrix:
    def test_all_feature_cols_present(self):
        df = build_feature_matrix(_make_ohlcv(300))
        expected = [
            "log_return", "rsi_14", "bb_width_20", "macd_hist",
            "atr_14", "volume_ratio", "vwap_20",
        ]
        for col in expected:
            assert col in df.columns, f"Missing: {col}"

    def test_output_shape_preserved(self):
        raw = _make_ohlcv(300)
        df = build_feature_matrix(raw)
        assert len(df) == len(raw)
