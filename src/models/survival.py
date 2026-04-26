"""
Survival analysis model for crypto market risk.

Treats each symbol's price trajectory as a survival process where
the "event" is a significant drawdown (e.g., >5% drop within 24h).
Cox Proportional Hazards is fit on the feature matrix.

Reference: lifelines CoxPHFitter
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
import structlog

from src.features.pipeline import FEATURE_COLS

logger = structlog.get_logger(__name__)

EVENT_THRESHOLD = -0.03   # -3% cumulative drawdown from current price → "event"
DURATION_WINDOW = 24      # maximum look-ahead window (hours)


def label_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert panel OHLCV data into proper survival analysis format.

    Event definition:
        The asset's log return from time t first falls below EVENT_THRESHOLD
        (-3%) within the next DURATION_WINDOW (24h) bars.

    Labels:
        duration  — hours until the first threshold crossing.
                    Right-censored at DURATION_WINDOW if no crossing occurs.
        event     — 1 if crossing observed, 0 if censored.

    Why not duration=1?
        Cox PH needs variation in observed times to rank concordant pairs.
        Using time-to-event (1 … 24h) provides that variation.

    Event rate empirically ~7–10% on 21-day hourly data, suitable for Cox PH.
    """
    results = []
    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("open_time").reset_index(drop=True)
        n = len(group)
        close = group["close"].values

        durations = np.empty(n, dtype=float)
        events = np.zeros(n, dtype=int)

        for i in range(n):
            future_close = close[i + 1 : i + DURATION_WINDOW + 1]
            if future_close.size == 0:
                durations[i] = float(DURATION_WINDOW)
                continue
            # Cumulative log return from current price to each future bar
            running_return = np.log(future_close / close[i])
            below = running_return < EVENT_THRESHOLD
            if below.any():
                durations[i] = float(np.argmax(below) + 1)
                events[i] = 1
            else:
                durations[i] = float(DURATION_WINDOW)  # right-censored

        group["duration"] = durations
        group["event"] = events
        results.append(group)

    return pd.concat(results, ignore_index=True).dropna(subset=["log_return"])


class CryptoSurvivalModel:
    """Cox PH risk model wrapper."""

    def __init__(self, penalizer: float = 0.1) -> None:
        self.penalizer = penalizer
        self._model = CoxPHFitter(penalizer=penalizer)
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "CryptoSurvivalModel":
        df_labeled = label_events(df)
        feature_cols = [c for c in FEATURE_COLS if c in df_labeled.columns]
        train_df = df_labeled[feature_cols + ["duration", "event"]].dropna()

        logger.info("model_fit_start", rows=len(train_df), features=len(feature_cols))
        self._model.fit(train_df, duration_col="duration", event_col="event")
        self._fitted = True
        logger.info("model_fit_done", concordance=round(self._model.concordance_index_, 4))
        return self

    def predict_risk_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Returns partial hazard (higher = higher risk).
        Scaled to [0, 100] across the provided sample.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before scoring.")

        feature_cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[feature_cols].dropna()
        raw = self._model.predict_partial_hazard(X)

        # Normalize to [0, 100]
        lo, hi = raw.min(), raw.max()
        if hi == lo:
            return pd.Series(50.0, index=raw.index)
        return ((raw - lo) / (hi - lo) * 100).rename("risk_score")

    def summary(self) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        return self._model.summary

    @property
    def concordance_index(self) -> float:
        if not self._fitted:
            return float("nan")
        return self._model.concordance_index_
