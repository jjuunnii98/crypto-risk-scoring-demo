"""
RiskScorer — end-to-end inference pipeline.

Collects live data → builds features → scores symbols.
Used by the FastAPI endpoints.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

import structlog

from src.ingestion.collector import DataCollector
from src.features.pipeline import build_features, get_feature_matrix
from src.models.survival import CryptoSurvivalModel

logger = structlog.get_logger(__name__)


@dataclass
class RiskResult:
    symbol: str
    risk_score: float      # 0–100, higher = higher risk
    last_price: float
    spread_bps: float
    depth_imbalance: float
    realized_vol_24h: float | None
    funding_rate_z: float | None
    scored_at: datetime


class RiskScorer:
    """Runs the full pipeline: collect → featurize → score."""

    def __init__(self, model: CryptoSurvivalModel) -> None:
        self._model = model

    async def score_symbols(self, symbols: list[str]) -> list[RiskResult]:
        async with DataCollector() as collector:
            snapshots = await collector.collect_all(symbols)

        if not snapshots:
            return []

        funding_rates = {
            snap.symbol: [f.model_dump() for f in snap.funding_rates]
            for snap in snapshots
        }

        df_klines = DataCollector.snapshots_to_kline_df(snapshots)
        df_features = build_features(df_klines, funding_rates)

        latest = (
            df_features
            .sort_values("open_time")
            .groupby("symbol")
            .last()
            .reset_index()
        )

        X = get_feature_matrix(latest)
        scores = self._model.predict_risk_score(X)
        # reindex to latest — fillna(50) for any rows that couldn't be scored
        latest["risk_score"] = scores.reindex(latest.index).fillna(50.0)

        ticker_map = {snap.symbol: snap.ticker for snap in snapshots}
        book_map = {snap.symbol: snap.order_book for snap in snapshots}

        results = []
        for _, row in latest.iterrows():
            sym = row["symbol"]
            ticker = ticker_map.get(sym)
            book = book_map.get(sym)
            results.append(
                RiskResult(
                    symbol=sym,
                    risk_score=round(float(row.get("risk_score", 50.0)), 2),
                    last_price=ticker.last_price if ticker else float("nan"),
                    spread_bps=round(book.spread_bps, 3) if book else float("nan"),
                    depth_imbalance=round(book.depth_imbalance, 4) if book else float("nan"),
                    realized_vol_24h=row.get("realized_vol_24h"),
                    funding_rate_z=row.get("funding_rate_z"),
                    scored_at=datetime.now(timezone.utc),
                )
            )

        logger.info("scored", symbols=[r.symbol for r in results])
        return results
