"""
DataCollector — orchestrates parallel data collection across symbols.

Fetches OHLCV klines, 24hr ticker, order book depth, and futures funding
rates for each target symbol, validates the raw API responses into typed
schemas, and optionally persists them to Parquet.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import structlog

from config.settings import settings
from .binance_client import BinanceClient
from .schemas import (
    FundingRateRecord,
    KlineRecord,
    OrderBookLevel,
    OrderBookSnapshot,
    SymbolSnapshot,
    Ticker24hr,
)

logger = structlog.get_logger(__name__)

# Kline array positional indices (Binance spec)
_K_OPEN_TIME = 0
_K_OPEN = 1
_K_HIGH = 2
_K_LOW = 3
_K_CLOSE = 4
_K_VOLUME = 5
_K_CLOSE_TIME = 6
_K_QUOTE_VOL = 7
_K_NUM_TRADES = 8
_K_TAKER_BASE = 9
_K_TAKER_QUOTE = 10


class DataCollector:
    """Collects and validates market data for a list of symbols."""

    def __init__(self, client: BinanceClient | None = None) -> None:
        self._client = client
        self._owned = client is None

    async def __aenter__(self) -> "DataCollector":
        if self._owned:
            self._client = BinanceClient()
        return self

    async def __aexit__(self, *_) -> None:
        if self._owned and self._client:
            await self._client.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def collect_all(
        self,
        symbols: list[str] | None = None,
        save: bool = False,
    ) -> list[SymbolSnapshot]:
        """
        Fetch data for all symbols concurrently and return validated snapshots.

        Args:
            symbols: Override the default symbol list from settings.
            save:    Persist each snapshot to Parquet under data/raw/.
        """
        targets = symbols or settings.symbols
        logger.info("collect_all_start", symbols=targets)

        tasks = [self._collect_symbol(sym) for sym in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        snapshots: list[SymbolSnapshot] = []
        for symbol, result in zip(targets, results):
            if isinstance(result, Exception):
                logger.error("collect_symbol_failed", symbol=symbol, error=str(result))
            else:
                snapshots.append(result)
                if save:
                    self._save_snapshot(result)

        logger.info("collect_all_done", total=len(snapshots), failed=len(targets) - len(snapshots))
        return snapshots

    async def collect_symbol(self, symbol: str, save: bool = False) -> SymbolSnapshot:
        """Collect and validate data for a single symbol."""
        snapshot = await self._collect_symbol(symbol)
        if save:
            self._save_snapshot(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # Internal collection logic
    # ------------------------------------------------------------------

    async def _collect_symbol(self, symbol: str) -> SymbolSnapshot:
        log = logger.bind(symbol=symbol)
        log.info("collecting")

        klines_raw, ticker_raw, book_raw, funding_raw = await asyncio.gather(
            self._client.get_klines(symbol),
            self._client.get_ticker_24hr(symbol),
            self._client.get_order_book(symbol),
            self._safe_funding(symbol),
        )

        klines = self._parse_klines(symbol, klines_raw)
        ticker_raw.setdefault("symbol", symbol)
        ticker = Ticker24hr(**ticker_raw)
        order_book = self._parse_order_book(symbol, book_raw)
        funding = self._parse_funding(symbol, funding_raw)

        snapshot = SymbolSnapshot(
            symbol=symbol,
            collected_at=datetime.now(timezone.utc),
            klines=klines,
            ticker=ticker,
            order_book=order_book,
            funding_rates=funding,
        )
        log.info(
            "collected",
            klines=len(klines),
            spread_bps=round(order_book.spread_bps, 3),
            depth_imbalance=round(order_book.depth_imbalance, 3),
        )
        return snapshot

    async def _safe_funding(self, symbol: str) -> list[dict]:
        """Funding rates are only available for futures pairs; swallow errors gracefully."""
        try:
            return await self._client.get_funding_rate(symbol)
        except Exception as exc:
            logger.warning("funding_rate_unavailable", symbol=symbol, error=str(exc))
            return []

    # ------------------------------------------------------------------
    # Parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_klines(symbol: str, raw: list[list]) -> list[KlineRecord]:
        records = []
        for row in raw:
            records.append(
                KlineRecord(
                    symbol=symbol,
                    open_time=datetime.utcfromtimestamp(row[_K_OPEN_TIME] / 1000),
                    close_time=datetime.utcfromtimestamp(row[_K_CLOSE_TIME] / 1000),
                    open=row[_K_OPEN],
                    high=row[_K_HIGH],
                    low=row[_K_LOW],
                    close=row[_K_CLOSE],
                    volume=row[_K_VOLUME],
                    quote_volume=row[_K_QUOTE_VOL],
                    num_trades=int(row[_K_NUM_TRADES]),
                    taker_buy_base_volume=row[_K_TAKER_BASE],
                    taker_buy_quote_volume=row[_K_TAKER_QUOTE],
                )
            )
        return records

    @staticmethod
    def _parse_order_book(symbol: str, raw: dict) -> OrderBookSnapshot:
        bids = [OrderBookLevel(price=float(p), quantity=float(q)) for p, q in raw["bids"]]
        asks = [OrderBookLevel(price=float(p), quantity=float(q)) for p, q in raw["asks"]]
        return OrderBookSnapshot(
            symbol=symbol,
            last_update_id=raw["lastUpdateId"],
            bids=bids,
            asks=asks,
        )

    @staticmethod
    def _parse_funding(symbol: str, raw: list[dict]) -> list[FundingRateRecord]:
        records = []
        for entry in raw:
            entry.setdefault("symbol", symbol)
            try:
                records.append(FundingRateRecord(**entry))
            except Exception:
                pass
        return records

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _save_snapshot(snapshot: SymbolSnapshot) -> None:
        base = Path(settings.raw_data_dir)
        date_str = snapshot.collected_at.strftime("%Y%m%d")

        klines_path = base / "klines" / snapshot.symbol / f"{date_str}.parquet"
        klines_path.parent.mkdir(parents=True, exist_ok=True)
        df_klines = pd.DataFrame([k.model_dump() for k in snapshot.klines])
        df_klines.to_parquet(klines_path, index=False)

        if snapshot.funding_rates:
            funding_path = base / "funding" / snapshot.symbol / f"{date_str}.parquet"
            funding_path.parent.mkdir(parents=True, exist_ok=True)
            df_funding = pd.DataFrame([f.model_dump() for f in snapshot.funding_rates])
            df_funding.to_parquet(funding_path, index=False)

        logger.info("snapshot_saved", symbol=snapshot.symbol, date=date_str)

    # ------------------------------------------------------------------
    # DataFrame helpers
    # ------------------------------------------------------------------

    @staticmethod
    def snapshots_to_kline_df(snapshots: list[SymbolSnapshot]) -> pd.DataFrame:
        """Flatten multiple snapshots into one tidy DataFrame for feature engineering."""
        rows = []
        for snap in snapshots:
            for k in snap.klines:
                rows.append(k.model_dump())
        return pd.DataFrame(rows).sort_values(["symbol", "open_time"]).reset_index(drop=True)

    @staticmethod
    def snapshots_to_ticker_df(snapshots: list[SymbolSnapshot]) -> pd.DataFrame:
        rows = [snap.ticker.model_dump() for snap in snapshots]
        return pd.DataFrame(rows)
