"""
Unit tests for the ingestion layer.

Uses httpx.MockTransport to avoid real network calls.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from src.ingestion.collector import DataCollector
from src.ingestion.schemas import KlineRecord, OrderBookSnapshot, Ticker24hr


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _make_kline_row(ts: int = 1_700_000_000_000) -> list:
    return [
        ts, "42000.00", "42500.00", "41800.00", "42200.00",
        "100.50", ts + 3_599_999, "4231000.00", 1500,
        "60.30", "2538600.00", "0",
    ]


MOCK_TICKER = {
    "symbol": "BTCUSDT",
    "priceChange": "1200.00",
    "priceChangePercent": "2.93",
    "weightedAvgPrice": "41500.00",
    "lastPrice": "42200.00",
    "volume": "25000.00",
    "quoteVolume": "1037500000.00",
    "highPrice": "43000.00",
    "lowPrice": "40500.00",
    "count": 980000,
}

MOCK_ORDER_BOOK = {
    "lastUpdateId": 99999,
    "bids": [["42190.00", "1.5"], ["42180.00", "2.0"]],
    "asks": [["42200.00", "1.2"], ["42210.00", "0.8"]],
}


# ------------------------------------------------------------------
# Schema tests
# ------------------------------------------------------------------

class TestKlineRecord:
    def test_parses_raw_row(self):
        row = _make_kline_row()
        rec = KlineRecord(
            symbol="BTCUSDT",
            open_time=datetime.utcfromtimestamp(row[0] / 1000),
            close_time=datetime.utcfromtimestamp(row[6] / 1000),
            open=row[1], high=row[2], low=row[3], close=row[4],
            volume=row[5], quote_volume=row[7], num_trades=int(row[8]),
            taker_buy_base_volume=row[9], taker_buy_quote_volume=row[10],
        )
        assert rec.close == 42200.0
        assert rec.num_trades == 1500

    def test_float_coercion(self):
        row = _make_kline_row()
        rec = KlineRecord(
            symbol="BTCUSDT",
            open_time=datetime.utcfromtimestamp(row[0] / 1000),
            close_time=datetime.utcfromtimestamp(row[6] / 1000),
            open=row[1], high=row[2], low=row[3], close=row[4],
            volume=row[5], quote_volume=row[7], num_trades=int(row[8]),
            taker_buy_base_volume=row[9], taker_buy_quote_volume=row[10],
        )
        assert isinstance(rec.close, float)


class TestOrderBookSnapshot:
    def _make_snapshot(self) -> OrderBookSnapshot:
        bids = MOCK_ORDER_BOOK["bids"]
        asks = MOCK_ORDER_BOOK["asks"]
        from src.ingestion.schemas import OrderBookLevel
        return OrderBookSnapshot(
            symbol="BTCUSDT",
            last_update_id=MOCK_ORDER_BOOK["lastUpdateId"],
            bids=[OrderBookLevel(price=float(p), quantity=float(q)) for p, q in bids],
            asks=[OrderBookLevel(price=float(p), quantity=float(q)) for p, q in asks],
        )

    def test_spread(self):
        snap = self._make_snapshot()
        assert snap.spread == pytest.approx(10.0)

    def test_spread_bps(self):
        snap = self._make_snapshot()
        # spread=10, mid~42195 → ~2.37 bps
        assert snap.spread_bps == pytest.approx(2.37, abs=0.1)

    def test_depth_imbalance_in_range(self):
        snap = self._make_snapshot()
        assert -1.0 <= snap.depth_imbalance <= 1.0


class TestTicker24hr:
    def test_parses_mock(self):
        ticker = Ticker24hr(**MOCK_TICKER)
        assert ticker.last_price == 42200.0
        assert ticker.price_change_pct == pytest.approx(2.93)


# ------------------------------------------------------------------
# Collector tests (mocked network)
# ------------------------------------------------------------------

class TestDataCollector:
    @pytest.mark.asyncio
    async def test_collect_symbol_success(self):
        klines_raw = [_make_kline_row(1_700_000_000_000 + i * 3_600_000) for i in range(5)]

        with (
            patch("src.ingestion.collector.BinanceClient") as MockClient,
        ):
            client = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=None)
            client.get_klines = AsyncMock(return_value=klines_raw)
            client.get_ticker_24hr = AsyncMock(return_value=MOCK_TICKER)
            client.get_order_book = AsyncMock(return_value=MOCK_ORDER_BOOK)
            client.get_funding_rate = AsyncMock(return_value=[])

            async with DataCollector(client=client) as collector:
                snapshot = await collector.collect_symbol("BTCUSDT")

        assert snapshot.symbol == "BTCUSDT"
        assert len(snapshot.klines) == 5
        assert snapshot.ticker.last_price == 42200.0

    @pytest.mark.asyncio
    async def test_funding_rate_failure_is_graceful(self):
        klines_raw = [_make_kline_row()]
        client = AsyncMock()
        client.get_klines = AsyncMock(return_value=klines_raw)
        client.get_ticker_24hr = AsyncMock(return_value=MOCK_TICKER)
        client.get_order_book = AsyncMock(return_value=MOCK_ORDER_BOOK)
        client.get_funding_rate = AsyncMock(side_effect=Exception("futures unavailable"))

        async with DataCollector(client=client) as collector:
            snapshot = await collector.collect_symbol("BTCUSDT")

        assert snapshot.funding_rates == []
