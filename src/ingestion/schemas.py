"""
Data schemas for validated ingestion output.

Pydantic models act as contracts between the ingestion layer
and the feature engineering pipeline.
"""

from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class KlineRecord(BaseModel):
    """Single OHLCV candlestick bar."""

    symbol: str
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    num_trades: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float

    @field_validator("open", "high", "low", "close", "volume", mode="before")
    @classmethod
    def parse_float(cls, v):
        return float(v)


class Ticker24hr(BaseModel):
    """24-hour rolling ticker statistics."""

    symbol: str
    price_change: float = Field(alias="priceChange")
    price_change_pct: float = Field(alias="priceChangePercent")
    weighted_avg_price: float = Field(alias="weightedAvgPrice")
    last_price: float = Field(alias="lastPrice")
    volume: float
    quote_volume: float = Field(alias="quoteVolume")
    high: float = Field(alias="highPrice")
    low: float = Field(alias="lowPrice")
    count: int = Field(alias="count")

    model_config = {"populate_by_name": True}

    @field_validator(
        "price_change",
        "price_change_pct",
        "weighted_avg_price",
        "last_price",
        "volume",
        "quote_volume",
        "high",
        "low",
        mode="before",
    )
    @classmethod
    def parse_float(cls, v):
        return float(v)


class OrderBookLevel(BaseModel):
    price: float
    quantity: float


class OrderBookSnapshot(BaseModel):
    """Bid/ask depth snapshot."""

    symbol: str
    last_update_id: int
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        return (self.spread / self.mid_price) * 10_000 if self.mid_price else 0.0

    @property
    def bid_depth(self) -> float:
        return sum(level.quantity for level in self.bids)

    @property
    def ask_depth(self) -> float:
        return sum(level.quantity for level in self.asks)

    @property
    def depth_imbalance(self) -> float:
        """(bid_depth - ask_depth) / (bid_depth + ask_depth), range [-1, 1]."""
        total = self.bid_depth + self.ask_depth
        return (self.bid_depth - self.ask_depth) / total if total else 0.0


class FundingRateRecord(BaseModel):
    """Single historical funding rate entry."""

    symbol: str
    funding_rate: float = Field(alias="fundingRate")
    funding_time: datetime = Field(alias="fundingTime")

    model_config = {"populate_by_name": True}

    @field_validator("funding_rate", mode="before")
    @classmethod
    def parse_float(cls, v):
        return float(v)

    @field_validator("funding_time", mode="before")
    @classmethod
    def parse_ms(cls, v):
        return datetime.utcfromtimestamp(int(v) / 1000)


class SymbolSnapshot(BaseModel):
    """All collected data for one symbol at a point in time."""

    symbol: str
    collected_at: datetime
    klines: list[KlineRecord]
    ticker: Ticker24hr
    order_book: OrderBookSnapshot
    funding_rates: list[FundingRateRecord]
