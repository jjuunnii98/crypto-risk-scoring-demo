from datetime import datetime
from pydantic import BaseModel, Field


class RiskScoreResponse(BaseModel):
    symbol: str
    risk_score: float = Field(ge=0, le=100, description="Risk score 0 (low) – 100 (high)")
    last_price: float
    spread_bps: float
    depth_imbalance: float
    realized_vol_24h: float | None
    funding_rate_z: float | None
    scored_at: datetime


class BatchScoreRequest(BaseModel):
    symbols: list[str] = Field(default=["BTCUSDT", "ETHUSDT"])


class BatchScoreResponse(BaseModel):
    results: list[RiskScoreResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"


class ModelInfoResponse(BaseModel):
    fitted: bool
    concordance_index: float | None
    penalizer: float
    event_definition: str
    event_threshold_pct: float
    duration_window_hours: int
    features: list[str]
