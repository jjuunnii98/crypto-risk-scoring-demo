"""
Integration tests for the FastAPI application.

Uses httpx.AsyncClient with FastAPI's ASGI transport — no live server needed.
The model artifact (data/raw/risk_model.pkl) must exist; run notebook 03 first.

Live Binance calls are patched so tests are fast and deterministic.
"""

import math
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.main import app
from src.api.schemas import ModelInfoResponse
from src.models.scorer import RiskResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_risk_result(symbol: str, risk_score: float = 42.0) -> RiskResult:
    return RiskResult(
        symbol=symbol,
        risk_score=risk_score,
        last_price=50_000.0,
        spread_bps=0.5,
        depth_imbalance=0.1,
        realized_vol_24h=0.03,
        funding_rate_z=0.5,
        scored_at=datetime.now(timezone.utc),
    )


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_returns_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


# ---------------------------------------------------------------------------
# /model/info
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_model_info_fitted(client):
    mock_model = MagicMock()
    mock_model._fitted = True
    mock_model.concordance_index = 0.85
    mock_model.penalizer = 0.1

    with patch("src.api.main._scorer") as mock_scorer:
        mock_scorer._model = mock_model

        resp = await client.get("/model/info")

    assert resp.status_code == 200
    info = resp.json()
    assert info["fitted"] is True
    assert info["concordance_index"] == pytest.approx(0.85)
    assert info["event_threshold_pct"] == pytest.approx(3.0)
    assert info["duration_window_hours"] == 24
    assert len(info["features"]) == 16


# ---------------------------------------------------------------------------
# POST /score — patched scorer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_score_returns_results(client):
    mock_results = [
        _make_risk_result("BTCUSDT", 30.0),
        _make_risk_result("ETHUSDT", 70.0),
    ]
    with patch("src.api.main._scorer") as mock_scorer:
        mock_scorer._model._fitted = True
        mock_scorer.score_symbols = AsyncMock(return_value=mock_results)

        resp = await client.post("/score", json={"symbols": ["BTCUSDT", "ETHUSDT"]})

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    symbols = [r["symbol"] for r in body["results"]]
    assert "BTCUSDT" in symbols
    assert "ETHUSDT" in symbols


@pytest.mark.asyncio
async def test_post_score_risk_score_range(client):
    mock_results = [_make_risk_result("BTCUSDT", 55.5)]
    with patch("src.api.main._scorer") as mock_scorer:
        mock_scorer._model._fitted = True
        mock_scorer.score_symbols = AsyncMock(return_value=mock_results)

        resp = await client.post("/score", json={"symbols": ["BTCUSDT"]})

    result = resp.json()["results"][0]
    assert 0 <= result["risk_score"] <= 100


@pytest.mark.asyncio
async def test_post_score_unfitted_model_returns_400(client):
    with patch("src.api.main._scorer") as mock_scorer:
        mock_scorer._model._fitted = False

        resp = await client.post("/score", json={"symbols": ["BTCUSDT"]})

    assert resp.status_code == 400
    assert "not yet trained" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_post_score_empty_result(client):
    with patch("src.api.main._scorer") as mock_scorer:
        mock_scorer._model._fitted = True
        mock_scorer.score_symbols = AsyncMock(return_value=[])

        resp = await client.post("/score", json={"symbols": ["XYZUSDT"]})

    assert resp.status_code == 200
    assert resp.json()["count"] == 0


# ---------------------------------------------------------------------------
# GET /score/{symbol}
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_score_symbol_returns_single(client):
    mock_results = [_make_risk_result("BTCUSDT", 42.0)]
    with patch("src.api.main._scorer") as mock_scorer:
        mock_scorer._model._fitted = True
        mock_scorer.score_symbols = AsyncMock(return_value=mock_results)

        resp = await client.get("/score/btcusdt")

    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "BTCUSDT"
    assert body["risk_score"] == pytest.approx(42.0)


@pytest.mark.asyncio
async def test_get_score_symbol_uppercases_input(client):
    mock_results = [_make_risk_result("ETHUSDT", 60.0)]
    with patch("src.api.main._scorer") as mock_scorer:
        mock_scorer._model._fitted = True
        mock_scorer.score_symbols = AsyncMock(return_value=mock_results)

        resp = await client.get("/score/ethusdt")

    assert resp.status_code == 200
    assert resp.json()["symbol"] == "ETHUSDT"


@pytest.mark.asyncio
async def test_get_score_symbol_not_found_returns_404(client):
    with patch("src.api.main._scorer") as mock_scorer:
        mock_scorer._model._fitted = True
        mock_scorer.score_symbols = AsyncMock(return_value=[])

        resp = await client.get("/score/FAKEUSDT")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 503 when scorer not initialized
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_score_503_when_scorer_none(client):
    with patch("src.api.main._scorer", None):
        resp = await client.post("/score", json={"symbols": ["BTCUSDT"]})
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_model_info_503_when_scorer_none(client):
    with patch("src.api.main._scorer", None):
        resp = await client.get("/model/info")
    assert resp.status_code == 503
