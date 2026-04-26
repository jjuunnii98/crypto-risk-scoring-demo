# Crypto Risk Scoring Demo

![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-40%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

End-to-end pipeline that scores cryptocurrency market risk in real time. Pulls live data from the Binance public API, computes technical and volatility features, and returns a risk score (0–100) using a **Cox Proportional Hazards survival model** (C-index 0.862). Served via a FastAPI service — no API key required.

---

## Architecture

```
Binance Public API  (no key required)
        │
        ▼
┌───────────────────────┐
│    Data Ingestion     │  src/ingestion/
│  BinanceClient        │  async httpx + tenacity retry
│  DataCollector        │  asyncio.gather — parallel multi-symbol
│  Pydantic Schemas     │  OHLCV · ticker · order book · funding rate
└──────────┬────────────┘
           │  SymbolSnapshot
           ▼
┌───────────────────────┐
│  Feature Engineering  │  src/features/
│  technical.py         │  RSI · BB · MACD · ATR · VWAP · buy/sell ratio
│  volatility.py        │  Realized vol · Garman-Klass · drawdown · funding z
│  pipeline.py          │  16-feature design matrix
└──────────┬────────────┘
           │  Feature DataFrame
           ▼
┌───────────────────────┐
│     Risk Model        │  src/models/
│  CryptoSurvivalModel  │  Cox PH via lifelines — C-index 0.862
│  RiskScorer           │  collect → featurize → score (async)
└──────────┬────────────┘
           │  RiskResult  score ∈ [0, 100]
           ▼
┌───────────────────────┐
│       FastAPI         │  src/api/
│  POST /score          │  batch scoring
│  GET  /score/{symbol} │  single symbol
│  GET  /model/info     │  C-index · features · event definition
│  GET  /health         │  liveness probe
└───────────────────────┘
```

---

## Survival Analysis Model

The risk model frames drawdown prediction as a **survival problem**.

| | |
|---|---|
| **Event** | Cumulative log return from current price first crosses **−3%** within the next 24 hours |
| **Duration** | Hours until first crossing (right-censored at 24h if no event) |
| **Event rate** | ~7.3% on 21 days of hourly data |
| **Model** | Cox Proportional Hazards (`lifelines.CoxPHFitter`, penalizer=0.1) |
| **C-index** | **0.862** |
| **Output** | Partial hazard normalized to [0, 100] across the live sample |

Using time-to-event (1–24h) rather than a binary label preserves the ordering information that Cox PH needs to rank concordant pairs, and naturally handles the asymmetry between "event in 1h" vs "event in 23h."

---

## Features (16 total)

**Technical indicators**

| Feature | Description |
|---|---|
| `log_return` | 1-bar log return |
| `rsi_14` | Relative Strength Index (14-period) |
| `bb_width_20` | Bollinger Band width (20-period) |
| `bb_pct_20` | Price position within Bollinger Bands |
| `macd_hist` | MACD histogram (12/26/9) |
| `atr_pct_14` | ATR as % of price (14-period) |
| `volume_ratio` | Volume vs 20-bar rolling mean |
| `vwap_divergence` | Price deviation from VWAP (20-period) |
| `buy_sell_ratio` | Taker buy / taker sell volume |

**Volatility & tail risk**

| Feature | Description |
|---|---|
| `realized_vol_24h` | Annualized realized volatility (24h) |
| `realized_vol_168h` | Annualized realized volatility (168h / 1 week) |
| `gk_vol_24h` | Garman-Klass high-low volatility estimator (24h) |
| `downside_dev_24h` | Downside deviation — Sortino numerator (24h) |
| `max_drawdown_24h` | Rolling maximum drawdown (24h) |
| `max_drawdown_168h` | Rolling maximum drawdown (168h) |
| `funding_rate_z` | Funding rate z-score (30-period rolling) |

---

## API

Start the server (trained model artifact included):

```bash
uvicorn src.api.main:app --reload
```

**`POST /score`** — batch symbols

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]}'
```

```json
{
  "results": [
    {
      "symbol": "BTCUSDT",
      "risk_score": 12.4,
      "last_price": 78023.15,
      "spread_bps": 0.001,
      "depth_imbalance": 0.4434,
      "realized_vol_24h": 0.0345,
      "funding_rate_z": 1.206,
      "scored_at": "2026-04-26T07:23:28Z"
    }
  ],
  "count": 3
}
```

**`GET /score/{symbol}`** — single symbol

```bash
curl http://localhost:8000/score/BTCUSDT
```

**`GET /model/info`** — model metadata

```bash
curl http://localhost:8000/model/info
```

```json
{
  "fitted": true,
  "concordance_index": 0.8620,
  "penalizer": 0.1,
  "event_threshold_pct": 3.0,
  "duration_window_hours": 24,
  "features": ["log_return", "rsi_14", "bb_width_20", "..."]
}
```

---

## Quickstart

```bash
# 1. Create virtual environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) Retrain the model — run notebooks in order
#    notebooks/01_data_exploration.ipynb
#    notebooks/02_feature_engineering.ipynb
#    notebooks/03_model_training.ipynb   ← saves data/raw/risk_model.pkl

# 3. Start the API (pre-trained artifact already included)
uvicorn src.api.main:app --reload

# 4. Run tests
pytest tests/ -v   # 40 tests
```

---

## Project Structure

```
crypto-risk-scoring-demo/
├── config/
│   └── settings.py               # pydantic-settings: symbols, intervals, API URLs
├── src/
│   ├── ingestion/
│   │   ├── binance_client.py     # async httpx client with tenacity retry
│   │   ├── collector.py          # parallel multi-symbol collector
│   │   └── schemas.py            # Pydantic v2 data models
│   ├── features/
│   │   ├── technical.py          # RSI, BB, MACD, ATR, VWAP, volume features
│   │   ├── volatility.py         # realized vol, GK, drawdown, funding z-score
│   │   └── pipeline.py           # 16-feature design matrix assembly
│   ├── models/
│   │   ├── survival.py           # Cox PH model + event labeling
│   │   └── scorer.py             # end-to-end async inference pipeline
│   └── api/
│       ├── main.py               # FastAPI app with lifespan model loading
│       └── schemas.py            # request/response schemas
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── tests/
│   ├── test_ingestion.py         # schema validation + mocked async collector
│   ├── test_features.py          # feature unit tests (21 cases)
│   └── test_api.py               # FastAPI integration tests (11 cases)
├── data/raw/
│   └── risk_model.pkl            # pre-trained Cox PH artifact
└── requirements.txt
```

---

## Data Sources

All Binance public endpoints — no API key required.

| Endpoint | Data |
|---|---|
| `GET /api/v3/klines` | OHLCV candlesticks (hourly, 21-day window) |
| `GET /api/v3/ticker/24hr` | 24h rolling price/volume stats |
| `GET /api/v3/depth` | Order book snapshot (bid-ask spread, depth imbalance) |
| `GET /fapi/v1/fundingRate` | Futures funding rate history |

---

## Tech Stack

`Python 3.11` · `FastAPI` · `httpx` · `pydantic-v2` · `lifelines` · `pandas` · `numpy` · `tenacity` · `structlog` · `pytest-asyncio`

---

## Related Work

- [survival-analysis-finance](https://github.com/jjuunnii98/survival-analysis-finance) — Cox PH and AFT models applied to cryptocurrency volatility and default prediction
