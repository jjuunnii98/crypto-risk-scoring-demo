# Crypto Risk Scoring Demo

End-to-end crypto market risk scoring system: data ingestion from Binance public API, feature engineering (technical indicators + volatility), survival analysis–based risk scoring, and FastAPI deployment.

---

## System Architecture

```
Binance Public API
      │
      ▼
┌─────────────────────┐
│   Data Ingestion    │  src/ingestion/
│  - BinanceClient    │  Async HTTP, retry logic, rate-limit handling
│  - DataCollector    │  Parallel multi-symbol collection
│  - Schemas          │  Pydantic validation (OHLCV, ticker, order book, funding)
└────────┬────────────┘
         │ SymbolSnapshot
         ▼
┌─────────────────────┐
│ Feature Engineering │  src/features/
│  - Technical        │  RSI, Bollinger Bands, MACD, ATR, VWAP, buy/sell pressure
│  - Volatility       │  Realized vol, Garman-Klass, downside deviation, max drawdown
│  - Pipeline         │  Assembles design matrix per symbol
└────────┬────────────┘
         │ Feature DataFrame
         ▼
┌─────────────────────┐
│   Risk Model        │  src/models/
│  - CoxPH Survival   │  Event = cumulative log return < -3% within 24h
│  - RiskScorer       │  Live inference: collect → featurize → score
└────────┬────────────┘
         │ RiskResult (0–100)
         ▼
┌─────────────────────┐
│     FastAPI         │  src/api/
│  POST /score        │  Batch scoring endpoint
│  GET  /score/{sym}  │  Single symbol
│  GET  /health       │  Health check
│  GET  /model/info   │  Model metadata (C-index, features, event definition)
└─────────────────────┘
```

## Data Sources (Binance Public API — no API key required)

| Endpoint | Data | Usage |
|---|---|---|
| `GET /api/v3/klines` | OHLCV candlesticks | Primary time-series features |
| `GET /api/v3/ticker/24hr` | 24h rolling stats | Price change, volume summary |
| `GET /api/v3/depth` | Order book | Spread, depth imbalance |
| `GET /fapi/v1/fundingRate` | Futures funding rate | Positioning/sentiment signal |

## Features

**Technical indicators**
- Log returns, RSI(14), Bollinger Bands(20), MACD(12/26/9)
- ATR(14), VWAP(20), buy/sell volume ratio

**Volatility & tail risk**
- Realized volatility: 24h, 72h, 168h windows (annualized)
- Garman-Klass high/low estimator
- Downside deviation (Sortino numerator)
- Rolling max drawdown: 24h, 168h

**Market microstructure**
- Bid-ask spread (bps)
- Order book depth imbalance
- Funding rate z-score (30-period rolling)

## Project Structure

```
crypto-risk-scoring-demo/
├── config/
│   └── settings.py          # Symbols, intervals, API config (pydantic-settings)
├── src/
│   ├── ingestion/
│   │   ├── binance_client.py    # Async HTTP client with retry/rate-limit handling
│   │   ├── collector.py         # Parallel multi-symbol collector
│   │   └── schemas.py           # Pydantic data models
│   ├── features/
│   │   ├── technical.py         # Technical indicator computations
│   │   ├── volatility.py        # Volatility and tail-risk features
│   │   └── pipeline.py          # Feature assembly pipeline
│   ├── models/
│   │   ├── survival.py          # Cox PH survival model (lifelines)
│   │   └── scorer.py            # End-to-end inference pipeline
│   └── api/
│       ├── main.py              # FastAPI app
│       └── schemas.py           # Request/response Pydantic schemas
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── tests/
│   ├── test_ingestion.py        # Pydantic schema + async collector tests
│   ├── test_features.py         # Feature engineering unit tests (21 cases)
│   └── test_api.py              # FastAPI integration tests (11 cases)
├── data/raw/                    # gitignored — Parquet outputs
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
```

**1. Explore and train — run the notebooks in order**

```
notebooks/01_data_exploration.ipynb   # Live data pull + EDA
notebooks/02_feature_engineering.ipynb # Feature pipeline + labeled dataset
notebooks/03_model_training.ipynb      # Cox PH fit → saves data/raw/risk_model.pkl
```

**2. Start the API (requires the fitted model from step 1)**

```bash
uvicorn src.api.main:app --reload
```

```bash
# Batch score
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]}'

# Single symbol
curl http://localhost:8000/score/BTCUSDT

# Model metadata (C-index, feature list, event definition)
curl http://localhost:8000/model/info
```

**3. Run tests**

```bash
pytest tests/ -v   # 40 tests
```

## Tech Stack

`Python 3.11` · `httpx` · `pydantic` · `pandas` · `numpy` · `lifelines` · `FastAPI` · `uvicorn` · `tenacity` · `structlog`

## Related Work

- [survival-analysis-finance](https://github.com/jjuunnii98/survival-analysis-finance) — Cox PH and AFT models for cryptocurrency volatility and default prediction
