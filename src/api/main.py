"""
FastAPI application entry point.

Run with:
    uvicorn src.api.main:app --reload

Model artifact is loaded from data/raw/risk_model.pkl if available.
Generate it by running notebooks/03_model_training.ipynb first.
"""

import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    ModelInfoResponse,
    RiskScoreResponse,
)
from src.features.pipeline import FEATURE_COLS
from src.models.scorer import RiskScorer
from src.models.survival import CryptoSurvivalModel, DURATION_WINDOW, EVENT_THRESHOLD

logger = structlog.get_logger(__name__)

# Absolute path so the artifact is found regardless of working directory
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "risk_model.pkl"
_scorer: RiskScorer | None = None


def _load_model() -> CryptoSurvivalModel:
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info("model_loaded", path=str(MODEL_PATH), fitted=model._fitted)
        return model
    logger.warning("model_artifact_not_found", path=str(MODEL_PATH))
    return CryptoSurvivalModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scorer
    model = _load_model()
    _scorer = RiskScorer(model)
    logger.info("app_startup", model_fitted=model._fitted)
    yield
    logger.info("app_shutdown")


app = FastAPI(
    title="Crypto Risk Scoring API",
    description=(
        "End-to-end crypto market risk scoring system. "
        "Fetches live Binance data, computes technical and volatility features, "
        "and returns a survival-model-based risk score (0–100)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/score", response_model=BatchScoreResponse, tags=["scoring"])
async def score(request: BatchScoreRequest) -> BatchScoreResponse:
    if not _scorer:
        raise HTTPException(status_code=503, detail="Scorer not initialized.")

    if not _scorer._model._fitted:
        raise HTTPException(
            status_code=400,
            detail=(
                "Model is not yet trained. "
                "Run the training notebook (notebooks/03_model_training.ipynb) first."
            ),
        )

    results = await _scorer.score_symbols(request.symbols)
    return BatchScoreResponse(
        results=[RiskScoreResponse(**vars(r)) for r in results],
        count=len(results),
    )


@app.get("/score/{symbol}", response_model=RiskScoreResponse, tags=["scoring"])
async def score_single(symbol: str) -> RiskScoreResponse:
    resp = await score(BatchScoreRequest(symbols=[symbol.upper()]))
    if not resp.results:
        raise HTTPException(status_code=404, detail=f"No data returned for {symbol}.")
    return resp.results[0]


@app.get("/model/info", response_model=ModelInfoResponse, tags=["ops"])
async def model_info() -> ModelInfoResponse:
    if not _scorer:
        raise HTTPException(status_code=503, detail="Scorer not initialized.")
    model = _scorer._model
    return ModelInfoResponse(
        fitted=model._fitted,
        concordance_index=model.concordance_index if model._fitted else None,
        penalizer=model.penalizer,
        event_definition="Cumulative log return from current price first crosses EVENT_THRESHOLD within DURATION_WINDOW hours.",
        event_threshold_pct=abs(EVENT_THRESHOLD) * 100,
        duration_window_hours=DURATION_WINDOW,
        features=FEATURE_COLS,
    )
