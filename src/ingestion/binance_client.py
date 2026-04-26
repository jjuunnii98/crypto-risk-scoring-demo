"""
Low-level Binance REST API client.

Wraps httpx with retry logic and rate-limit awareness.
All endpoints used are public and require no API key.
"""

import time
import structlog
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import settings

logger = structlog.get_logger(__name__)

_RETRYABLE = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


class BinanceClient:
    """Thin async HTTP client for Binance public REST endpoints."""

    def __init__(self) -> None:
        self._spot = httpx.AsyncClient(
            base_url=settings.binance_base_url,
            timeout=settings.request_timeout,
            headers={"Content-Type": "application/json"},
        )
        self._futures = httpx.AsyncClient(
            base_url=settings.binance_futures_url,
            timeout=settings.request_timeout,
            headers={"Content-Type": "application/json"},
        )

    async def __aenter__(self) -> "BinanceClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def close(self) -> None:
        await self._spot.aclose()
        await self._futures.aclose()

    # ------------------------------------------------------------------
    # Spot endpoints
    # ------------------------------------------------------------------

    async def get_klines(
        self,
        symbol: str,
        interval: str = settings.kline_interval,
        limit: int = settings.kline_lookback,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[list]:
        """
        GET /api/v3/klines

        Returns raw list of kline arrays:
        [open_time, open, high, low, close, volume, close_time,
         quote_volume, num_trades, taker_buy_base, taker_buy_quote, ignore]
        """
        params: dict = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        return await self._get_spot("/api/v3/klines", params)

    async def get_ticker_24hr(self, symbol: str) -> dict:
        """GET /api/v3/ticker/24hr — 24-hour rolling window statistics."""
        return await self._get_spot("/api/v3/ticker/24hr", {"symbol": symbol})

    async def get_ticker_24hr_all(self) -> list[dict]:
        """GET /api/v3/ticker/24hr — stats for all symbols (no symbol param)."""
        return await self._get_spot("/api/v3/ticker/24hr", {})

    async def get_order_book(
        self,
        symbol: str,
        limit: int = settings.order_book_limit,
    ) -> dict:
        """
        GET /api/v3/depth

        Returns {"lastUpdateId": int, "bids": [[price, qty], ...], "asks": [...]}
        """
        return await self._get_spot("/api/v3/depth", {"symbol": symbol, "limit": limit})

    async def get_recent_trades(self, symbol: str, limit: int = 500) -> list[dict]:
        """GET /api/v3/trades — most recent trades (max 1000)."""
        return await self._get_spot(
            "/api/v3/trades", {"symbol": symbol, "limit": limit}
        )

    async def get_exchange_info(self) -> dict:
        """GET /api/v3/exchangeInfo — available symbols and trading rules."""
        return await self._get_spot("/api/v3/exchangeInfo", {})

    async def get_server_time(self) -> int:
        """GET /api/v3/time — server timestamp in milliseconds."""
        data = await self._get_spot("/api/v3/time", {})
        return data["serverTime"]

    # ------------------------------------------------------------------
    # Futures endpoints
    # ------------------------------------------------------------------

    async def get_funding_rate(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[dict]:
        """
        GET /fapi/v1/fundingRate

        Returns historical funding rates — a key signal for positioning risk.
        """
        return await self._get_futures(
            "/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit}
        )

    async def get_open_interest(self, symbol: str) -> dict:
        """GET /fapi/v1/openInterest — current open interest for a futures symbol."""
        return await self._get_futures("/fapi/v1/openInterest", {"symbol": symbol})

    async def get_long_short_ratio(
        self,
        symbol: str,
        period: str = "1h",
        limit: int = 30,
    ) -> list[dict]:
        """
        GET /futures/data/globalLongShortAccountRatio

        Top-trader long/short ratio — sentiment proxy for risk model features.
        """
        return await self._get_futures(
            "/futures/data/globalLongShortAccountRatio",
            {"symbol": symbol, "period": period, "limit": limit},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(
            min=settings.retry_wait_min, max=settings.retry_wait_max
        ),
        reraise=True,
    )
    async def _get_spot(self, path: str, params: dict):
        t0 = time.perf_counter()
        response = await self._spot.get(path, params=params)
        self._raise_for_status(response, path)
        elapsed = time.perf_counter() - t0
        logger.debug("binance_spot_request", path=path, params=params, elapsed=elapsed)
        return response.json()

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(settings.max_retries),
        wait=wait_exponential(
            min=settings.retry_wait_min, max=settings.retry_wait_max
        ),
        reraise=True,
    )
    async def _get_futures(self, path: str, params: dict):
        t0 = time.perf_counter()
        response = await self._futures.get(path, params=params)
        self._raise_for_status(response, path)
        elapsed = time.perf_counter() - t0
        logger.debug(
            "binance_futures_request", path=path, params=params, elapsed=elapsed
        )
        return response.json()

    @staticmethod
    def _raise_for_status(response: httpx.Response, path: str) -> None:
        if response.status_code == 429:
            raise httpx.HTTPStatusError(
                f"Rate limited on {path}",
                request=response.request,
                response=response,
            )
        if response.status_code == 418:
            raise httpx.HTTPStatusError(
                f"IP banned by Binance on {path}",
                request=response.request,
                response=response,
            )
        response.raise_for_status()
