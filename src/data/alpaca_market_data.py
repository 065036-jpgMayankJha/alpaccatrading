"""Batched, cached access to Alpaca's historical stock-bars data endpoint."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from math import isfinite
from typing import Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

BAR_COLUMNS = ("open", "high", "low", "close", "volume")
DATA_ENDPOINT = "https://data.alpaca.markets/v2/stocks/bars"
SUPPORTED_FEEDS = {"iex", "sip", "delayed_sip", "overnight", "boats", "otc"}
_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")
_TIMEFRAME_PATTERN = re.compile(r"^[1-9][0-9]*(Min|Hour|Day|Week|Month)$")


class MarketDataError(RuntimeError):
    """An Alpaca data API or response-validation failure."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class MarketDataBatch:
    """Validated OHLCV bars and per-symbol coverage diagnostics.

    Requested symbols without returned bars are present in ``bars`` as empty
    DataFrames and listed in ``missing_symbols``.
    """

    bars: Mapping[str, pd.DataFrame]
    requested_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    quality: Mapping[str, Mapping[str, object]]
    from_cache: bool = False


class AlpacaMarketDataClient:
    """Fetch multiple symbols together and cache the resulting batch in memory.

    Credentials are loaded lazily from ``APCA_API_KEY_ID`` /
    ``APCA_API_SECRET_KEY`` (or ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY``).
    The configurable feed comes from ``ALPACA_DATA_FEED``. Requests are always
    sent to the market-data host; this client contains no trading operations.

    ``transport`` and ``clock`` are injectable for deterministic offline tests.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        secret_key: str | None = None,
        feed: str | None = None,
        cache_ttl_seconds: float = 60.0,
        timeout_seconds: float = 15.0,
        max_retries: int = 2,
        transport: Callable[[Request, float], object] | None = None,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if cache_ttl_seconds < 0 or timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("Cache TTL must be non-negative; timeout positive; retries non-negative.")
        self._api_key = api_key
        self._secret_key = secret_key
        self.feed = (feed or os.getenv("ALPACA_DATA_FEED", "iex")).strip().lower()
        if self.feed not in SUPPORTED_FEEDS:
            raise ValueError(f"Unsupported ALPACA_DATA_FEED {self.feed!r}.")
        self.cache_ttl_seconds = cache_ttl_seconds
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._transport = transport or self._default_transport
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleeper = sleeper
        self._cache: dict[tuple[object, ...], tuple[float, MarketDataBatch]] = {}
        self._cache_lock = threading.RLock()

    @staticmethod
    def _default_transport(request: Request, timeout: float) -> object:
        return urlopen(request, timeout=timeout)

    def fetch_bars(
        self,
        symbols: list[str] | tuple[str, ...],
        timeframe: str,
        lookback: str | int | float | timedelta | pd.Timedelta,
    ) -> MarketDataBatch:
        """Fetch bars for all requested tickers through one paginated endpoint.

        ``lookback`` is a positive duration (such as ``"90 days"``) or a
        number of days. The returned frames have a sorted, UTC DatetimeIndex
        named ``timestamp`` and exactly the lowercase OHLCV columns.
        """
        normalized_symbols = self._validate_symbols(symbols)
        normalized_timeframe = self._validate_timeframe(timeframe)
        lookback_delta = self._parse_lookback(lookback)
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("The client clock must return a timezone-aware datetime.")
        end = now.astimezone(timezone.utc)
        start = end - lookback_delta
        cache_key = (
            normalized_symbols,
            normalized_timeframe,
            lookback_delta.total_seconds(),
            self.feed,
        )
        now_monotonic = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached and now_monotonic - cached[0] < self.cache_ttl_seconds:
                return self._copy_batch(cached[1], from_cache=True)

        api_key, secret_key = self._credentials()
        all_bars: dict[str, list[dict[str, object]]] = {
            symbol: [] for symbol in normalized_symbols
        }
        page_token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            params: dict[str, str | int] = {
                "symbols": ",".join(normalized_symbols),
                "timeframe": normalized_timeframe,
                "start": self._format_timestamp(start),
                "end": self._format_timestamp(end),
                "limit": 10000,
                "sort": "asc",
                "feed": self.feed,
            }
            if page_token is not None:
                params["page_token"] = page_token
            payload = self._request_page(params, api_key, secret_key)
            page_bars = payload.get("bars")
            if not isinstance(page_bars, dict):
                raise MarketDataError("Alpaca response must contain a symbol-to-bars 'bars' object.")
            for symbol, rows in page_bars.items():
                if symbol not in all_bars:
                    raise MarketDataError(f"Alpaca returned unrequested symbol {symbol!r}.")
                if not isinstance(rows, list):
                    raise MarketDataError(f"Alpaca bars for {symbol} must be a list.")
                all_bars[symbol].extend(rows)
            next_token = payload.get("next_page_token")
            if not next_token:
                break
            if not isinstance(next_token, str) or next_token in seen_tokens:
                raise MarketDataError("Alpaca returned an invalid or repeated pagination token.")
            seen_tokens.add(next_token)
            page_token = next_token

        batch = self._build_batch(normalized_symbols, all_bars)
        with self._cache_lock:
            self._cache[cache_key] = (time.monotonic(), batch)
        return self._copy_batch(batch, from_cache=False)

    def clear_cache(self) -> None:
        """Remove all cached batches."""
        with self._cache_lock:
            self._cache.clear()

    @staticmethod
    def _validate_symbols(symbols: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        if not isinstance(symbols, (list, tuple)) or not symbols:
            raise ValueError("symbols must be a non-empty list of ticker strings.")
        if len(symbols) > 100:
            raise ValueError("At most 100 symbols may be requested in one batch.")
        normalized: list[str] = []
        for symbol in symbols:
            if not isinstance(symbol, str):
                raise ValueError("Every symbol must be a string.")
            ticker = symbol.strip().upper()
            if not _SYMBOL_PATTERN.fullmatch(ticker):
                raise ValueError(f"Invalid ticker symbol {symbol!r}.")
            if ticker in normalized:
                raise ValueError(f"Duplicate ticker symbol {ticker!r}.")
            normalized.append(ticker)
        return tuple(normalized)

    @staticmethod
    def _validate_timeframe(timeframe: str) -> str:
        if not isinstance(timeframe, str) or not _TIMEFRAME_PATTERN.fullmatch(timeframe):
            raise ValueError("timeframe must use Alpaca format such as '1Day', '1Hour', or '5Min'.")
        return timeframe

    @staticmethod
    def _parse_lookback(
        lookback: str | int | float | timedelta | pd.Timedelta,
    ) -> timedelta:
        try:
            if isinstance(lookback, bool):
                raise ValueError
            if isinstance(lookback, (int, float)):
                duration = pd.Timedelta(days=float(lookback))
            else:
                duration = pd.Timedelta(lookback)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("lookback must be a positive duration or number of days.") from exc
        if pd.isna(duration) or duration <= pd.Timedelta(0):
            raise ValueError("lookback must be a positive duration.")
        return duration.to_pytimedelta()

    def _credentials(self) -> tuple[str, str]:
        api_key = self._api_key or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
        secret_key = (
            self._secret_key
            or os.getenv("APCA_API_SECRET_KEY")
            or os.getenv("ALPACA_SECRET_KEY")
        )
        if not api_key or not secret_key:
            raise MarketDataError(
                "Alpaca market-data credentials are missing; configure APCA_API_KEY_ID "
                "and APCA_API_SECRET_KEY (or ALPACA_API_KEY and ALPACA_SECRET_KEY)."
            )
        return api_key, secret_key

    @staticmethod
    def _format_timestamp(value: datetime) -> str:
        return value.isoformat(timespec="seconds").replace("+00:00", "Z")

    def _request_page(
        self,
        params: Mapping[str, str | int],
        api_key: str,
        secret_key: str,
    ) -> dict[str, object]:
        url = f"{DATA_ENDPOINT}?{urlencode(params)}"
        request = Request(
            url,
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
                "Accept": "application/json",
            },
        )
        for attempt in range(self.max_retries + 1):
            try:
                response = self._transport(request, self.timeout_seconds)
                with response as result:
                    raw = result.read()
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise MarketDataError("Alpaca response must be a JSON object.")
                return payload
            except HTTPError as exc:
                if exc.code == 429 and attempt < self.max_retries:
                    self._sleeper(self._retry_delay(exc.headers, attempt))
                    continue
                detail = self._read_error(exc)
                if exc.code == 429:
                    raise MarketDataError(
                        f"Alpaca rate limit exceeded after {attempt + 1} attempt(s): {detail}",
                        status_code=exc.code,
                    ) from exc
                raise MarketDataError(
                    f"Alpaca market-data request failed with HTTP {exc.code}: {detail}",
                    status_code=exc.code,
                ) from exc
            except URLError as exc:
                raise MarketDataError(f"Alpaca market-data network error: {exc.reason}") from exc
            except (TimeoutError, OSError) as exc:
                raise MarketDataError(f"Alpaca market-data network error: {exc}") from exc
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise MarketDataError("Alpaca returned invalid JSON.") from exc
        raise MarketDataError("Alpaca market-data request exhausted retries.")

    @staticmethod
    def _retry_delay(headers: object, attempt: int) -> float:
        retry_after = headers.get("Retry-After") if headers is not None else None
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
        return min(2.0**attempt, 8.0)

    @staticmethod
    def _read_error(exc: HTTPError) -> str:
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
        except OSError:
            body = ""
        return body or exc.reason

    def _build_batch(
        self,
        symbols: tuple[str, ...],
        raw_bars: Mapping[str, list[dict[str, object]]],
    ) -> MarketDataBatch:
        frames: dict[str, pd.DataFrame] = {}
        quality: dict[str, Mapping[str, object]] = {}
        missing: list[str] = []
        for symbol in symbols:
            rows = raw_bars[symbol]
            if not rows:
                frames[symbol] = pd.DataFrame(columns=BAR_COLUMNS, index=pd.DatetimeIndex([], name="timestamp"))
                missing.append(symbol)
                quality[symbol] = {
                    "bar_count": 0,
                    "first_timestamp": None,
                    "last_timestamp": None,
                    "errors": ["No bars returned for requested symbol in lookback window."],
                }
                continue
            frame = self._normalize_symbol_bars(symbol, rows)
            frames[symbol] = frame
            quality[symbol] = {
                "bar_count": len(frame),
                "first_timestamp": frame.index[0].isoformat(),
                "last_timestamp": frame.index[-1].isoformat(),
                "errors": [],
            }
        return MarketDataBatch(
            bars=frames,
            requested_symbols=symbols,
            missing_symbols=tuple(missing),
            quality=quality,
        )

    @staticmethod
    def _normalize_symbol_bars(
        symbol: str,
        rows: list[dict[str, object]],
    ) -> pd.DataFrame:
        normalized_rows: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, dict):
                raise MarketDataError(f"Alpaca returned a malformed bar for {symbol}.")
            try:
                timestamp = pd.to_datetime(row["t"], utc=True, errors="raise")
                if pd.isna(timestamp):
                    raise ValueError("timestamp is missing")
                values = {
                    "open": float(row["o"]),
                    "high": float(row["h"]),
                    "low": float(row["l"]),
                    "close": float(row["c"]),
                    "volume": float(row["v"]),
                }
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise MarketDataError(f"Alpaca returned an incomplete or non-numeric bar for {symbol}.") from exc
            if not all(isfinite(value) for value in values.values()):
                raise MarketDataError(f"Alpaca returned non-finite values for {symbol}.")
            if any(values[name] <= 0 for name in ("open", "high", "low", "close")):
                raise MarketDataError(f"Alpaca returned non-positive prices for {symbol}.")
            if values["volume"] < 0:
                raise MarketDataError(f"Alpaca returned negative volume for {symbol}.")
            if (
                values["high"] < values["low"]
                or not values["low"] <= values["open"] <= values["high"]
                or not values["low"] <= values["close"] <= values["high"]
            ):
                raise MarketDataError(f"Alpaca returned an invalid OHLC range for {symbol}.")
            normalized_rows.append({"timestamp": timestamp, **values})
        frame = pd.DataFrame(normalized_rows).set_index("timestamp")
        if frame.index.has_duplicates:
            raise MarketDataError(f"Alpaca returned duplicate timestamps for {symbol}.")
        frame.index = pd.DatetimeIndex(frame.index, name="timestamp")
        return frame.loc[:, BAR_COLUMNS].sort_index()

    @staticmethod
    def _copy_batch(batch: MarketDataBatch, *, from_cache: bool) -> MarketDataBatch:
        return MarketDataBatch(
            bars={symbol: frame.copy(deep=True) for symbol, frame in batch.bars.items()},
            requested_symbols=batch.requested_symbols,
            missing_symbols=batch.missing_symbols,
            quality=deepcopy(batch.quality),
            from_cache=from_cache,
        )
