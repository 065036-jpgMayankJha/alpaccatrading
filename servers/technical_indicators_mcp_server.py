"""Local stdio MCP tools for batched Alpaca bars and cached indicators."""

from __future__ import annotations

import json
import threading
import time
from mcp.server.fastmcp import FastMCP

from src.data.alpaca_market_data import (
    AlpacaMarketDataClient,
    MarketDataBatch,
    MarketDataError,
)
from src.strategy.technical_features import calculate_technical_features

_FEATURE_CACHE_TTL_SECONDS = 60.0


class TechnicalIndicatorTools:
    """Business logic used by the FastMCP tools, separately testable offline."""

    def __init__(self, client: AlpacaMarketDataClient) -> None:
        self.client = client
        self._feature_cache: dict[
            tuple[object, ...], tuple[float, dict[str, object]]
        ] = {}
        self._cache_lock = threading.RLock()

    def fetch_ohlcv(
        self,
        symbols: list[str],
        timeframe: str = "1Day",
        lookback: str = "90 days",
    ) -> dict[str, object]:
        """Return batched OHLCV in compact columnar form with data-quality info."""
        try:
            batch = self.client.fetch_bars(symbols, timeframe, lookback)
        except ValueError as exc:
            return self._error("invalid_input", str(exc))
        except MarketDataError as exc:
            return self._error(
                "market_data_error",
                str(exc),
                status_code=exc.status_code,
            )
        return self._serialize_batch(batch)

    def get_technical_features(
        self,
        symbols: list[str],
        timeframe: str = "1Day",
        lookback: str = "90 days",
    ) -> dict[str, object]:
        """Return normalized indicators, using the client's cached bars."""
        try:
            requested = self.client._validate_symbols(symbols)
            normalized_timeframe = self.client._validate_timeframe(timeframe)
            duration = self.client._parse_lookback(lookback)
        except ValueError as exc:
            return self._error("invalid_input", str(exc))

        cache_key = (
            requested,
            normalized_timeframe,
            duration.total_seconds(),
            self.client.feed,
        )
        with self._cache_lock:
            cached = self._feature_cache.get(cache_key)
            if cached and time.monotonic() - cached[0] < _FEATURE_CACHE_TTL_SECONDS:
                result = json.loads(json.dumps(cached[1]))
                result["from_feature_cache"] = True
                return result

        fetch_symbols = list(requested)
        if "SPY" not in fetch_symbols:
            fetch_symbols.append("SPY")
        try:
            batch = self.client.fetch_bars(fetch_symbols, timeframe, lookback)
        except ValueError as exc:
            return self._error("invalid_input", str(exc))
        except MarketDataError as exc:
            return self._error(
                "market_data_error",
                str(exc),
                status_code=exc.status_code,
            )

        benchmark_frame = batch.bars["SPY"]
        feature_results: dict[str, object] = {}
        for symbol in requested:
            frame = batch.bars[symbol]
            if frame.empty:
                feature_results[symbol] = {
                    "ok": False,
                    "error": "No bars returned for requested symbol in lookback window.",
                }
                continue
            if benchmark_frame.empty:
                feature_results[symbol] = {
                    "ok": False,
                    "error": "SPY benchmark bars are missing; relative strength unavailable.",
                }
                continue
            try:
                common_index = frame.index.intersection(benchmark_frame.index)
                aligned_frame = frame.loc[common_index]
                aligned_benchmark = benchmark_frame.loc[common_index, "close"]
                features = calculate_technical_features(
                    aligned_frame,
                    aligned_benchmark,
                )
            except ValueError as exc:
                feature_results[symbol] = {"ok": False, "error": str(exc)}
                continue
            feature_results[symbol] = {
                "ok": True,
                "indicators": dict(features.indicators),
                "scores": dict(features.scores),
            }

        result: dict[str, object] = {
            "ok": True,
            "features": feature_results,
            "coverage": {
                "requested_symbols": list(requested),
                "missing_symbols": [
                    symbol for symbol in requested if symbol in batch.missing_symbols
                ],
                "symbols": {
                    symbol: dict(batch.quality[symbol]) for symbol in requested
                },
                "benchmark": dict(batch.quality["SPY"]),
            },
            "from_feature_cache": False,
        }
        with self._cache_lock:
            self._feature_cache[cache_key] = (time.monotonic(), result)
        return json.loads(json.dumps(result))

    @staticmethod
    def _serialize_batch(batch: MarketDataBatch) -> dict[str, object]:
        data: dict[str, object] = {}
        for symbol, frame in batch.bars.items():
            data[symbol] = {
                "columns": ["open", "high", "low", "close", "volume"],
                "timestamps": [stamp.isoformat() for stamp in frame.index],
                "values": frame.to_numpy().tolist(),
                "quality": dict(batch.quality[symbol]),
            }
        return {
            "ok": True,
            "symbols": data,
            "coverage": {
                "requested_symbols": list(batch.requested_symbols),
                "missing_symbols": list(batch.missing_symbols),
                "available_count": len(batch.requested_symbols) - len(batch.missing_symbols),
            },
            "from_cache": batch.from_cache,
        }

    @staticmethod
    def _error(
        code: str,
        message: str,
        *,
        status_code: int | None = None,
    ) -> dict[str, object]:
        error: dict[str, object] = {"code": code, "message": message}
        if status_code is not None:
            error["status_code"] = status_code
        return {"ok": False, "error": error}


def create_server(client: AlpacaMarketDataClient | None = None) -> FastMCP:
    """Create an independently testable FastMCP server instance."""
    tools = TechnicalIndicatorTools(client or AlpacaMarketDataClient())
    server = FastMCP("alpaca-technical-indicators")

    @server.tool()
    def fetch_ohlcv(
        symbols: list[str],
        timeframe: str = "1Day",
        lookback: str = "90 days",
    ) -> dict[str, object]:
        """Fetch batched OHLCV bars. Rows are timestamped and include coverage."""
        return tools.fetch_ohlcv(symbols, timeframe, lookback)

    @server.tool()
    def get_technical_features(
        symbols: list[str],
        timeframe: str = "1Day",
        lookback: str = "90 days",
    ) -> dict[str, object]:
        """Return cached indicators and normalized scores for symbols vs SPY."""
        return tools.get_technical_features(symbols, timeframe, lookback)

    return server


mcp = create_server()


if __name__ == "__main__":
    mcp.run(transport="stdio")
