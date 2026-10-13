import asyncio
import json
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import numpy as np

from servers.technical_indicators_mcp_server import (
    create_server,
)
from src.data.alpaca_market_data import AlpacaMarketDataClient


class MockResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.payload


def test_mcp_tool_input_validation_returns_structured_error():
    server = create_server(AlpacaMarketDataClient(api_key="k", secret_key="s"))
    _, response = asyncio.run(
        server.call_tool("fetch_ohlcv", {"symbols": [], "timeframe": "1Day"})
    )
    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_input"
    assert "symbols" in response["error"]["message"]

    _, features = asyncio.run(
        server.call_tool(
            "get_technical_features",
            {"symbols": ["AAPL"], "timeframe": "daily", "lookback": "30 days"},
        )
    )
    assert features["ok"] is False
    assert features["error"]["code"] == "invalid_input"
    assert "timeframe" in features["error"]["message"]


def test_mcp_fetch_tool_returns_compact_batch_and_missing_coverage():
    def transport(request, _timeout):
        symbols = parse_qs(urlparse(request.full_url).query)["symbols"][0].split(",")
        return MockResponse({"bars": {symbols[0]: [{
            "t": "2026-10-01T00:00:00Z",
            "o": 100,
            "h": 101,
            "l": 99,
            "c": 100,
            "v": 500,
        }]}})

    client = AlpacaMarketDataClient(
        api_key="k",
        secret_key="s",
        transport=transport,
        clock=lambda: datetime(2026, 10, 14, tzinfo=timezone.utc),
    )
    server = create_server(client)
    _, response = asyncio.run(
        server.call_tool(
            "fetch_ohlcv",
            {"symbols": ["AAPL", "MSFT"], "timeframe": "1Day", "lookback": "30 days"},
        )
    )
    assert response["ok"] is True
    assert response["coverage"]["missing_symbols"] == ["MSFT"]
    assert response["symbols"]["AAPL"]["values"] == [[100, 101, 99, 100, 500]]
    assert response["symbols"]["MSFT"]["quality"]["errors"]


def test_mcp_feature_tool_uses_shared_indicator_engine_and_feature_cache():
    timestamps = np.arange(
        np.datetime64("2026-08-01"),
        np.datetime64("2026-10-01"),
        np.timedelta64(1, "D"),
    )[:60]
    rows_by_symbol = {}
    for symbol, prices in (
        ("AAPL", np.linspace(100.0, 130.0, len(timestamps))),
        ("SPY", np.linspace(100.0, 110.0, len(timestamps))),
    ):
        rows_by_symbol[symbol] = [
            {
                "t": f"{stamp}T00:00:00Z",
                "o": float(price),
                "h": float(price + 1),
                "l": float(price - 1),
                "c": float(price),
                "v": 1000,
            }
            for stamp, price in zip(timestamps, prices)
        ]
    requests = []

    def transport(request, _timeout):
        requests.append(request)
        symbols = parse_qs(urlparse(request.full_url).query)["symbols"][0].split(",")
        return MockResponse({"bars": {symbol: rows_by_symbol[symbol] for symbol in symbols}})

    client = AlpacaMarketDataClient(
        api_key="k",
        secret_key="s",
        transport=transport,
        clock=lambda: datetime(2026, 10, 14, tzinfo=timezone.utc),
    )
    server = create_server(client)

    async def read_features():
        first = await server.call_tool(
            "get_technical_features",
            {"symbols": ["AAPL"], "timeframe": "1Day", "lookback": "90 days"},
        )
        second = await server.call_tool(
            "get_technical_features",
            {"symbols": ["AAPL"], "timeframe": "1Day", "lookback": "90 days"},
        )
        return first[1], second[1]

    first, second = asyncio.run(read_features())

    assert len(requests) == 1
    requested = parse_qs(urlparse(requests[0].full_url).query)["symbols"][0]
    assert requested == "AAPL,SPY"
    assert first["ok"] is True
    assert first["features"]["AAPL"]["ok"] is True
    assert first["features"]["AAPL"]["indicators"]["atr_pct"] > 0
    assert -1 <= first["features"]["AAPL"]["scores"]["relative_strength"] <= 1
    assert second["from_feature_cache"] is True


def test_fastmcp_server_registers_both_tools():
    server = create_server(
        AlpacaMarketDataClient(api_key="k", secret_key="s")
    )
    assert server.name == "alpaca-technical-indicators"
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == {
        "fetch_ohlcv",
        "get_technical_features",
    }
