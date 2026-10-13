import json
from datetime import datetime, timezone
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from src.data.alpaca_market_data import AlpacaMarketDataClient, MarketDataError


class MockResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.payload


def bar(timestamp, price):
    return {
        "t": timestamp,
        "o": price,
        "h": price + 1,
        "l": price - 1,
        "c": price,
        "v": 1000,
    }


def make_client(transport, **kwargs):
    return AlpacaMarketDataClient(
        api_key="test-key",
        secret_key="test-secret",
        transport=transport,
        clock=lambda: datetime(2026, 10, 14, tzinfo=timezone.utc),
        sleeper=lambda _: None,
        **kwargs,
    )


def test_multi_symbol_batch_uses_one_endpoint_query_across_pages():
    requests = []
    responses = iter(
        [
            {
                "bars": {
                    "AAPL": [bar("2026-10-01T00:00:00Z", 100)],
                },
                "next_page_token": "next",
            },
            {
                "bars": {
                    "AAPL": [bar("2026-10-02T00:00:00Z", 101)],
                    "MSFT": [bar("2026-10-02T00:00:00Z", 200)],
                },
                "next_page_token": None,
            },
        ]
    )

    def transport(request, _timeout):
        requests.append(request)
        return MockResponse(next(responses))

    batch = make_client(transport).fetch_bars(["AAPL", "MSFT"], "1Day", "30 days")

    assert len(requests) == 2
    first_query = parse_qs(urlparse(requests[0].full_url).query)
    second_query = parse_qs(urlparse(requests[1].full_url).query)
    assert first_query["symbols"] == ["AAPL,MSFT"]
    assert first_query["timeframe"] == ["1Day"]
    assert second_query["page_token"] == ["next"]
    assert requests[0].get_header("Apca-api-key-id") == "test-key"
    assert batch.bars["AAPL"].columns.tolist() == ["open", "high", "low", "close", "volume"]
    assert isinstance(batch.bars["AAPL"].index, pd.DatetimeIndex)
    assert str(batch.bars["AAPL"].index.tz) == "UTC"
    assert batch.bars["AAPL"].iloc[-1]["close"] == 101
    assert batch.missing_symbols == ()


def test_missing_symbols_are_returned_as_empty_frames_with_quality_error():
    client = make_client(lambda *_: MockResponse({"bars": {"AAPL": []}}))
    batch = client.fetch_bars(["AAPL", "MSFT"], "1Day", 30)
    assert batch.bars["MSFT"].empty
    assert batch.missing_symbols == ("AAPL", "MSFT")
    assert batch.quality["MSFT"]["errors"]
    assert batch.quality["AAPL"]["bar_count"] == 0


def test_cache_reuses_batch_and_returns_defensive_copies():
    requests = []

    def transport(*args):
        requests.append(args[0])
        return MockResponse({"bars": {"AAPL": [bar("2026-10-01T00:00:00Z", 100)]}})

    client = make_client(transport)
    first = client.fetch_bars(["AAPL"], "1Day", "30 days")
    first.bars["AAPL"].iloc[0, 0] = -500
    second = client.fetch_bars(["AAPL"], "1Day", "30 days")
    assert len(requests) == 1
    assert second.from_cache
    assert second.bars["AAPL"].iloc[0, 0] == 100


def test_missing_credentials_are_reported_without_network_request(monkeypatch):
    for variable in (
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
    ):
        monkeypatch.delenv(variable, raising=False)
    calls = []
    client = AlpacaMarketDataClient(transport=lambda *args: calls.append(args))
    with pytest.raises(MarketDataError, match="credentials are missing"):
        client.fetch_bars(["AAPL"], "1Day", 30)
    assert calls == []


def test_credentials_and_feed_can_be_loaded_from_environment(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "environment-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "environment-secret")
    monkeypatch.setenv("ALPACA_DATA_FEED", "sip")
    requests = []

    def transport(request, _timeout):
        requests.append(request)
        return MockResponse({"bars": {"AAPL": []}})

    client = AlpacaMarketDataClient(
        transport=transport,
        clock=lambda: datetime(2026, 10, 14, tzinfo=timezone.utc),
    )
    client.fetch_bars(["AAPL"], "1Day", 30)
    assert requests[0].get_header("Apca-api-key-id") == "environment-key"
    assert requests[0].get_header("Apca-api-secret-key") == "environment-secret"
    assert parse_qs(urlparse(requests[0].full_url).query)["feed"] == ["sip"]


def test_http_errors_include_status_and_message():
    def transport(*_):
        raise HTTPError(
            "https://data.alpaca.markets/v2/stocks/bars",
            503,
            "Unavailable",
            {},
            BytesIO(b"service unavailable"),
        )

    with pytest.raises(MarketDataError, match="HTTP 503") as caught:
        make_client(transport, max_retries=0).fetch_bars(["AAPL"], "1Day", 30)
    assert caught.value.status_code == 503


def test_invalid_api_credentials_report_authentication_status():
    def transport(*_):
        raise HTTPError(
            "https://data.alpaca.markets/v2/stocks/bars",
            401,
            "Unauthorized",
            {},
            BytesIO(b"invalid credentials"),
        )

    with pytest.raises(MarketDataError, match="HTTP 401") as caught:
        make_client(transport).fetch_bars(["AAPL"], "1Day", 30)
    assert caught.value.status_code == 401


def test_rate_limit_retries_after_retry_after_header():
    calls = []

    def transport(*_):
        calls.append(1)
        if len(calls) == 1:
            raise HTTPError(
                "https://data.alpaca.markets/v2/stocks/bars",
                429,
                "Too Many Requests",
                {"Retry-After": "0"},
                None,
            )
        return MockResponse({"bars": {"AAPL": []}})

    batch = make_client(transport).fetch_bars(["AAPL"], "1Day", 30)
    assert len(calls) == 2
    assert batch.missing_symbols == ("AAPL",)


@pytest.mark.parametrize(
    ("symbols", "timeframe", "lookback", "message"),
    [
        ([], "1Day", 30, "non-empty list"),
        (["AAPL", "AAPL"], "1Day", 30, "Duplicate"),
        (["bad ticker"], "1Day", 30, "Invalid ticker"),
        (["AAPL"], "day", 30, "timeframe"),
        (["AAPL"], "1Day", 0, "positive duration"),
    ],
)
def test_invalid_request_parameters_are_rejected(symbols, timeframe, lookback, message):
    client = make_client(lambda *_: pytest.fail("Invalid input made a request"))
    with pytest.raises(ValueError, match=message):
        client.fetch_bars(symbols, timeframe, lookback)
