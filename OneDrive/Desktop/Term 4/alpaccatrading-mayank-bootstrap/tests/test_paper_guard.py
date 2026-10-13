import pytest

from src.execution.paper_guard import assert_paper_endpoint


def test_accepts_alpaca_paper_endpoint():
    assert assert_paper_endpoint("https://paper-api.alpaca.markets/v2") == "https://paper-api.alpaca.markets/v2"


@pytest.mark.parametrize("url", [
    "https://api.alpaca.markets",
    "http://paper-api.alpaca.markets",
    "https://paper-api.alpaca.markets.evil.example",
    "https://user:password@paper-api.alpaca.markets",
    "",
])
def test_rejects_unsafe_endpoints(url):
    with pytest.raises(ValueError):
        assert_paper_endpoint(url)
