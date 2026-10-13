"""Fail-closed endpoint guard for future paper-only order execution."""

from urllib.parse import urlparse

PAPER_HOST = "paper-api.alpaca.markets"


def assert_paper_endpoint(base_url: str) -> str:
    """Return a normalized paper endpoint or raise before any order call.

    This guard is necessary but not sufficient: credentials must also be paper
    credentials and the executor must require human approval and risk approval.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("Trading endpoint is missing; refusing to proceed.")
    parsed = urlparse(base_url.strip())
    if parsed.scheme != "https" or parsed.hostname != PAPER_HOST:
        raise ValueError("Paper-only safety guard: endpoint must use https://paper-api.alpaca.markets.")
    if parsed.username or parsed.password:
        raise ValueError("Credentials must not be embedded in the trading endpoint URL.")
    if parsed.query or parsed.fragment:
        raise ValueError("Trading endpoint URL must not contain a query or fragment.")
    return f"https://{PAPER_HOST}" + (parsed.path.rstrip("/") if parsed.path else "")
