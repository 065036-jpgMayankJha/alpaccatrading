import json
from pathlib import Path

from src.data.alpaca_market_data import AlpacaMarketDataClient


EXPECTED_SYMBOLS = {
    "AMZN", "NVDA", "AAPL", "MSFT", "GOOGL", "TSLA", "AVGO", "BRK.B",
    "JPM", "MA", "COST", "AMD", "C", "BSX", "LMT", "WDC", "MU", "SNDK",
    "TTWO", "WBD", "LLY", "ASML", "NFLX", "META", "ORCL", "V", "CRM",
    "QCOM", "ADBE", "TSM", "UBER", "XOM", "CVX", "CAT", "GE", "UNH",
    "PANW", "CRWD", "HOOD", "COIN",
}


def load_universe_config():
    config_path = Path(__file__).parents[1] / "config" / "universe.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def test_universe_matches_adjusted_portfolio_and_classification_counts():
    universe = load_universe_config()
    growth = universe["growth_satellites"]
    diversified = [
        symbol
        for group in universe["diversified"].values()
        for symbol in group
    ]
    combined = growth + diversified
    assert len(growth) == 12
    assert len(diversified) == 28
    assert len(combined) == 40
    assert len(set(combined)) == 40
    assert set(combined) == EXPECTED_SYMBOLS
    assert "EA" not in combined
    assert "TTWO" in combined
    assert set(growth).isdisjoint(diversified)
    assert universe["target_symbol_count"] == 40
    assert universe["benchmark"] == "SPY"

    classification = universe["classification"]
    assert classification["status"] == "provisional"
    assert "not a measured beta ranking" in classification["methodology"]
    assert "no member is represented here as proven high-beta" in classification["beta_evidence"]
    assert "not been verified" in classification["index_membership_evidence"]
    assert "not a beta or high-growth classification" in classification["ttwo_basis"]

    adjustment = universe["portfolio_adjustments"][0]
    assert adjustment["original_ticker"] == "EA"
    assert adjustment["replacement_ticker"] == "TTWO"
    assert adjustment["effective_suspension_date"] == "2026-08-05"
    assert adjustment["sources"]["nasdaq_notice"] == (
        "https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2026-545"
    )
    assert adjustment["sources"]["ttwo_official_listing"] == (
        "https://www.take2games.com/ir"
    )


def test_alpaca_symbol_normalization_preserves_brk_dot_b():
    assert AlpacaMarketDataClient._validate_symbols(["brk.b"]) == ("BRK.B",)


def test_universe_loader_includes_ttwo_and_preserves_brk_b():
    from trading_crew import load_universe

    symbols, benchmark = load_universe()
    assert len(symbols) == 40
    assert "TTWO" in symbols
    assert "EA" not in symbols
    assert "BRK.B" in symbols
    assert benchmark == "SPY"
