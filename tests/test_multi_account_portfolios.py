from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PORTFOLIOS = ROOT / "config" / "portfolios"


@pytest.mark.parametrize("owner", ["mayank", "aman", "aditi"])
def test_each_personal_portfolio_has_10_to_15_unique_symbols(owner: str) -> None:
    config = json.loads((PORTFOLIOS / f"{owner}.json").read_text(encoding="utf-8"))
    growth = config["growth_satellites"]
    diversified = [symbol for group in config["diversified"].values() for symbol in group]
    symbols = growth + diversified
    assert 10 <= len(symbols) <= 15
    assert len(symbols) == len(set(symbols))
    assert len(symbols) == config["target_symbol_count"]
    assert config["benchmark"] == "SPY"


def test_three_portfolios_partition_the_original_40_symbols() -> None:
    union: list[str] = []
    for owner in ("mayank", "aman", "aditi"):
        config = json.loads((PORTFOLIOS / f"{owner}.json").read_text(encoding="utf-8"))
        union.extend(config["growth_satellites"])
        union.extend(symbol for group in config["diversified"].values() for symbol in group)
    assert len(union) == 40
    assert len(set(union)) == 40


def test_proposal_engine_accepts_each_personal_portfolio() -> None:
    from src.execution.proposal_engine import load_policy
    for owner in ("mayank", "aman", "aditi"):
        result = load_policy(ROOT / "config" / "strategy.yaml", PORTFOLIOS / f"{owner}.json")
        symbols = result[1]
        assert 10 <= len(symbols) <= 15
        assert result[2] == "SPY"
