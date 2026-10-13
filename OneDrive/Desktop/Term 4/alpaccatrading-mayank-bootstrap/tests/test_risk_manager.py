import pytest

from src.strategy.risk_manager import RiskConfig, validate_order_proposals


def test_buy_inside_cash_position_limits_is_approved_but_not_executed():
    decisions = validate_order_proposals(
        [{"symbol": "AAPL", "side": "buy", "notional": 4000}],
        equity=100000, cash=100000, config=RiskConfig(),
    )
    assert decisions[0].approved is True
    assert "human approval" in decisions[0].reasons[0]


def test_position_cap_rejects_oversized_buy():
    decision = validate_order_proposals(
        [{"symbol": "AAPL", "side": "buy", "notional": 6000}],
        equity=100000, cash=100000,
    )[0]
    assert not decision.approved
    assert any("Position cap" in reason for reason in decision.reasons)


def test_reserve_prevents_overspending():
    decision = validate_order_proposals(
        [{"symbol": "AAPL", "side": "buy", "notional": 5000}],
        equity=100000, cash=14000,
    )[0]
    assert not decision.approved
    assert any("Cash reserve" in reason for reason in decision.reasons)


def test_unheld_sell_rejected_when_shorting_disabled():
    decision = validate_order_proposals(
        [{"symbol": "AAPL", "side": "sell", "notional": 1000}],
        equity=100000, cash=100000,
    )[0]
    assert not decision.approved
    assert any("short selling is disabled" in reason for reason in decision.reasons)


def test_open_order_and_duplicate_buy_are_rejected():
    decisions = validate_order_proposals(
        [
            {"symbol": "AAPL", "side": "buy", "notional": 1000},
            {"symbol": "AAPL", "side": "buy", "notional": 1000},
            {"symbol": "MSFT", "side": "buy", "notional": 1000},
        ],
        equity=100000, cash=100000,
        open_order_symbols=["MSFT"],
    )
    assert decisions[0].approved
    assert not decisions[1].approved
    assert not decisions[2].approved


def test_non_positive_equity_raises():
    with pytest.raises(ValueError):
        validate_order_proposals([], equity=0, cash=0)


def test_sell_notional_cannot_exceed_position_when_shorting_disabled():
    decision = validate_order_proposals(
        [{"symbol": "AAPL", "side": "sell", "notional": 2500}],
        equity=100000, cash=50000, positions_market_value={"AAPL": 2000},
    )[0]
    assert not decision.approved
    assert any("exceeds held position" in reason for reason in decision.reasons)
