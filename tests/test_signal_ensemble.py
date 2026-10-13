import pytest

from src.strategy.signal_ensemble import score_components


def test_buy_signal_when_weighted_score_clears_threshold():
    result = score_components({
        "trend": 1.0, "momentum": 0.8, "relative_strength": 0.5,
        "volume": 0.4, "news_sentiment": 0.2,
    })
    assert result.action == "BUY"
    assert result.score > 0.45
    assert pytest.approx(sum(result.effective_weights.values()), abs=1e-7) == 1.0


def test_missing_component_weights_are_renormalized():
    result = score_components({"trend": 1.0, "momentum": None})
    assert result.score == 1.0
    assert result.effective_weights == {"trend": 1.0}


def test_negative_signal_does_not_short_unheld_stock():
    result = score_components({"trend": -1.0, "momentum": -1.0}, held=False)
    assert result.action == "HOLD"
    assert "short position" in result.reason


def test_negative_signal_sells_held_stock():
    result = score_components({"trend": -1.0, "momentum": -1.0}, held=True)
    assert result.action == "SELL"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.1, 1.1, "not-a-number"])
def test_invalid_component_is_rejected(bad):
    with pytest.raises(ValueError):
        score_components({"trend": bad})


def test_invalid_weight_is_rejected():
    with pytest.raises(ValueError):
        score_components({"trend": 1.0}, weights={"trend": -0.5})
