import numpy as np
import pandas as pd
import pytest

from src.strategy.signal_ensemble import score_components
from src.strategy.technical_features import calculate_technical_features


def market_data(prices, *, volumes=None, index=None):
    index = index if index is not None else pd.RangeIndex(len(prices))
    prices = np.asarray(prices, dtype=float)
    volume_values = np.asarray(volumes if volumes is not None else [1000] * len(prices))
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": volume_values,
        },
        index=index,
    )


def features_for(data, benchmark=None):
    if benchmark is None:
        benchmark = pd.Series([100.0] * len(data), index=data.index)
    return calculate_technical_features(data, benchmark)


def test_rising_prices_have_positive_trend_and_momentum():
    result = features_for(market_data(np.arange(100.0, 170.0)))
    assert result.scores["trend"] > 0
    assert result.scores["momentum"] > 0
    assert result.indicators["rsi_14"] == pytest.approx(100.0)


def test_falling_prices_have_negative_trend_and_momentum():
    result = features_for(market_data(np.arange(170.0, 100.0, -1.0)))
    assert result.scores["trend"] < 0
    assert result.scores["momentum"] < 0
    assert result.indicators["rsi_14"] == pytest.approx(0.0)


def test_flat_prices_are_neutral_for_directional_indicators():
    result = features_for(market_data([100.0] * 60))
    assert result.scores == {
        "trend": 0.0,
        "momentum": 0.0,
        "relative_strength": 0.0,
        "volume": 0.0,
    }
    assert result.indicators["rsi_14"] == 50.0
    assert result.indicators["macd_histogram"] == 0.0


def test_macd_and_atr_values_are_available_after_warmup():
    result = features_for(market_data(np.arange(100.0, 160.0)))
    assert result.indicators["macd_line"] > result.indicators["macd_signal"]
    assert result.indicators["macd_histogram"] > 0
    assert result.indicators["atr_14"] == pytest.approx(2.0)
    assert result.indicators["atr_pct"] == pytest.approx(2.0 / 159.0 * 100.0)


def test_relative_strength_compares_stock_return_to_spy():
    index = pd.RangeIndex(60)
    stock = market_data(np.linspace(100.0, 130.0, 60), index=index)
    benchmark = pd.Series(np.linspace(100.0, 110.0, 60), index=index)
    result = calculate_technical_features(stock, benchmark)
    assert result.indicators["relative_strength_pct"] > 0
    assert result.scores["relative_strength"] > 0


def test_relative_volume_compares_latest_bar_to_prior_20_bars():
    volumes = [100.0] * 59 + [200.0]
    result = features_for(market_data(np.arange(100.0, 160.0), volumes=volumes))
    assert result.indicators["relative_volume"] == pytest.approx(2.0)
    assert result.scores["volume"] > 0


def test_insufficient_history_is_rejected():
    data = market_data(np.arange(100.0, 149.0))
    with pytest.raises(ValueError, match="At least 50"):
        features_for(data)


def test_minimum_history_produces_warmed_up_values():
    result = features_for(market_data(np.arange(100.0, 150.0)))
    assert result.indicators["ema_50"] > 0
    assert result.indicators["rsi_14"] == 100.0
    assert result.indicators["atr_14"] == pytest.approx(2.0)
    assert all(np.isfinite(value) for value in result.indicators.values())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns="volume"), "missing required columns"),
        (
            lambda frame: frame.assign(close=["bad"] + frame["close"].iloc[1:].tolist()),
            "numeric values",
        ),
        (
            lambda frame: frame.assign(close=frame["close"].mask(frame.index == 0)),
            "missing or non-finite",
        ),
        (
            lambda frame: frame.assign(high=frame["low"] - 1),
            "low <= open/close <= high",
        ),
    ],
)
def test_malformed_price_series_are_rejected(mutate, message):
    data = mutate(market_data(np.arange(100.0, 160.0)))
    with pytest.raises(ValueError, match=message):
        features_for(data)


def test_missing_or_misaligned_benchmark_data_is_rejected():
    data = market_data(np.arange(100.0, 160.0))
    missing = pd.Series([100.0] * len(data), index=data.index)
    missing.iloc[-1] = float("nan")
    with pytest.raises(ValueError, match="benchmark_close"):
        calculate_technical_features(data, missing)

    misaligned = pd.Series([100.0] * len(data), index=pd.RangeIndex(1, 61))
    with pytest.raises(ValueError, match="timestamps must exactly match"):
        calculate_technical_features(data, misaligned)


def test_scores_are_bounded_and_compatible_with_existing_ensemble():
    result = features_for(market_data(np.linspace(75.0, 130.0, 60)))
    assert all(-1.0 <= score <= 1.0 for score in result.scores.values())
    ensemble = score_components(result.scores)
    assert set(ensemble.components_used) == set(result.scores)
    assert "news_sentiment" not in ensemble.components_used
