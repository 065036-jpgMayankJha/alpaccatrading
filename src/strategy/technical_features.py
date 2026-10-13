"""Deterministic, completed-bar technical indicators for research signals.

The functions in this module only read the supplied history and return values
for its final row. They do not fetch data, submit orders, or create news scores.

Warm-up policy
--------------
At least 50 aligned bars are required (the EMA 50 warm-up). RSI and ATR use
14-period Wilder smoothing; MACD uses EMA(12), EMA(26), then EMA(9) of the
MACD line. Their early undefined values are not backfilled. A 20-period
relative-strength return and 20-prior-bar volume baseline need 21 rows; these
requirements are covered by the 50-bar minimum. All bars are assumed complete.

Score policy
------------
Each score is bounded to [-1, 1]. Trend is tanh(10 * (EMA20 / EMA50 - 1)).
Momentum is the mean of normalized RSI ((RSI - 50) / 50) and
tanh(100 * MACD-histogram / close). Relative strength is tanh(the 20-bar
stock-minus-benchmark return difference in percentage points / 5). Volume is
the sign of the latest close change multiplied by tanh(log(relative volume)),
where relative volume compares the latest bar with the preceding 20-bar mean.
Zero price change, zero volume baseline, or relative volume exactly one yields
a neutral volume score; unchanged gains and losses yield RSI 50. Thus an
indicator without directional evidence is neutral rather than imputed.

The returned ``scores`` mapping can be passed directly to
``score_components``. News sentiment is deliberately omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log, tanh
from typing import Mapping

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_TREND_FAST = 20
EMA_TREND_SLOW = 50
RELATIVE_STRENGTH_PERIOD = 20
RELATIVE_VOLUME_PERIOD = 20
MINIMUM_BARS = max(
    EMA_TREND_SLOW,
    RELATIVE_STRENGTH_PERIOD + 1,
    RELATIVE_VOLUME_PERIOD + 1,
)


@dataclass(frozen=True)
class TechnicalFeatures:
    """Latest raw indicator values and ensemble-compatible normalized scores."""

    indicators: Mapping[str, float]
    scores: Mapping[str, float]


def _validated_series(series: pd.Series, label: str, index: pd.Index) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise ValueError(f"{label} must be a pandas Series.")
    if not series.index.equals(index):
        raise ValueError(f"{label} timestamps must exactly match the OHLCV index.")
    try:
        values = pd.to_numeric(series, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values.") from exc
    if values.isna().any() or not values.map(isfinite).all():
        raise ValueError(f"{label} must not contain missing or non-finite values.")
    if (values <= 0).any():
        raise ValueError(f"{label} values must be greater than zero.")
    return values


def _bounded(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _wilder_average(values: pd.Series, period: int) -> pd.Series:
    """Apply Wilder smoothing with an initial simple-average seed."""
    result = pd.Series(float("nan"), index=values.index, dtype=float)
    first_valid = values.first_valid_index()
    if first_valid is None:
        return result
    start = values.index.get_loc(first_valid)
    seed_position = start + period - 1
    if seed_position >= len(values):
        return result
    result.iloc[seed_position] = float(values.iloc[start : seed_position + 1].mean())
    for position in range(seed_position + 1, len(values)):
        result.iloc[position] = (
            result.iloc[position - 1] * (period - 1) + float(values.iloc[position])
        ) / period
    return result


def calculate_technical_features(
    ohlcv: pd.DataFrame,
    benchmark_close: pd.Series,
) -> TechnicalFeatures:
    """Calculate terminal-bar indicators from aligned OHLCV and benchmark data.

    Args:
        ohlcv: Chronologically ordered completed bars with lowercase
            ``open``, ``high``, ``low``, ``close``, and ``volume`` columns.
        benchmark_close: Positive SPY close values with exactly the same index
            as ``ohlcv``; mismatched or missing timestamps are rejected.

    Raises:
        ValueError: If the data is malformed, misaligned, or shorter than the
            documented 50-bar warm-up requirement.
    """
    if not isinstance(ohlcv, pd.DataFrame):
        raise ValueError("ohlcv must be a pandas DataFrame.")
    if not ohlcv.columns.is_unique:
        raise ValueError("ohlcv column names must be unique.")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in ohlcv.columns]
    if missing_columns:
        raise ValueError(f"ohlcv is missing required columns: {', '.join(missing_columns)}.")
    if len(ohlcv) < MINIMUM_BARS:
        raise ValueError(
            f"At least {MINIMUM_BARS} completed OHLCV bars are required; "
            f"received {len(ohlcv)}."
        )
    if ohlcv.index.has_duplicates or not ohlcv.index.is_monotonic_increasing:
        raise ValueError("ohlcv timestamps must be unique and chronologically ordered.")

    try:
        data = ohlcv.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("OHLCV columns must contain numeric values.") from exc
    finite_values = data.apply(lambda column: column.map(isfinite))
    if data.isna().any().any() or not finite_values.all().all():
        raise ValueError("OHLCV data must not contain missing or non-finite values.")
    if (data.loc[:, ("open", "high", "low", "close")] <= 0).any().any():
        raise ValueError("OHLC price values must be greater than zero.")
    if (data["volume"] < 0).any():
        raise ValueError("OHLCV volume values must be non-negative.")
    invalid_ranges = (
        (data["high"] < data["low"])
        | (data["open"] < data["low"])
        | (data["open"] > data["high"])
        | (data["close"] < data["low"])
        | (data["close"] > data["high"])
    )
    if invalid_ranges.any():
        raise ValueError("Each bar must satisfy low <= open/close <= high.")

    benchmark = _validated_series(benchmark_close, "benchmark_close", data.index)

    close = data["close"]
    ema_20 = close.ewm(span=EMA_TREND_FAST, adjust=False, min_periods=EMA_TREND_FAST).mean()
    ema_50 = close.ewm(span=EMA_TREND_SLOW, adjust=False, min_periods=EMA_TREND_SLOW).mean()
    rsi_delta = close.diff()
    average_gain = _wilder_average(rsi_delta.clip(lower=0), RSI_PERIOD)
    average_loss = _wilder_average(-rsi_delta.clip(upper=0), RSI_PERIOD)
    gain = float(average_gain.iloc[-1])
    loss = float(average_loss.iloc[-1])
    if gain == 0 and loss == 0:
        rsi_14 = 50.0
    elif loss == 0:
        rsi_14 = 100.0
    elif gain == 0:
        rsi_14 = 0.0
    else:
        rsi_14 = 100.0 - 100.0 / (1.0 + gain / loss)

    macd_line = (
        close.ewm(span=MACD_FAST, adjust=False, min_periods=MACD_FAST).mean()
        - close.ewm(span=MACD_SLOW, adjust=False, min_periods=MACD_SLOW).mean()
    )
    macd_signal = macd_line.ewm(
        span=MACD_SIGNAL, adjust=False, min_periods=MACD_SIGNAL
    ).mean()
    macd_histogram = macd_line - macd_signal

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14 = _wilder_average(true_range, ATR_PERIOD)

    latest_close = float(close.iloc[-1])
    latest_ema_20 = float(ema_20.iloc[-1])
    latest_ema_50 = float(ema_50.iloc[-1])
    latest_macd_line = float(macd_line.iloc[-1])
    latest_macd_signal = float(macd_signal.iloc[-1])
    latest_macd_histogram = float(macd_histogram.iloc[-1])
    latest_atr = float(atr_14.iloc[-1])

    stock_return = latest_close / float(close.iloc[-RELATIVE_STRENGTH_PERIOD - 1]) - 1.0
    benchmark_return = (
        float(benchmark.iloc[-1])
        / float(benchmark.iloc[-RELATIVE_STRENGTH_PERIOD - 1])
        - 1.0
    )
    relative_strength_pct = (stock_return - benchmark_return) * 100.0

    prior_volume_average = float(data["volume"].iloc[-RELATIVE_VOLUME_PERIOD - 1 : -1].mean())
    latest_volume = float(data["volume"].iloc[-1])
    relative_volume = (
        latest_volume / prior_volume_average if prior_volume_average > 0 else 0.0
    )
    latest_price_change = latest_close - float(close.iloc[-2])
    volume_direction = int(latest_price_change > 0) - int(latest_price_change < 0)
    volume_score = (
        volume_direction * tanh(log(relative_volume))
        if relative_volume > 0
        else 0.0
    )

    trend_score = tanh(10.0 * (latest_ema_20 / latest_ema_50 - 1.0))
    rsi_score = (rsi_14 - 50.0) / 50.0
    macd_score = tanh(100.0 * latest_macd_histogram / latest_close)
    scores = {
        "trend": _bounded(trend_score),
        "momentum": _bounded((rsi_score + macd_score) / 2.0),
        "relative_strength": _bounded(tanh(relative_strength_pct / 5.0)),
        "volume": _bounded(volume_score),
    }
    indicators = {
        "close": latest_close,
        "ema_20": latest_ema_20,
        "ema_50": latest_ema_50,
        "rsi_14": rsi_14,
        "macd_line": latest_macd_line,
        "macd_signal": latest_macd_signal,
        "macd_histogram": latest_macd_histogram,
        "atr_14": latest_atr,
        "atr_pct": latest_atr / latest_close * 100.0,
        "relative_strength_pct": relative_strength_pct,
        "relative_volume": relative_volume,
    }
    if not all(isfinite(value) for value in (*indicators.values(), *scores.values())):
        raise ValueError("Indicator calculation produced a non-finite result.")
    return TechnicalFeatures(indicators=indicators, scores=scores)
