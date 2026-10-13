"""Deterministic composite-score logic; no LLM calculations are performed here."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Optional

DEFAULT_WEIGHTS: dict[str, float] = {
    "trend": 0.30,
    "momentum": 0.25,
    "relative_strength": 0.20,
    "volume": 0.10,
    "news_sentiment": 0.15,
}


@dataclass(frozen=True)
class SignalResult:
    """Reproducible signal for a single symbol."""

    score: float
    action: str
    components_used: dict[str, float]
    effective_weights: dict[str, float]
    reason: str


def _validate_score(name: str, value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Component {name!r} must be a numeric value in [-1, 1].") from exc
    if not isfinite(numeric) or not -1.0 <= numeric <= 1.0:
        raise ValueError(f"Component {name!r} must be finite and in [-1, 1].")
    return numeric


def score_components(
    components: Mapping[str, Optional[float]],
    weights: Optional[Mapping[str, float]] = None,
    *,
    buy_threshold: float = 0.45,
    sell_threshold: float = -0.45,
    held: bool = False,
) -> SignalResult:
    """Combine valid normalized components and choose BUY/SELL/HOLD.

    Missing/None components are excluded and remaining positive weights are
    renormalized. `held=False` never produces SELL (which avoids interpreting
    a negative signal as an instruction to open a short position).
    """
    weight_map = dict(weights or DEFAULT_WEIGHTS)
    if not weight_map:
        raise ValueError("At least one component weight is required.")
    for name, weight in weight_map.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Component names must be non-empty strings.")
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Weight for {name!r} must be numeric.") from exc
        if not isfinite(numeric_weight) or numeric_weight < 0:
            raise ValueError(f"Weight for {name!r} must be finite and non-negative.")
        weight_map[name] = numeric_weight
    if sum(weight_map.values()) <= 0:
        raise ValueError("At least one weight must be greater than zero.")
    if not (-1 <= sell_threshold < buy_threshold <= 1):
        raise ValueError("Thresholds must satisfy -1 <= sell < buy <= 1.")

    valid: dict[str, float] = {}
    for name, value in components.items():
        if value is None or name not in weight_map or weight_map[name] == 0:
            continue
        valid[name] = _validate_score(name, value)
    active_weight_sum = sum(weight_map[name] for name in valid)
    if active_weight_sum <= 0:
        raise ValueError("No valid components with positive configured weights were supplied.")

    effective = {name: weight_map[name] / active_weight_sum for name in valid}
    score = sum(valid[name] * effective[name] for name in valid)
    score = max(-1.0, min(1.0, score))

    if score >= buy_threshold:
        action = "BUY"
        reason = f"Composite score {score:.3f} meets buy threshold {buy_threshold:.3f}."
    elif score <= sell_threshold and held:
        action = "SELL"
        reason = f"Composite score {score:.3f} meets sell threshold {sell_threshold:.3f} for a held position."
    else:
        action = "HOLD"
        if score <= sell_threshold and not held:
            reason = f"Negative signal {score:.3f}; remain out rather than opening a short position."
        else:
            reason = f"Composite score {score:.3f} is inside the configured hold region."

    return SignalResult(
        score=round(score, 6),
        action=action,
        components_used={k: round(v, 6) for k, v in valid.items()},
        effective_weights={k: round(v, 8) for k, v in effective.items()},
        reason=reason,
    )
