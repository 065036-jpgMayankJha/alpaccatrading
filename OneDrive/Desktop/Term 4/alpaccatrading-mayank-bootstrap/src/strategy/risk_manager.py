"""Deterministic pre-trade risk checks. This module does not submit orders."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RiskConfig:
    cash_reserve_pct: float = 0.10
    max_position_pct: float = 0.05
    max_new_orders_per_run: int = 5
    allow_short_selling: bool = False
    paper_only: bool = True
    require_human_approval: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.cash_reserve_pct < 1:
            raise ValueError("cash_reserve_pct must be in [0, 1).")
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1].")
        if self.max_new_orders_per_run < 0:
            raise ValueError("max_new_orders_per_run must be non-negative.")


@dataclass(frozen=True)
class RiskDecision:
    symbol: str
    side: str
    notional: float
    approved: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric, non-negative value.") from exc
    if not isfinite(number) or number < 0:
        raise ValueError(f"{label} must be finite and non-negative.")
    return number


def validate_order_proposals(
    orders: Sequence[Mapping[str, Any]],
    *,
    equity: float,
    cash: float,
    positions_market_value: Mapping[str, float] | None = None,
    open_order_symbols: Sequence[str] = (),
    config: RiskConfig | None = None,
) -> list[RiskDecision]:
    """Return an approve/reject decision per proposal; never places an order.

    Order proposal shape: {"symbol": "AAPL", "side": "buy"|"sell",
    "notional": 1000}. Sell proposals are allowed only against existing holdings;
    quantities, fills, and exact broker constraints must be checked by the executor.
    """
    cfg = config or RiskConfig()
    total_equity = _finite_nonnegative(equity, "equity")
    available_cash = _finite_nonnegative(cash, "cash")
    positions = {str(k).upper(): _finite_nonnegative(v, f"position {k}")
                 for k, v in (positions_market_value or {}).items()}
    open_symbols = {str(s).strip().upper() for s in open_order_symbols}
    if total_equity <= 0:
        raise ValueError("equity must be greater than zero.")

    reserve = total_equity * cfg.cash_reserve_pct
    position_cap = total_equity * cfg.max_position_pct
    max_spendable_cash = max(0.0, available_cash - reserve)
    buys_seen: set[str] = set()
    accepted_buy_notional = 0.0
    buy_count = 0
    decisions: list[RiskDecision] = []

    for order in orders:
        raw_symbol = str(order.get("symbol", "")).strip().upper()
        raw_side = str(order.get("side", "")).strip().lower()
        try:
            notional = _finite_nonnegative(order.get("notional", 0), "notional")
        except ValueError as exc:
            decisions.append(RiskDecision(raw_symbol or "UNKNOWN", raw_side, 0.0, False, (str(exc),)))
            continue
        reasons: list[str] = []
        if not raw_symbol or not raw_symbol.replace(".", "").isalnum():
            reasons.append("Invalid or missing symbol.")
        if raw_side not in {"buy", "sell"}:
            reasons.append("Side must be 'buy' or 'sell'.")
        if notional <= 0:
            reasons.append("Notional must be greater than zero.")
        held_value = positions.get(raw_symbol, 0.0)
        if raw_side == "sell" and not cfg.allow_short_selling and held_value <= 0:
            reasons.append("Sell rejected: no held position and short selling is disabled.")
        elif raw_side == "sell" and not cfg.allow_short_selling and notional > held_value + 1e-8:
            reasons.append(f"Sell rejected: notional exceeds held position market value ({held_value:.2f} USD); short selling is disabled.")
        if raw_symbol in open_symbols:
            reasons.append("An open order already exists for this symbol.")
        if raw_side == "buy" and raw_symbol in buys_seen:
            reasons.append("Duplicate buy proposal for this symbol in the current run.")

        if raw_side == "buy" and not reasons:
            buy_count += 1
            if buy_count > cfg.max_new_orders_per_run:
                reasons.append("Maximum new orders per run exceeded.")
            current_value = positions.get(raw_symbol, 0.0)
            if current_value + notional > position_cap + 1e-8:
                reasons.append(f"Position cap exceeded ({position_cap:.2f} USD).")
            if accepted_buy_notional + notional > max_spendable_cash + 1e-8:
                reasons.append(f"Cash reserve would be breached; max spendable cash is {max_spendable_cash:.2f} USD.")
            if not reasons:
                accepted_buy_notional += notional
                buys_seen.add(raw_symbol)
        elif raw_side == "buy":
            # Avoid counting invalid proposals toward available cash; still prevent duplicates later.
            if raw_symbol:
                buys_seen.add(raw_symbol)

        decisions.append(RiskDecision(
            symbol=raw_symbol or "UNKNOWN",
            side=raw_side,
            notional=round(notional, 2),
            approved=not reasons,
            reasons=tuple(reasons) if reasons else ("Passed configured pre-trade checks; human approval and broker validation still required.",),
        ))
    return decisions
