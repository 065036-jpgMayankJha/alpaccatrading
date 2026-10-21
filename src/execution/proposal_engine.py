"""Validate deterministic run summaries and build risk-checked paper proposals."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import yaml

from src.strategy.risk_manager import RiskConfig, validate_order_proposals
from src.strategy.signal_ensemble import score_components

_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")
_EASTERN = ZoneInfo("America/New_York")
_INDICATOR_FIELDS = {
    "close", "ema_20", "ema_50", "rsi_14", "macd_line", "macd_signal",
    "macd_histogram", "atr_14", "atr_pct", "relative_strength_pct",
    "relative_volume",
}
_COMPONENT_FIELDS = {"trend", "momentum", "relative_strength", "volume", "news_sentiment"}


class ProposalValidationError(ValueError):
    """Invalid run summary, strategy policy, or account snapshot."""

    def __init__(self, message: str):
        for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
            secret = os.getenv(name)
            if secret:
                message = message.replace(secret, "[REDACTED]")
        super().__init__(message)


def _parse_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ProposalValidationError(f"{label} must be an ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        raise ProposalValidationError(f"{label} must be an ISO-8601 timestamp.") from None
    if parsed.tzinfo is None:
        raise ProposalValidationError(f"{label} must include a timezone.")
    return parsed.astimezone(timezone.utc)


def _finite_number(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ProposalValidationError(f"{label} must be a finite number.")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ProposalValidationError(f"{label} must be a finite number.") from None
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ProposalValidationError(f"{label} must be a finite number" + (f" >= {minimum}." if minimum is not None else "."))
    return result


def _load_json_object(source: Mapping[str, object] | str | Path, label: str) -> dict[str, object]:
    if isinstance(source, Mapping):
        return dict(source)
    try:
        loaded = json.loads(Path(source).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProposalValidationError(f"Unable to load {label}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ProposalValidationError(f"{label} must be a JSON object.")
    return loaded


def load_policy(
    strategy_path: str | Path,
    universe_path: str | Path,
) -> tuple[dict[str, object], list[str], str, RiskConfig, float, float]:
    """Read deterministic strategy weights, the active universe and risk policy."""
    try:
        config = yaml.safe_load(Path(strategy_path).read_text(encoding="utf-8"))
        universe = json.loads(Path(universe_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ProposalValidationError(f"Unable to load paper-trading configuration: {exc}") from exc
    strategy = config.get("strategy") if isinstance(config, dict) else None
    risk = config.get("risk") if isinstance(config, dict) else None
    if not isinstance(strategy, dict) or not isinstance(risk, dict):
        raise ProposalValidationError("Strategy configuration must include strategy and risk objects.")
    weights = strategy.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ProposalValidationError("Strategy weights must be a non-empty mapping.")
    try:
        normalized_weights = {str(key): float(value) for key, value in weights.items()}
        buy_threshold = float(strategy["buy_threshold"])
        sell_threshold = float(strategy["sell_threshold"])
        cash_reserve_pct = float(risk["cash_reserve_pct"])
        max_position_pct = float(risk["max_position_pct"])
        max_orders = int(risk["max_new_orders_per_run"])
        buy_notional = float(risk["buy_notional_usd"])
        max_feature_age_hours = float(risk["max_feature_age_hours"])
        allow_short = risk["allow_short_selling"]
        paper_only = risk["paper_only"]
        human_approval = risk["require_human_approval"]
    except (KeyError, TypeError, ValueError):
        raise ProposalValidationError("Strategy or risk configuration has missing or invalid numeric fields.") from None
    if not all(math.isfinite(weight) and weight >= 0 for weight in normalized_weights.values()):
        raise ProposalValidationError("Strategy weights must be finite and non-negative.")
    if sum(normalized_weights.values()) <= 0:
        raise ProposalValidationError("At least one strategy weight must be greater than zero.")
    if not (-1 <= sell_threshold < buy_threshold <= 1):
        raise ProposalValidationError("Strategy thresholds must satisfy -1 <= sell < buy <= 1.")
    if allow_short is not False or paper_only is not True or human_approval is not True:
        raise ProposalValidationError("Paper policy must disable short selling and require paper-only human approval.")
    try:
        risk_config = RiskConfig(
            cash_reserve_pct=cash_reserve_pct,
            max_position_pct=max_position_pct,
            max_new_orders_per_run=max_orders,
            allow_short_selling=False,
            paper_only=True,
            require_human_approval=True,
        )
    except (TypeError, ValueError) as exc:
        raise ProposalValidationError(f"Invalid risk limits: {exc}") from exc
    if not math.isfinite(buy_notional) or buy_notional <= 0:
        raise ProposalValidationError("risk.buy_notional_usd must be finite and positive.")
    if not math.isfinite(max_feature_age_hours) or max_feature_age_hours <= 0:
        raise ProposalValidationError("risk.max_feature_age_hours must be finite and positive.")
    if not isinstance(universe, dict):
        raise ProposalValidationError("Universe configuration must be an object.")
    growth = universe.get("growth_satellites")
    diversified = universe.get("diversified")
    if not isinstance(growth, list) or not isinstance(diversified, dict):
        raise ProposalValidationError("Universe must contain growth_satellites and diversified groups.")
    diversified_symbols: list[object] = []
    for group in diversified.values():
        if not isinstance(group, list):
            raise ProposalValidationError("Every diversified group must be a list.")
        diversified_symbols.extend(group)
    symbols = list(growth) + diversified_symbols
    symbol_count = len(symbols)
    is_default_universe = (
        symbol_count == 40 and len(growth) == 12 and len(diversified_symbols) == 28
    )
    is_personal_portfolio = 10 <= symbol_count <= 15
    if (
        any(not isinstance(item, str) for item in symbols)
        or len(set(symbols)) != symbol_count
        or set(growth).intersection(diversified_symbols)
        or universe.get("target_symbol_count") != symbol_count
        or not (is_default_universe or is_personal_portfolio)
        or any(not isinstance(item, str) or not _SYMBOL_RE.fullmatch(item) for item in symbols)
    ):
        raise ProposalValidationError(
            "Universe must be the canonical 40-symbol universe or an individual portfolio "
            "with 10-15 unique valid tickers and a matching target_symbol_count."
        )
    benchmark = universe.get("benchmark")
    if not isinstance(benchmark, str) or benchmark.strip().upper() != "SPY":
        raise ProposalValidationError("Universe benchmark must remain SPY.")
    return (
        {
            "name": str(strategy.get("name", "unnamed")),
            "version": str(strategy.get("version", "unknown")),
            "weights": normalized_weights,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "max_feature_age_hours": max_feature_age_hours,
        },
        symbols,
        benchmark.strip().upper(),
        risk_config,
        buy_notional,
        max_feature_age_hours,
    )


def _validate_snapshot(snapshot: Mapping[str, object]) -> tuple[
    Mapping[str, object], dict[str, dict[str, object]], list[Mapping[str, object]], Mapping[str, object]
]:
    account = snapshot.get("account")
    positions = snapshot.get("positions")
    open_orders = snapshot.get("open_orders")
    clock = snapshot.get("clock")
    if not isinstance(account, Mapping) or not isinstance(positions, list) or not isinstance(open_orders, list) or not isinstance(clock, Mapping):
        raise ProposalValidationError("Account snapshot must contain account, positions, open_orders and clock.")
    normalized_positions: dict[str, dict[str, object]] = {}
    for raw_position in positions:
        if not isinstance(raw_position, Mapping):
            raise ProposalValidationError("Account snapshot contains an invalid position.")
        symbol = str(raw_position.get("symbol", "")).strip().upper()
        if not _SYMBOL_RE.fullmatch(symbol) or symbol in normalized_positions:
            raise ProposalValidationError("Account snapshot contains an invalid or duplicate position symbol.")
        qty = _finite_number(raw_position.get("qty"), f"{symbol} quantity")
        market_value = _finite_number(raw_position.get("market_value"), f"{symbol} market value")
        if qty < 0:
            raise ProposalValidationError(f"Short position detected for {symbol}; short selling is disabled.")
        if qty == 0 or market_value <= 0:
            raise ProposalValidationError(f"Position data is insufficient for {symbol}; refusing to size proposals.")
        normalized_positions[symbol] = {**dict(raw_position), "symbol": symbol, "qty": qty, "market_value": market_value}
    for order in open_orders:
        if not isinstance(order, Mapping) or not _SYMBOL_RE.fullmatch(str(order.get("symbol", "")).strip().upper()):
            raise ProposalValidationError("Account snapshot contains an invalid open order.")
    if not isinstance(clock.get("is_open"), bool):
        raise ProposalValidationError("Account snapshot market clock must include boolean is_open.")
    _parse_timestamp(clock.get("timestamp"), "market clock timestamp")
    for field in ("trading_blocked", "account_blocked", "trade_suspended_by_user"):
        if not isinstance(account.get(field), bool):
            raise ProposalValidationError(f"Account snapshot must include boolean {field}.")
    if not isinstance(account.get("status"), str):
        raise ProposalValidationError("Account snapshot must include account status.")
    return account, normalized_positions, open_orders, clock


def _load_ledger(path: str | Path) -> list[dict[str, object]]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProposalValidationError(f"Unable to read paper order ledger: {exc}") from exc
    if not isinstance(ledger, list):
        raise ProposalValidationError("Paper order ledger must be a JSON list.")
    if any(not isinstance(record, dict) for record in ledger):
        raise ProposalValidationError("Paper order ledger contains an invalid record.")
    return ledger


def _ledger_attempt_count(
    records: list[dict[str, object]],
    run_id: str,
    strategy: Mapping[str, object],
) -> int:
    return sum(
        1
        for record in records
        if record.get("submission_attempted") is True
        and record.get("run_id") == run_id
        and record.get("strategy_name") == strategy.get("name")
        and record.get("strategy_version") == strategy.get("version")
    )


def make_client_order_id(strategy_name: str, strategy_version: str, symbol: str, side: str, date: str) -> str:
    material = "|".join((strategy_name, strategy_version, symbol.upper(), side.lower(), date))
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    return f"paper-{digest}"


def _json_without_credentials(value: object) -> str:
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        secret = os.getenv(name)
        if secret:
            serialized = serialized.replace(secret, "[REDACTED]")
    return serialized


def _rejected_signal(
    symbol: str,
    action: str,
    score: object,
    reason: str,
    signal: Mapping[str, object] | None = None,
    *,
    validated_signal: bool = False,
) -> dict[str, object]:
    source = signal or {}
    safe_source: dict[str, object] = {}
    safe_score = (
        score
        if isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(float(score))
        else None
    )
    score_value = source.get("score")
    if (
        isinstance(score_value, (int, float))
        and not isinstance(score_value, bool)
        and math.isfinite(float(score_value))
    ):
        safe_source["score"] = score_value
    action_value = source.get("action")
    if isinstance(action_value, str) and action_value in {"BUY", "HOLD", "SELL", "UNAVAILABLE"}:
        safe_source["action"] = action_value
    quality_value = source.get("data_quality")
    if isinstance(quality_value, str) and quality_value in {"valid", "invalid", "missing", "unavailable", "UNAVAILABLE"}:
        safe_source["data_quality"] = quality_value
    for field in ("data_timestamp", "reason"):
        value = source.get(field)
        if isinstance(value, str):
            safe_source[field] = value
    for field, allowed in (("indicators", _INDICATOR_FIELDS), ("component_scores", _COMPONENT_FIELDS)):
        values = source.get(field)
        if isinstance(values, Mapping):
            safe_source[field] = {
                str(key): value
                for key, value in values.items()
                if key in allowed
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            }
    if validated_signal:
        final_signal = safe_source
        data_quality = str(safe_source.get("data_quality", "valid"))
        quality_reason = None
        raw_unvalidated_signal = None
    else:
        final_signal = {
            "action": "UNAVAILABLE",
            "data_quality": "unavailable",
            "quality_reason": reason,
        }
        data_quality = "unavailable"
        quality_reason = reason
        raw_unvalidated_signal = (
            {"status": "raw_unvalidated", "data": safe_source} if signal is not None else None
        )
    decision: dict[str, object] = {
        "symbol": symbol,
        "side": action.lower() if action in {"BUY", "SELL"} else None,
        "signal_action": action,
        "data_quality": data_quality,
        "signal_score": safe_score,
        "decision": "rejected",
        "risk_decision": {"approved": False, "reasons": [reason]},
        "proposal": None,
        "signal": final_signal,
    }
    if quality_reason is not None:
        decision["data_quality_reason"] = quality_reason
    if raw_unvalidated_signal is not None:
        decision["raw_unvalidated_signal"] = raw_unvalidated_signal
    return decision


def build_proposals(
    run_summary_source: Mapping[str, object] | str | Path,
    account_snapshot: Mapping[str, object],
    *,
    strategy_path: str | Path = "config/strategy.yaml",
    universe_path: str | Path = "config/universe.json",
    ledger_path: str | Path = "logs/paper_order_ledger.json",
    now: datetime | None = None,
) -> dict[str, object]:
    """Recompute signals and produce one explicit decision per portfolio symbol."""
    strategy, universe, _benchmark, risk_config, buy_notional, max_age_hours = load_policy(
        strategy_path, universe_path
    )
    summary = _load_json_object(run_summary_source, "run summary")
    run_id = summary.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ProposalValidationError("Run summary must include a non-empty run_id.")
    run_time = _parse_timestamp(summary.get("timestamp"), "run summary timestamp")
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        raise ProposalValidationError("Current time must include a timezone.")
    current_time = current_time.astimezone(timezone.utc)
    if run_time > current_time:
        raise ProposalValidationError("Run summary timestamp is in the future.")
    requested = summary.get("symbols_analysed")
    signals = summary.get("signals")
    if not isinstance(requested, list) or not requested or not isinstance(signals, Mapping):
        raise ProposalValidationError("Run summary must contain symbols_analysed and signals.")
    normalized_requested: list[str] = []
    for raw_symbol in requested:
        if not isinstance(raw_symbol, str):
            raise ProposalValidationError("Run summary symbols must be strings.")
        symbol = raw_symbol.strip().upper()
        if not _SYMBOL_RE.fullmatch(symbol) or symbol not in universe:
            raise ProposalValidationError(f"Run summary contains symbol outside configured universe: {symbol!r}.")
        if symbol in normalized_requested:
            raise ProposalValidationError(f"Run summary contains duplicate symbol {symbol!r}.")
        normalized_requested.append(symbol)
    normalized_signals: dict[str, object] = {}
    for raw_symbol, value in signals.items():
        if not isinstance(raw_symbol, str) or raw_symbol.strip().upper() not in normalized_requested:
            raise ProposalValidationError("Run summary signals contain an unrequested symbol.")
        normalized_symbol = raw_symbol.strip().upper()
        if normalized_symbol in normalized_signals:
            raise ProposalValidationError("Run summary signals contain duplicate normalized symbols.")
        normalized_signals[normalized_symbol] = value

    account, positions, open_orders, clock = _validate_snapshot(account_snapshot)
    equity = _finite_number(account.get("equity"), "account equity", minimum=0)
    cash = _finite_number(account.get("cash"), "account cash", minimum=0)
    buying_power = _finite_number(account.get("buying_power"), "account buying power", minimum=0)
    if equity <= 0:
        raise ProposalValidationError("Account equity must be greater than zero.")
    clock_time = _parse_timestamp(clock.get("timestamp"), "market clock timestamp")
    trading_date = clock_time.astimezone(_EASTERN).date().isoformat()
    held_values = {
        symbol: float(position["market_value"])
        for symbol, position in positions.items()
        if float(position["qty"]) > 0 and float(position["market_value"]) > 0
    }
    open_symbols = [str(order["symbol"]).strip().upper() for order in open_orders]
    ledger_records = _load_ledger(ledger_path)
    open_ids = {
        str(record["client_order_id"])
        for record in ledger_records
        if record.get("submission_attempted") is True
        and isinstance(record.get("client_order_id"), str)
    }
    previously_attempted = _ledger_attempt_count(ledger_records, run_id, strategy)

    account_restrictions: list[str] = []
    status = str(account.get("status", "")).upper()
    if status != "ACTIVE":
        account_restrictions.append(f"Account status is {status or 'unknown'}, not ACTIVE.")
    for field, label in (
        ("trading_blocked", "Trading is blocked on the account."),
        ("account_blocked", "The account is blocked."),
        ("trade_suspended_by_user", "Trading is suspended by the account holder."),
    ):
        if account.get(field) is True:
            account_restrictions.append(label)
    if clock.get("is_open") is not True:
        account_restrictions.append("Market is closed; paper order submission requires regular trading hours.")

    decisions: list[dict[str, object]] = []
    raw_proposals: list[dict[str, object]] = []
    raw_to_decision: list[int] = []
    for symbol in normalized_requested:
        raw_signal = normalized_signals.get(symbol)
        if not isinstance(raw_signal, Mapping):
            decisions.append(_rejected_signal(symbol, "UNAVAILABLE", None, "Missing signal record."))
            continue
        data_status = raw_signal.get("data_quality")
        try:
            supplied_score = _finite_number(raw_signal.get("score"), f"{symbol} signal score")
            if not -1 <= supplied_score <= 1:
                raise ProposalValidationError(f"{symbol} signal score must be within [-1, 1].")
        except ProposalValidationError as exc:
            decisions.append(_rejected_signal(symbol, "UNAVAILABLE", raw_signal.get("score"), str(exc), raw_signal))
            continue
        components = raw_signal.get("component_scores")
        indicators = raw_signal.get("indicators")
        if (
            data_status != "valid"
            or not isinstance(components, Mapping)
            or not components
            or not isinstance(indicators, Mapping)
            or not indicators
        ):
            decisions.append(_rejected_signal(symbol, "UNAVAILABLE", supplied_score, "Feature data is missing or not marked valid.", raw_signal))
            continue
        quality_details = raw_signal.get("quality_details")
        if isinstance(quality_details, Mapping):
            errors = quality_details.get("errors", [])
            if errors:
                decisions.append(_rejected_signal(symbol, "UNAVAILABLE", supplied_score, "Feature data quality reports validation errors.", raw_signal))
                continue
        invalid_component = None
        normalized_components: dict[str, float] = {}
        for name, value in components.items():
            try:
                numeric = _finite_number(value, f"{symbol} component {name}")
            except ProposalValidationError as exc:
                invalid_component = str(exc)
                break
            if not -1 <= numeric <= 1:
                invalid_component = f"{symbol} component {name} must be within [-1, 1]."
                break
            normalized_components[str(name)] = numeric
        if invalid_component:
            decisions.append(_rejected_signal(symbol, "UNAVAILABLE", supplied_score, invalid_component, raw_signal))
            continue
        timestamp = raw_signal.get("data_timestamp")
        try:
            feature_time = _parse_timestamp(timestamp, f"{symbol} data timestamp")
            age_hours = (current_time - feature_time).total_seconds() / 3600
            if feature_time > current_time:
                raise ProposalValidationError(
                    f"{symbol} feature timestamp is in the future."
                )
            if feature_time > run_time:
                raise ProposalValidationError(
                    f"{symbol} feature timestamp is later than the run summary timestamp."
                )
            if age_hours > max_age_hours:
                raise ProposalValidationError(
                    f"{symbol} feature data is stale ({age_hours:.1f} hours old; limit {max_age_hours:.1f})."
                )
        except ProposalValidationError as exc:
            decisions.append(_rejected_signal(symbol, "UNAVAILABLE", supplied_score, str(exc), raw_signal))
            continue
        for key, value in indicators.items():
            if key not in _INDICATOR_FIELDS:
                decisions.append(_rejected_signal(symbol, "UNAVAILABLE", supplied_score, "Feature data contains an unexpected indicator field.", raw_signal))
                break
            try:
                _finite_number(value, f"{symbol} indicator {key}")
            except ProposalValidationError as exc:
                decisions.append(_rejected_signal(symbol, "UNAVAILABLE", supplied_score, str(exc), raw_signal))
                break
        else:
            position = positions.get(symbol)
            held = bool(position and float(position["qty"]) > 0 and float(position["market_value"]) > 0)
            try:
                result = score_components(
                    normalized_components,
                    strategy["weights"],  # type: ignore[arg-type]
                    buy_threshold=float(strategy["buy_threshold"]),
                    sell_threshold=float(strategy["sell_threshold"]),
                    held=held,
                )
            except (KeyError, TypeError, ValueError) as exc:
                decisions.append(_rejected_signal(symbol, "UNAVAILABLE", supplied_score, f"Unable to recompute deterministic signal: {exc}", raw_signal))
                continue
            signal = {
                "score": result.score,
                "action": result.action,
                "reason": result.reason,
                "component_scores": result.components_used,
                "effective_weights": result.effective_weights,
                "indicators": {key: value for key, value in indicators.items() if key in _INDICATOR_FIELDS},
                "data_timestamp": timestamp,
                "data_quality": data_status,
            }
            if result.action == "HOLD":
                decisions.append(_rejected_signal(
                    symbol, "HOLD", result.score,
                    "No proposal: deterministic signal is HOLD.",
                    signal, validated_signal=True,
                ))
                decisions[-1]["decision"] = "no_proposal"
                continue
            if result.action == "SELL":
                if not held or position is None:
                    decisions.append(_rejected_signal(
                        symbol, "SELL", result.score,
                        "Sell rejected: no verified long position is held.",
                        signal, validated_signal=True,
                    ))
                    continue
                qty = float(position["qty"])
                notional = float(position["market_value"])
                current_price = position.get("current_price")
                if qty <= 0 or notional <= 0 or current_price is None or float(current_price) <= 0:
                    decisions.append(_rejected_signal(
                        symbol, "SELL", result.score,
                        "Sell rejected: position quantity, market value or current price is unavailable.",
                        signal, validated_signal=True,
                    ))
                    continue
            else:
                qty = 0.0
                notional = buy_notional
                current_price = indicators.get("close")
            client_order_id = make_client_order_id(
                str(strategy["name"]), str(strategy["version"]), symbol, result.action, trading_date
            )
            if client_order_id in open_ids:
                decisions.append(_rejected_signal(
                    symbol, result.action, result.score,
                    "Rejected: this strategy/symbol/side already has a submitted or unresolved order in the local ledger for this US trading date.",
                    signal, validated_signal=True,
                ))
                continue
            proposal = {
                "symbol": symbol,
                "side": result.action.lower(),
                "notional": round(notional, 2),
                "qty": round(qty, 8) if qty else None,
                "signal_score": result.score,
                "signal_action": result.action,
                "signal_reason": result.reason,
                "indicators": {key: value for key, value in indicators.items() if key in _INDICATOR_FIELDS},
                "component_scores": result.components_used,
                "effective_weights": result.effective_weights,
                "data_timestamp": timestamp,
                "data_quality": data_status,
                "estimated_price": current_price,
                "existing_exposure": held_values.get(symbol, 0.0),
                "client_order_id": client_order_id,
            }
            decisions.append({
                "symbol": symbol,
                "side": proposal["side"],
                "signal_action": result.action,
                "signal_score": result.score,
                "decision": "pending_risk",
                "signal": signal,
                "proposal": proposal,
                "risk_decision": None,
            })
            raw_proposals.append({"symbol": symbol, "side": proposal["side"], "notional": notional})
            raw_to_decision.append(len(decisions) - 1)

    risk_decisions = validate_order_proposals(
        raw_proposals,
        equity=equity,
        cash=cash,
        positions_market_value=held_values,
        open_order_symbols=open_symbols,
        config=risk_config,
    )
    remaining_buying_power = buying_power
    approved_count = previously_attempted
    for raw_index, risk_decision in enumerate(risk_decisions):
        decision_index = raw_to_decision[raw_index]
        entry = decisions[decision_index]
        proposal = entry["proposal"]
        assert isinstance(proposal, dict)
        buying_power_before = remaining_buying_power
        reasons = list(risk_decision.reasons) if not risk_decision.approved else []
        if risk_decision.approved and risk_decision.side == "buy":
            if risk_decision.notional > remaining_buying_power + 1e-8:
                reasons.append(f"Buying-power limit exceeded; remaining buying power is {remaining_buying_power:.2f} USD.")
        if account_restrictions:
            reasons.extend(account_restrictions)
        if risk_decision.approved and not reasons:
            if approved_count >= risk_config.max_new_orders_per_run:
                reasons.append("Maximum new orders per run exceeded.")
            else:
                approved_count += 1
        approved = risk_decision.approved and not reasons
        if approved and risk_decision.side == "buy":
            remaining_buying_power -= risk_decision.notional
        entry["risk_decision"] = {
            "approved": approved,
            "reasons": reasons or ["Passed configured limits and current account buying-power check; human confirmation and market-open check remain required."],
            "notional": risk_decision.notional,
        }
        entry["decision"] = "accepted" if approved else "rejected"
        proposal["buying_power_remaining_before"] = round(buying_power_before, 2)
        proposal["cash_reserve_usd"] = round(equity * risk_config.cash_reserve_pct, 2)
        proposal["position_limit_usd"] = round(equity * risk_config.max_position_pct, 2)
        if approved and risk_decision.side == "buy" and remaining_buying_power < -1e-8:
            raise ProposalValidationError("Internal error: approved proposals exceed buying power.")

    return {
        "run_id": run_id,
        "account_snapshot": dict(account_snapshot),
        "strategy": strategy,
        "trading_date_us": trading_date,
        "market_open": clock["is_open"],
        "buying_power_start": buying_power,
        "buying_power_remaining": round(remaining_buying_power, 2),
        "cash_reserve_usd": round(equity * risk_config.cash_reserve_pct, 2),
        "position_limit_usd": round(equity * risk_config.max_position_pct, 2),
        "decisions": decisions,
    }


def write_audit_report(report: Mapping[str, object], directory: str | Path = "logs") -> Path:
    """Persist a sanitized local audit report in the ignored logs directory."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    run_id = hashlib.sha256(str(report.get("run_id", "run")).encode("utf-8")).hexdigest()[:12]
    path = target / f"paper-audit-{run_id}-{uuid.uuid4().hex[:8]}.json"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    serialized = _json_without_credentials(report)
    try:
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def append_submission_ledger(
    record: Mapping[str, object],
    path: str | Path = "logs/paper_order_ledger.json",
) -> None:
    """Atomically append safe whitelisted details for one attempted submission."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = {
        str(item["client_order_id"])
        for item in _load_ledger(target)
        if item.get("submission_attempted") is True
        and isinstance(item.get("client_order_id"), str)
    }
    client_id = str(record.get("client_order_id", ""))
    if not client_id or client_id in existing:
        raise ProposalValidationError("Submission ledger already contains this client_order_id.")
    try:
        prior = json.loads(target.read_text(encoding="utf-8")) if target.exists() else []
    except (OSError, json.JSONDecodeError) as exc:
        raise ProposalValidationError(f"Unable to read paper order ledger: {exc}") from exc
    if not isinstance(prior, list):
        raise ProposalValidationError("Paper order ledger must be a JSON list.")
    safe_keys = {
        "run_id", "strategy_name", "strategy_version", "symbol", "side", "notional",
        "qty", "client_order_id", "broker_order_id", "broker_status", "outcome",
        "operator_response", "submission_attempted", "timestamp",
    }
    safe_record = {key: record[key] for key in safe_keys if key in record}
    safe_record["submission_attempted"] = True
    prior.append(safe_record)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(_json_without_credentials(prior), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def update_submission_ledger(
    client_order_id: str,
    updates: Mapping[str, object],
    path: str | Path = "logs/paper_order_ledger.json",
) -> None:
    """Update the outcome of a previously persisted submission attempt."""
    target = Path(path)
    try:
        prior = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProposalValidationError(f"Unable to read paper order ledger: {exc}") from exc
    if not isinstance(prior, list):
        raise ProposalValidationError("Paper order ledger must be a JSON list.")
    safe_keys = {
        "broker_order_id", "broker_status", "outcome", "error", "timestamp",
    }
    match = next(
        (
            record for record in prior
            if isinstance(record, dict) and record.get("client_order_id") == client_order_id
        ),
        None,
    )
    if match is None or match.get("submission_attempted") is not True:
        raise ProposalValidationError("Submission ledger has no matching attempted order.")
    match.update({key: updates[key] for key in safe_keys if key in updates})
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(_json_without_credentials(prior), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
