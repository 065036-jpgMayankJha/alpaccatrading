"""Paper-only Alpaca account access and explicitly approved order submission."""

from __future__ import annotations

import json
import math
import os
import re
import socket
from datetime import datetime
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from src.execution.paper_guard import assert_paper_endpoint


class PaperExecutionError(RuntimeError):
    """Sanitized error from the paper Trading API."""

    def __init__(self, message: str, *, status_code: int | None = None):
        safe_message = message
        for name in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
            secret = os.getenv(name)
            if secret:
                safe_message = safe_message.replace(secret, "[REDACTED]")
        super().__init__(safe_message)
        self.status_code = status_code


class AlpacaPaperExecutor:
    """Access Alpaca account state and submit only explicitly approved paper orders.

    The URL is validated on every operation. Credentials are read as a matching
    ALPACA_ or APCA_ pair and are never included in errors or returned records.
    """

    DEFAULT_URL = "https://paper-api.alpaca.markets"
    APPROVAL_PHRASE = "APPROVE PAPER ORDER"
    _CLIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,48}$")

    def __init__(
        self,
        *,
        trading_url: str | None = None,
        api_key: str | None = None,
        secret_key: str | None = None,
        environ: Mapping[str, str] | None = None,
        transport: Callable[[Request, float], object] | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive.")
        self.trading_url = trading_url or os.getenv("ALPACA_TRADING_URL", self.DEFAULT_URL)
        self.trading_url = assert_paper_endpoint(self.trading_url)
        self._api_key, self._secret_key = self._credentials(api_key, secret_key, environ)
        self._transport = transport or self._default_transport
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _credentials(
        api_key: str | None,
        secret_key: str | None,
        environ: Mapping[str, str] | None,
    ) -> tuple[str, str]:
        env = os.environ if environ is None else environ
        if bool(api_key) != bool(secret_key):
            raise PaperExecutionError("Both explicit paper credential values must be provided together.")
        if api_key and secret_key:
            return api_key, secret_key
        for key_name, secret_name in (
            ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"),
            ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"),
        ):
            key = env.get(key_name)
            secret = env.get(secret_name)
            if key and secret:
                return key, secret
        raise PaperExecutionError(
            "Paper trading credentials are missing; configure a complete ALPACA_ or APCA_ credential pair."
        )

    @staticmethod
    def _default_transport(request: Request, timeout: float) -> object:
        return urlopen(request, timeout=timeout)

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Mapping[str, str | int] | None = None,
        body: Mapping[str, object] | None = None,
        allow_not_found: bool = False,
    ) -> Any:
        assert_paper_endpoint(self.trading_url)
        url = f"{self.DEFAULT_URL}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        payload = json.dumps(body, allow_nan=False).encode("utf-8") if body is not None else None
        request = Request(
            url,
            data=payload,
            method=method,
            headers={
                "APCA-API-KEY-ID": self._api_key,
                "APCA-API-SECRET-KEY": self._secret_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            response = self._transport(request, self.timeout_seconds)
            if hasattr(response, "__enter__"):
                with response as opened:
                    raw = opened.read()
            else:
                raw = response.read()
        except HTTPError as exc:
            if allow_not_found and exc.code == 404:
                return None
            raise PaperExecutionError(
                f"Paper Trading API request failed with HTTP {exc.code}.",
                status_code=exc.code,
            ) from None
        except (TimeoutError, socket.timeout):
            raise PaperExecutionError("Paper Trading API request timed out.") from None
        except URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise PaperExecutionError("Paper Trading API request timed out.") from None
            raise PaperExecutionError("Paper Trading API connection failed.") from None
        except OSError:
            raise PaperExecutionError("Paper Trading API connection failed.") from None
        try:
            return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            raise PaperExecutionError("Paper Trading API returned malformed JSON.") from None

    @staticmethod
    def _number(value: object, label: str, *, required: bool = True) -> float | None:
        if value is None and not required:
            return None
        if isinstance(value, bool):
            raise PaperExecutionError(f"Paper API response has invalid {label}.")
        try:
            result = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise PaperExecutionError(f"Paper API response has invalid {label}.") from None
        if not math.isfinite(result):
            raise PaperExecutionError(f"Paper API response has invalid {label}.")
        return result

    @staticmethod
    def _boolean(value: object, label: str) -> bool:
        if not isinstance(value, bool):
            raise PaperExecutionError(f"Paper API response has invalid {label}.")
        return value

    def _optional_legacy_number(
        self, payload: Mapping[str, object], field: str, *, integer: bool = False
    ) -> float | int | None:
        value = payload.get(field)
        if value is None:
            return None
        try:
            parsed = self._number(value, field)
        except PaperExecutionError:
            return None
        if parsed is None or parsed < 0 or (integer and not parsed.is_integer()):
            return None
        return int(parsed) if integer else parsed

    def get_account(self) -> dict[str, object]:
        payload = self._request("GET", "/v2/account")
        if not isinstance(payload, dict):
            raise PaperExecutionError("Paper API returned an invalid account response.")
        status = payload.get("status")
        if not isinstance(status, str) or not status:
            raise PaperExecutionError("Paper API response has invalid account status.")
        pattern_day_trader = payload.get("pattern_day_trader")
        if not isinstance(pattern_day_trader, bool):
            pattern_day_trader = None
        return {
            "status": status,
            "currency": str(payload.get("currency", "")),
            "equity": self._number(payload.get("equity"), "account equity"),
            "cash": self._number(payload.get("cash"), "account cash"),
            "buying_power": self._number(payload.get("buying_power"), "buying power"),
            "trading_blocked": self._boolean(payload.get("trading_blocked"), "trading_blocked"),
            "account_blocked": self._boolean(payload.get("account_blocked"), "account_blocked"),
            "trade_suspended_by_user": self._boolean(payload.get("trade_suspended_by_user"), "trade_suspended_by_user"),
            "shorting_enabled": self._boolean(payload.get("shorting_enabled"), "shorting_enabled"),
            "pattern_day_trader": pattern_day_trader,
            "daytrade_count": self._optional_legacy_number(payload, "daytrade_count", integer=True),
            "daytrading_buying_power": self._optional_legacy_number(payload, "daytrading_buying_power"),
        }

    def get_positions(self) -> list[dict[str, object]]:
        payload = self._request("GET", "/v2/positions")
        if not isinstance(payload, list):
            raise PaperExecutionError("Paper API returned an invalid positions response.")
        result: list[dict[str, object]] = []
        for position in payload:
            if not isinstance(position, dict) or not isinstance(position.get("symbol"), str):
                raise PaperExecutionError("Paper API returned an invalid position.")
            qty = self._number(position.get("qty"), "position quantity")
            result.append({
                "symbol": str(position["symbol"]).strip().upper(),
                "qty": qty,
                "side": str(position.get("side", "")),
                "market_value": self._number(position.get("market_value"), "position market value"),
                "current_price": self._number(position.get("current_price"), "position current price", required=False),
            })
        return result

    def get_open_orders(self) -> list[dict[str, object]]:
        payloads: list[object] = []
        page_token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            query: dict[str, str | int] = {"status": "open", "limit": 500}
            if page_token:
                query["page_token"] = page_token
            payload = self._request("GET", "/v2/orders", query=query)
            if not isinstance(payload, list):
                raise PaperExecutionError("Paper API returned an invalid open-orders response.")
            payloads.extend(payload)
            if len(payload) < 500:
                break
            last = payload[-1]
            if not isinstance(last, dict) or not isinstance(last.get("id"), str):
                raise PaperExecutionError("Open-order pagination did not include a usable page token.")
            page_token = last["id"]
            if page_token in seen_tokens:
                raise PaperExecutionError("Paper API returned a repeated open-order page token.")
            seen_tokens.add(page_token)
        result: list[dict[str, object]] = []
        for order in payloads:
            if not isinstance(order, dict) or not isinstance(order.get("symbol"), str):
                raise PaperExecutionError("Paper API returned an invalid open order.")
            result.append({
                "id": str(order.get("id", "")),
                "client_order_id": str(order.get("client_order_id", "")),
                "symbol": str(order["symbol"]).strip().upper(),
                "side": str(order.get("side", "")).lower(),
                "qty": self._number(order.get("qty"), "order quantity", required=False),
                "notional": self._number(order.get("notional"), "order notional", required=False),
                "type": str(order.get("type", "")),
                "time_in_force": str(order.get("time_in_force", "")),
                "status": str(order.get("status", "")),
            })
        return result

    def get_clock(self) -> dict[str, object]:
        payload = self._request("GET", "/v2/clock")
        if not isinstance(payload, dict) or not isinstance(payload.get("is_open"), bool):
            raise PaperExecutionError("Paper API returned an invalid market-clock response.")
        timestamps: dict[str, str] = {}
        for key in ("timestamp", "next_open", "next_close"):
            value = payload.get(key)
            if not isinstance(value, str):
                raise PaperExecutionError(f"Paper API response has invalid market-clock {key}.")
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                raise PaperExecutionError(f"Paper API response has invalid market-clock {key}.") from None
            if parsed.tzinfo is None:
                raise PaperExecutionError(f"Paper API response has invalid market-clock {key}.")
            timestamps[key] = parsed.isoformat()
        return {
            "is_open": payload["is_open"],
            **timestamps,
        }

    def get_account_snapshot(self) -> dict[str, object]:
        return {
            "account": self.get_account(),
            "positions": self.get_positions(),
            "open_orders": self.get_open_orders(),
            "clock": self.get_clock(),
        }

    def get_order_by_client_order_id(self, client_order_id: str) -> dict[str, object] | None:
        if not isinstance(client_order_id, str) or not self._CLIENT_ID_PATTERN.fullmatch(client_order_id):
            raise ValueError("client_order_id must contain 1 to 48 characters.")
        payload = self._request(
            "GET",
            "/v2/orders:by_client_order_id",
            query={"client_order_id": client_order_id},
            allow_not_found=True,
        )
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise PaperExecutionError("Paper API returned an invalid order lookup response.")
        return self._safe_order_result(payload)

    @staticmethod
    def _safe_order_result(payload: Mapping[str, object]) -> dict[str, object]:
        return {
            "id": str(payload.get("id", "")),
            "client_order_id": str(payload.get("client_order_id", "")),
            "symbol": str(payload.get("symbol", "")).strip().upper(),
            "side": str(payload.get("side", "")).lower(),
            "qty": AlpacaPaperExecutor._number(payload.get("qty"), "order quantity", required=False),
            "notional": AlpacaPaperExecutor._number(payload.get("notional"), "order notional", required=False),
            "type": str(payload.get("type", "")),
            "time_in_force": str(payload.get("time_in_force", "")),
            "status": str(payload.get("status", "")),
        }

    def submit_order(
        self,
        proposal: Mapping[str, object],
        *,
        client_order_id: str,
        approval: str,
        risk_approved: bool,
        market_clock: Mapping[str, object],
    ) -> dict[str, object]:
        """Submit a DAY market order only after exact human/risk/clock checks."""
        assert_paper_endpoint(self.trading_url)
        if approval != self.APPROVAL_PHRASE:
            raise PaperExecutionError("Order not submitted: exact per-order human approval is required.")
        if not risk_approved:
            raise PaperExecutionError("Order not submitted: risk approval is required.")
        if market_clock.get("is_open") is not True:
            raise PaperExecutionError("Order not submitted: regular market session is closed.")
        symbol = str(proposal.get("symbol", "")).strip().upper()
        side = str(proposal.get("side", "")).strip().lower()
        if not symbol or side not in {"buy", "sell"}:
            raise PaperExecutionError("Order not submitted: proposal symbol or side is invalid.")
        if not isinstance(client_order_id, str) or not self._CLIENT_ID_PATTERN.fullmatch(client_order_id):
            raise PaperExecutionError("Order not submitted: client_order_id must contain 1 to 48 characters.")
        body: dict[str, object] = {
            "symbol": symbol,
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "extended_hours": False,
            "client_order_id": client_order_id,
        }
        quantity = self._number(proposal.get("qty"), "proposal quantity", required=False)
        notional = self._number(proposal.get("notional"), "proposal notional", required=False)
        if (quantity is None) == (notional is None):
            raise PaperExecutionError("Order not submitted: provide exactly one of quantity or notional.")
        if quantity is not None:
            if quantity <= 0:
                raise PaperExecutionError("Order not submitted: quantity must be positive.")
            body["qty"] = self._format_number(quantity)
        if notional is not None:
            if notional <= 0:
                raise PaperExecutionError("Order not submitted: notional must be positive.")
            body["notional"] = self._format_number(notional)

        existing = self.get_order_by_client_order_id(client_order_id)
        if existing is not None:
            return {"outcome": "already_exists", "order": existing}
        try:
            payload = self._request("POST", "/v2/orders", body=body)
        except PaperExecutionError as exc:
            if exc.status_code is not None and 400 <= exc.status_code < 500 and exc.status_code not in {408, 429}:
                raise
            try:
                reconciled = self.get_order_by_client_order_id(client_order_id)
            except PaperExecutionError:
                return {"outcome": "unresolved", "client_order_id": client_order_id}
            if reconciled is not None:
                return {"outcome": "reconciled", "order": reconciled}
            return {"outcome": "unresolved", "client_order_id": client_order_id}
        if not isinstance(payload, dict):
            raise PaperExecutionError("Paper API returned an invalid order-submission response.")
        return {"outcome": "submitted", "order": self._safe_order_result(payload)}

    @staticmethod
    def _format_number(value: float) -> str:
        return f"{value:.8f}".rstrip("0").rstrip(".")
