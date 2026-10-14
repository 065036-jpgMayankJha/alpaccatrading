import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from paper_trade import main as paper_trade_main
from src.execution.alpaca_paper_executor import AlpacaPaperExecutor, PaperExecutionError
from src.execution.proposal_engine import (
    ProposalValidationError,
    append_submission_ledger,
    build_proposals,
    make_client_order_id,
    write_audit_report,
)


class MockResponse:
    def __init__(self, payload):
        self.raw = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.raw


def make_executor(transport, **kwargs):
    return AlpacaPaperExecutor(
        environ={
            "ALPACA_API_KEY": "dummy-paper-key",
            "ALPACA_SECRET_KEY": "dummy-paper-secret",
        },
        transport=transport,
        **kwargs,
    )


def account_payload(**overrides):
    result = {
        "id": "account-id",
        "status": "ACTIVE",
        "currency": "USD",
        "equity": "100000",
        "cash": "100000",
        "buying_power": "50000",
        "trading_blocked": False,
        "account_blocked": False,
        "trade_suspended_by_user": False,
        "shorting_enabled": True,
    }
    result.update(overrides)
    return result


def position(symbol="AAPL", qty="10", market_value="1000", current_price="100"):
    return {
        "symbol": symbol,
        "qty": qty,
        "side": "long",
        "market_value": market_value,
        "current_price": current_price,
    }


def snapshot(*, positions=None, orders=None, is_open=True, **account_overrides):
    current = datetime.now(timezone.utc)
    return {
        "account": account_payload(**account_overrides),
        "positions": positions or [],
        "open_orders": orders or [],
        "clock": {
            "is_open": is_open,
            "timestamp": current.isoformat(),
            "next_open": current.isoformat(),
            "next_close": (current + timedelta(hours=6)).isoformat(),
        },
    }


def run_summary(
    symbols=("AAPL",),
    *,
    components=None,
    data_quality="valid",
    data_timestamp=None,
    score=0.9,
):
    current = datetime.now(timezone.utc)
    components = components or {
        "trend": 1.0,
        "momentum": 1.0,
        "relative_strength": 1.0,
        "volume": 1.0,
    }
    return {
        "run_id": "test-run-id",
        "timestamp": current.isoformat(),
        "symbols_analysed": list(symbols),
        "signals": {
            symbol: {
                "score": score,
                "action": "BUY",
                "data_quality": data_quality,
                "data_timestamp": data_timestamp or (current - timedelta(hours=1)).isoformat(),
                "component_scores": dict(components),
                "indicators": {"close": 100.0, "atr_pct": 2.0},
                "quality_details": {"errors": []},
            }
            for symbol in symbols
        },
    }


def decisions_for(result):
    return {item["symbol"]: item for item in result["decisions"]}


def test_paper_executor_uses_paper_endpoint_and_supports_both_env_prefixes():
    requests = []

    def transport(request, _timeout):
        requests.append(request)
        return MockResponse(account_payload())

    executor = make_executor(transport)
    executor.get_account()
    assert requests[0].full_url == "https://paper-api.alpaca.markets/v2/account"
    assert requests[0].get_header("Apca-api-key-id") == "dummy-paper-key"
    apca = AlpacaPaperExecutor(
        environ={"APCA_API_KEY_ID": "dummy-apca-key", "APCA_API_SECRET_KEY": "dummy-apca-secret"},
        transport=transport,
    )
    assert apca.get_account()["status"] == "ACTIVE"


def test_paper_executor_rejects_live_endpoint_before_transport():
    called = []
    with pytest.raises(ValueError, match="paper-api"):
        make_executor(
            lambda *_: called.append(True),
            trading_url="https://api.alpaca.markets",
        )
    assert called == []


def test_paper_endpoint_is_rechecked_for_each_operation():
    called = []
    executor = make_executor(lambda *_: called.append(True) or MockResponse(account_payload()))
    executor.trading_url = "https://api.alpaca.markets"
    with pytest.raises(ValueError, match="paper-api"):
        executor.get_account()
    assert called == []


def test_missing_credentials_and_http_failures_are_redacted():
    with pytest.raises(PaperExecutionError, match="credentials are missing") as missing:
        AlpacaPaperExecutor(environ={"ALPACA_API_KEY": "dummy-key"})
    assert "dummy-key" not in str(missing.value)
    with pytest.raises(PaperExecutionError, match="provided together"):
        AlpacaPaperExecutor(api_key="dummy-key", environ={})

    def fail(request, _timeout):
        raise HTTPError(request.full_url, 403, "denied dummy-paper-secret", {}, None)

    with pytest.raises(PaperExecutionError, match="HTTP 403") as error:
        make_executor(fail).get_account()
    assert "dummy-paper-secret" not in str(error.value)


@pytest.mark.parametrize(
    "transport, message",
    [
        (lambda *_: MockResponse(["not", "an", "account"]), "invalid account"),
        (lambda *_: type("Bad", (), {"read": lambda self: b"not-json"})(), "malformed JSON"),
        (lambda *_: (_ for _ in ()).throw(TimeoutError()), "timed out"),
    ],
)
def test_account_response_errors_are_explicit(transport, message):
    with pytest.raises(PaperExecutionError, match=message):
        make_executor(transport).get_account()


def test_modern_account_response_without_deprecated_pdt_fields_is_valid():
    account = make_executor(lambda *_: MockResponse(account_payload())).get_account()

    assert account["status"] == "ACTIVE"
    assert account["equity"] == 100000
    assert account["cash"] == 100000
    assert account["buying_power"] == 50000
    assert account["daytrade_count"] is None
    assert account["daytrading_buying_power"] is None
    assert account["pattern_day_trader"] is None


def test_legacy_pdt_fields_are_optional_compatibility_values():
    legacy = account_payload(
        daytrade_count="2",
        daytrading_buying_power="125000",
        pattern_day_trader=True,
    )
    account = make_executor(lambda *_: MockResponse(legacy)).get_account()

    assert account["daytrade_count"] == 2
    assert account["daytrading_buying_power"] == 125000
    assert account["pattern_day_trader"] is True

    malformed_legacy = account_payload(
        daytrade_count="not-a-count",
        daytrading_buying_power="NaN",
        pattern_day_trader="not-a-boolean",
    )
    account = make_executor(lambda *_: MockResponse(malformed_legacy)).get_account()
    assert account["daytrade_count"] is None
    assert account["daytrading_buying_power"] is None
    assert account["pattern_day_trader"] is None


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("status", None, "account status"),
        ("equity", None, "account equity"),
        ("cash", "NaN", "account cash"),
        ("buying_power", "invalid", "buying power"),
        ("trading_blocked", None, "trading_blocked"),
        ("account_blocked", "false", "account_blocked"),
        ("trade_suspended_by_user", 0, "trade_suspended_by_user"),
        ("shorting_enabled", None, "shorting_enabled"),
    ],
)
def test_account_rejects_missing_or_malformed_essential_fields(field, value, message):
    payload = account_payload(**{field: value})
    with pytest.raises(PaperExecutionError, match=message):
        make_executor(lambda *_: MockResponse(payload)).get_account()


def test_account_positions_orders_and_clock_are_parsed_safely():
    responses = {
        "/v2/account": account_payload(equity="120000", cash="30000", buying_power="40000"),
        "/v2/positions": [position("brk.b", "2", "800", "400")],
        "/v2/orders": [{
            "id": "order-1",
            "client_order_id": "paper-id",
            "symbol": "TTWO",
            "side": "buy",
            "qty": "2",
            "type": "market",
            "time_in_force": "day",
            "status": "accepted",
        }],
        "/v2/clock": {
            "is_open": True,
            "timestamp": "2026-10-14T14:00:00Z",
            "next_open": "2026-10-15T13:30:00Z",
            "next_close": "2026-10-14T20:00:00Z",
        },
    }

    def transport(request, _timeout):
        return MockResponse(responses[urlparse(request.full_url).path])

    data = make_executor(transport).get_account_snapshot()
    assert data["account"]["equity"] == 120000
    assert data["account"]["cash"] == 30000
    assert data["positions"][0]["symbol"] == "BRK.B"
    assert data["open_orders"][0]["symbol"] == "TTWO"
    assert data["clock"]["is_open"] is True
    assert data["account"]["daytrade_count"] is None


def test_open_order_retrieval_paginates_all_pages():
    pages = []
    first = [{"id": f"order-{n}", "symbol": "AAPL", "qty": "1", "side": "buy"} for n in range(500)]
    second = [{"id": "last", "symbol": "MSFT", "qty": "1", "side": "buy"}]

    def transport(request, _timeout):
        pages.append(parse_qs(urlparse(request.full_url).query))
        return MockResponse(first if len(pages) == 1 else second)

    orders = make_executor(transport).get_open_orders()
    assert len(orders) == 501
    assert pages[0]["status"] == ["open"]
    assert pages[1]["page_token"] == ["order-499"]


@pytest.mark.parametrize("status", ["pending_new", "rejected", "partially_filled", "filled", "cancelled"])
def test_broker_order_statuses_are_returned_without_faking_fills(status):
    payload = [{
        "id": "order-status",
        "symbol": "AAPL",
        "side": "buy",
        "qty": "1",
        "type": "market",
        "time_in_force": "day",
        "status": status,
    }]
    orders = make_executor(lambda *_: MockResponse(payload)).get_open_orders()
    assert orders[0]["status"] == status


def test_submit_requires_exact_approval_risk_and_open_market():
    called = []
    executor = make_executor(lambda request, _timeout: called.append(request) or MockResponse({}))
    proposal = {"symbol": "AAPL", "side": "buy", "notional": 1000}
    with pytest.raises(PaperExecutionError, match="exact per-order"):
        executor.submit_order(
            proposal,
            client_order_id="paper-test",
            approval="yes",
            risk_approved=True,
            market_clock={"is_open": True},
        )
    with pytest.raises(PaperExecutionError, match="risk approval"):
        executor.submit_order(
            proposal,
            client_order_id="paper-test",
            approval=executor.APPROVAL_PHRASE,
            risk_approved=False,
            market_clock={"is_open": True},
        )
    with pytest.raises(PaperExecutionError, match="closed"):
        executor.submit_order(
            proposal,
            client_order_id="paper-test",
            approval=executor.APPROVAL_PHRASE,
            risk_approved=True,
            market_clock={"is_open": False},
        )
    assert called == []


def test_submit_uses_day_market_order_and_queries_id_before_post():
    requests = []
    order_response = {
        "id": "broker-1",
        "client_order_id": "paper-test",
        "symbol": "AAPL",
        "side": "buy",
        "notional": "1000",
        "type": "market",
        "time_in_force": "day",
        "status": "accepted",
    }

    def transport(request, _timeout):
        requests.append(request)
        if request.method == "GET":
            raise HTTPError(request.full_url, 404, "missing", {}, None)
        return MockResponse(order_response)

    executor = make_executor(transport)
    result = executor.submit_order(
        {"symbol": "AAPL", "side": "buy", "notional": 1000},
        client_order_id="paper-test",
        approval=executor.APPROVAL_PHRASE,
        risk_approved=True,
        market_clock={"is_open": True},
    )
    assert result["outcome"] == "submitted"
    assert requests[-1].method == "POST"
    payload = json.loads(requests[-1].data)
    assert payload["type"] == "market"
    assert payload["time_in_force"] == "day"
    assert payload["extended_hours"] is False
    assert payload["notional"] == "1000"


def test_timed_out_submission_is_reconciled_or_left_unresolved():
    looked_up = {
        "id": "broker-1",
        "client_order_id": "paper-test",
        "symbol": "AAPL",
        "side": "buy",
        "notional": "1000",
        "type": "market",
        "time_in_force": "day",
        "status": "partially_filled",
    }
    lookup_count = 0

    def found_transport(request, _timeout):
        nonlocal lookup_count
        if request.method == "GET":
            lookup_count += 1
            if lookup_count == 1:
                raise HTTPError(request.full_url, 404, "missing", {}, None)
            return MockResponse(looked_up)
        raise TimeoutError()

    def absent_transport(request, _timeout):
        if request.method == "GET":
            raise HTTPError(request.full_url, 404, "missing", {}, None)
        raise TimeoutError()

    args = {
        "client_order_id": "paper-test",
        "approval": AlpacaPaperExecutor.APPROVAL_PHRASE,
        "risk_approved": True,
        "market_clock": {"is_open": True},
    }
    found = make_executor(found_transport).submit_order(
        {"symbol": "AAPL", "side": "buy", "notional": 1000}, **args
    )
    unresolved = make_executor(absent_transport).submit_order(
        {"symbol": "AAPL", "side": "buy", "notional": 1000}, **args
    )
    assert found["outcome"] == "reconciled"
    assert found["order"]["status"] == "partially_filled"
    assert unresolved == {"outcome": "unresolved", "client_order_id": "paper-test"}


def test_proposals_recompute_score_from_components_not_summary_action():
    summary = run_summary(score=-0.95)
    result = build_proposals(summary, snapshot())
    aapl = decisions_for(result)["AAPL"]
    assert aapl["signal_action"] == "BUY"
    assert aapl["signal_score"] == 1.0
    assert aapl["decision"] == "accepted"
    assert aapl["proposal"]["notional"] == 1000


def test_invalid_missing_stale_and_nonfinite_features_are_rejected():
    current = datetime.now(timezone.utc)
    stale = (current - timedelta(hours=130)).isoformat()
    cases = [
        (run_summary(data_quality="UNAVAILABLE"), "missing or not marked valid"),
        (run_summary(data_timestamp=stale), "stale"),
        (run_summary(components={"trend": float("nan")}), "finite number"),
        (run_summary(components={"trend": 1.2}), "within [-1, 1]"),
    ]
    for summary, expected in cases:
        result = build_proposals(summary, snapshot())
        aapl = decisions_for(result)["AAPL"]
        assert aapl["decision"] == "rejected"
        assert expected in " ".join(aapl["risk_decision"]["reasons"])


@pytest.mark.parametrize(
    "signal_kind, timestamp_kind, expected_action",
    [
        ("buy", "stale", "BUY"),
        ("sell", "stale", "SELL"),
        ("buy", "missing", "BUY"),
        ("sell", "missing", "SELL"),
        ("buy", "future", "BUY"),
        ("sell", "future", "SELL"),
    ],
)
def test_invalid_feature_timestamps_publish_only_unavailable_signal(
    signal_kind, timestamp_kind, expected_action
):
    now = datetime.now(timezone.utc)
    summary = run_summary(
        components=(
            {"trend": -1.0, "momentum": -1.0, "relative_strength": -1.0, "volume": -1.0}
            if signal_kind == "sell"
            else {"trend": 1.0, "momentum": 1.0, "relative_strength": 1.0, "volume": 1.0}
        )
    )
    summary["timestamp"] = (now - timedelta(minutes=1)).isoformat()
    summary["signals"]["AAPL"]["action"] = expected_action
    if timestamp_kind == "stale":
        summary["signals"]["AAPL"]["data_timestamp"] = (now - timedelta(hours=212)).isoformat()
    elif timestamp_kind == "missing":
        summary["signals"]["AAPL"]["data_timestamp"] = None
    else:
        summary["signals"]["AAPL"]["data_timestamp"] = (now + timedelta(minutes=1)).isoformat()

    account = (
        snapshot(positions=[position("AAPL", "10", "1000", "100")])
        if signal_kind == "sell"
        else snapshot()
    )
    result = build_proposals(summary, account, now=now)
    decision = decisions_for(result)["AAPL"]

    assert decision["decision"] == "rejected"
    assert decision["signal_action"] == "UNAVAILABLE"
    assert decision["proposal"] is None
    assert decision["signal"]["action"] == "UNAVAILABLE"
    assert decision["signal"]["data_quality"] == "unavailable"
    assert decision["data_quality"] == "unavailable"
    assert decision["data_quality_reason"]
    raw = decision["raw_unvalidated_signal"]
    assert raw["status"] == "raw_unvalidated"
    assert raw["data"]["action"] == expected_action
    assert raw["data"]["data_quality"] == "valid"
    assert all(
        item["signal_action"] != "UNAVAILABLE" and item["proposal"] is not None
        for item in result["decisions"]
        if item["decision"] == "accepted"
    )


def test_summary_outside_universe_or_invalid_timestamp_fails_closed():
    summary = run_summary(("INVALID",))
    with pytest.raises(ProposalValidationError, match="outside configured universe"):
        build_proposals(summary, snapshot())
    summary = run_summary()
    summary["timestamp"] = "2026-10-14T00:00:00"
    with pytest.raises(ProposalValidationError, match="include a timezone"):
        build_proposals(summary, snapshot())


def test_held_sell_is_allowed_but_unheld_negative_signal_never_short_sells():
    negative = {"trend": -1.0, "momentum": -1.0, "relative_strength": -1.0, "volume": -1.0}
    unheld = build_proposals(run_summary(components=negative), snapshot())
    assert decisions_for(unheld)["AAPL"]["signal_action"] == "HOLD"
    assert decisions_for(unheld)["AAPL"]["decision"] == "no_proposal"

    held = build_proposals(
        run_summary(components=negative),
        snapshot(positions=[position("AAPL", "10", "1000", "100")]),
    )
    sell = decisions_for(held)["AAPL"]
    assert sell["signal_action"] == "SELL"
    assert sell["decision"] == "accepted"
    assert sell["proposal"]["side"] == "sell"
    assert sell["proposal"]["qty"] == 10
    assert sell["proposal"]["notional"] == 1000


@pytest.mark.parametrize(
    "account_kwargs, positions, orders, is_open, expected",
    [
        ({"cash": "9000"}, [], [], True, "Cash reserve"),
        ({}, [position("AAPL", "45", "4500", "100")], [], True, "Position cap"),
        ({"buying_power": "500"}, [], [], True, "Buying-power"),
        ({}, [], [{"symbol": "AAPL"}], True, "open order already exists"),
        ({}, [], [], False, "Market is closed"),
        ({"trading_blocked": True}, [], [], True, "Trading is blocked"),
    ],
)
def test_cash_position_buying_power_open_order_and_market_limits(
    account_kwargs, positions, orders, is_open, expected
):
    result = build_proposals(
        run_summary(),
        snapshot(
            positions=positions,
            orders=orders,
            is_open=is_open,
            **account_kwargs,
        ),
    )
    decision = decisions_for(result)["AAPL"]
    assert decision["decision"] == "rejected"
    assert expected.lower() in " ".join(decision["risk_decision"]["reasons"]).lower()


def test_maximum_five_new_orders_per_run():
    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA"]
    result = build_proposals(run_summary(symbols), snapshot())
    decisions = decisions_for(result)
    assert sum(item["decision"] == "accepted" for item in decisions.values()) == 5
    assert "Maximum new orders" in " ".join(decisions["TSLA"]["risk_decision"]["reasons"])


def test_maximum_new_order_limit_also_counts_sells():
    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA"]
    negative = {"trend": -1.0, "momentum": -1.0, "relative_strength": -1.0, "volume": -1.0}
    positions = [position(symbol, "10", "1000", "100") for symbol in symbols]
    result = build_proposals(
        run_summary(symbols, components=negative),
        snapshot(positions=positions),
    )
    decisions = decisions_for(result)
    assert sum(item["decision"] == "accepted" for item in decisions.values()) == 5
    assert "Maximum new orders" in " ".join(decisions["TSLA"]["risk_decision"]["reasons"])


def test_per_run_order_limit_counts_previously_attempted_ledger_orders(tmp_path):
    symbols = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA"]
    summary = run_summary(symbols)
    ledger = tmp_path / "ledger.json"
    initial = build_proposals(summary, snapshot(), ledger_path=ledger)
    aapl = decisions_for(initial)["AAPL"]["proposal"]
    append_submission_ledger(
        {
            "run_id": summary["run_id"],
            "strategy_name": "momentum_trend_ensemble",
            "strategy_version": "v1",
            "symbol": "AAPL",
            "side": "buy",
            "client_order_id": aapl["client_order_id"],
            "outcome": "submitted",
        },
        ledger,
    )
    repeated = build_proposals(summary, snapshot(), ledger_path=ledger)
    decisions = decisions_for(repeated)
    assert sum(item["decision"] == "accepted" for item in decisions.values()) == 4
    assert "Maximum new orders" in " ".join(decisions["TSLA"]["risk_decision"]["reasons"])


def test_quality_errors_and_unusable_held_position_reject_candidates():
    summary = run_summary()
    summary["signals"]["AAPL"]["quality_details"] = {"errors": ["bad timestamp"]}
    bad_quality = build_proposals(summary, snapshot())
    assert bad_quality["decisions"][0]["decision"] == "rejected"

    negative = {"trend": -1.0, "momentum": -1.0, "relative_strength": -1.0, "volume": -1.0}
    bad_position = build_proposals(
        run_summary(components=negative),
        snapshot(positions=[position("AAPL", "10", "1000", None)]),
    )
    assert bad_position["decisions"][0]["decision"] == "rejected"
    assert "current price" in " ".join(bad_position["decisions"][0]["risk_decision"]["reasons"])


def test_duplicate_ledger_order_is_rejected(tmp_path):
    client_id = make_client_order_id("momentum_trend_ensemble", "v1", "AAPL", "BUY", datetime.now().date().isoformat())
    # Use the clock's US trading date from a first pass to make this independent of timezone.
    initial = build_proposals(run_summary(), snapshot(), ledger_path=tmp_path / "ledger.json")
    client_id = decisions_for(initial)["AAPL"]["proposal"]["client_order_id"]
    (tmp_path / "ledger.json").write_text(json.dumps([{
        "client_order_id": client_id,
        "submission_attempted": True,
        "outcome": "unresolved",
    }]), encoding="utf-8")
    replay = build_proposals(run_summary(), snapshot(), ledger_path=tmp_path / "ledger.json")
    decision = decisions_for(replay)["AAPL"]
    assert decision["decision"] == "rejected"
    assert "ledger" in " ".join(decision["risk_decision"]["reasons"])


def test_audit_report_redacts_credentials_and_ledger_blocks_duplicate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_SECRET_KEY", "distinct-dummy-secret")
    audit_path = write_audit_report(
        {"run_id": "test", "comment": "distinct-dummy-secret appeared in source"},
        tmp_path,
    )
    assert "distinct-dummy-secret" not in audit_path.read_text(encoding="utf-8")
    ledger = tmp_path / "ledger.json"
    record = {
        "run_id": "test",
        "symbol": "AAPL",
        "side": "buy",
        "client_order_id": "paper-unique",
        "outcome": "unresolved",
        "broker_status": None,
    }
    append_submission_ledger(record, ledger)
    with pytest.raises(ProposalValidationError, match="already contains"):
        append_submission_ledger(record, ledger)


class FakeExecutor:
    def __init__(self, account_snapshot=None):
        self.snapshot = account_snapshot or snapshot()
        self.submitted = []

    def get_account_snapshot(self):
        return self.snapshot

    def get_clock(self):
        return self.snapshot["clock"]

    def submit_order(self, proposal, **kwargs):
        self.submitted.append((proposal, kwargs))
        return {
            "outcome": "submitted",
            "order": {
                "id": "fake-broker-id",
                "client_order_id": kwargs["client_order_id"],
                "status": "accepted",
            },
        }


def write_summary(path: Path):
    path.write_text(json.dumps(run_summary()), encoding="utf-8")
    return path


def test_cli_dry_run_and_account_only_never_submit(tmp_path):
    summary_path = write_summary(tmp_path / "summary.json")
    executor = FakeExecutor()
    outputs = []
    code = paper_trade_main(
        ["--run-summary", str(summary_path), "--logs-dir", str(tmp_path / "logs")],
        executor_factory=lambda: executor,
        print_fn=outputs.append,
    )
    assert code == 0
    assert executor.submitted == []
    assert list((tmp_path / "logs").glob("paper-audit-*.json"))

    account_executor = FakeExecutor()
    code = paper_trade_main(
        ["--account-only", "--logs-dir", str(tmp_path / "account-logs")],
        executor_factory=lambda: account_executor,
        print_fn=outputs.append,
    )
    assert code == 0
    assert account_executor.submitted == []


def test_cli_account_only_accepts_modern_account_response(tmp_path):
    responses = {
        "/v2/account": account_payload(),
        "/v2/positions": [],
        "/v2/orders": [],
        "/v2/clock": {
            "is_open": False,
            "timestamp": "2026-10-14T00:00:00Z",
            "next_open": "2026-10-14T13:30:00Z",
            "next_close": "2026-10-14T20:00:00Z",
        },
    }
    output = []
    result = paper_trade_main(
        ["--account-only", "--logs-dir", str(tmp_path / "logs")],
        executor_factory=lambda: make_executor(
            lambda request, _timeout: MockResponse(responses[urlparse(request.full_url).path])
        ),
        print_fn=output.append,
    )

    assert result == 0
    assert "100000" in output[0]
    assert "daytrade_count" in output[0]
    assert "null" in output[0]


def test_cli_submit_mode_requires_per_order_exact_human_approval(tmp_path):
    summary_path = write_summary(tmp_path / "summary.json")
    executor = FakeExecutor()
    outputs = []
    code = paper_trade_main(
        [
            "--submit-approved",
            "--run-summary",
            str(summary_path),
            "--logs-dir",
            str(tmp_path / "logs"),
        ],
        executor_factory=lambda: executor,
        input_fn=lambda _prompt: "approve",
        print_fn=outputs.append,
    )
    assert code == 0
    assert executor.submitted == []
    audit_path = max((tmp_path / "logs").glob("paper-audit-*.json"), key=lambda path: path.stat().st_mtime_ns)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["operator_responses"][0]["response"] == "rejected"


def test_cli_submit_approved_paper_order_is_audited(tmp_path):
    summary_path = write_summary(tmp_path / "summary.json")
    executor = FakeExecutor()
    code = paper_trade_main(
        [
            "--submit-approved",
            "--run-summary",
            str(summary_path),
            "--logs-dir",
            str(tmp_path / "logs"),
        ],
        executor_factory=lambda: executor,
        input_fn=lambda _prompt: AlpacaPaperExecutor.APPROVAL_PHRASE,
        print_fn=lambda _text: None,
    )
    assert code == 0
    assert len(executor.submitted) == 1
    ledger = json.loads((tmp_path / "logs" / "paper_order_ledger.json").read_text(encoding="utf-8"))
    assert ledger[0]["broker_status"] == "accepted"
    assert ledger[0]["submission_attempted"] is True


def test_cli_persists_attempt_before_order_and_records_submission_error(tmp_path):
    summary_path = write_summary(tmp_path / "summary.json")

    class FailingExecutor(FakeExecutor):
        def submit_order(self, proposal, **kwargs):
            raise PaperExecutionError("Paper Trading API connection failed.")

    executor = FailingExecutor()
    code = paper_trade_main(
        [
            "--submit-approved",
            "--run-summary",
            str(summary_path),
            "--logs-dir",
            str(tmp_path / "logs"),
        ],
        executor_factory=lambda: executor,
        input_fn=lambda _prompt: AlpacaPaperExecutor.APPROVAL_PHRASE,
        print_fn=lambda _text: None,
    )
    assert code == 2
    ledger = json.loads((tmp_path / "logs" / "paper_order_ledger.json").read_text(encoding="utf-8"))
    assert ledger[0]["outcome"] == "submission_error"
    assert ledger[0]["submission_attempted"] is True
    assert "connection failed" in ledger[0]["error"]


def test_cli_requires_separate_exact_approval_for_each_order(tmp_path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(run_summary(("AAPL", "MSFT"))), encoding="utf-8")
    executor = FakeExecutor()
    responses = iter([AlpacaPaperExecutor.APPROVAL_PHRASE, "yes"])
    code = paper_trade_main(
        [
            "--submit-approved",
            "--run-summary",
            str(summary_path),
            "--logs-dir",
            str(tmp_path / "logs"),
        ],
        executor_factory=lambda: executor,
        input_fn=lambda _prompt: next(responses),
        print_fn=lambda _text: None,
    )
    assert code == 0
    assert len(executor.submitted) == 1
    audit_path = max((tmp_path / "logs").glob("paper-audit-*.json"), key=lambda path: path.stat().st_mtime_ns)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert [item["response"] for item in audit["operator_responses"]] == ["approved", "rejected"]


def test_cli_rechecks_risk_after_human_approval(tmp_path):
    summary_path = write_summary(tmp_path / "summary.json")

    class ChangingExecutor(FakeExecutor):
        def __init__(self):
            super().__init__()
            self.snapshots = [
                snapshot(),
                snapshot(),
                snapshot(trading_blocked=True),
            ]

        def get_account_snapshot(self):
            return self.snapshots.pop(0)

    executor = ChangingExecutor()
    code = paper_trade_main(
        [
            "--submit-approved",
            "--run-summary",
            str(summary_path),
            "--logs-dir",
            str(tmp_path / "logs"),
        ],
        executor_factory=lambda: executor,
        input_fn=lambda _prompt: AlpacaPaperExecutor.APPROVAL_PHRASE,
        print_fn=lambda _text: None,
    )
    assert code == 0
    assert executor.submitted == []
    audit_path = max((tmp_path / "logs").glob("paper-audit-*.json"), key=lambda path: path.stat().st_mtime_ns)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["submissions"][0]["outcome"] == "rejected_after_reconciliation"


def test_cli_missing_default_summary_fails_without_submission(tmp_path):
    executor = FakeExecutor()
    output = []
    code = paper_trade_main(
        ["--reports-dir", str(tmp_path / "empty"), "--logs-dir", str(tmp_path / "logs")],
        executor_factory=lambda: executor,
        print_fn=output.append,
    )
    assert code == 2
    assert executor.submitted == []
    assert "No JSON run summaries" in output[-1]


def test_client_order_id_is_deterministic_and_within_alpaca_limit():
    first = make_client_order_id("strategy", "v1", "brk.b", "BUY", "2026-10-14")
    second = make_client_order_id("strategy", "v1", "BRK.B", "buy", "2026-10-14")
    other_day = make_client_order_id("strategy", "v1", "BRK.B", "buy", "2026-10-15")
    assert first == second
    assert len(first) <= 48
    assert first != other_day


def test_cli_dry_run_switch_is_explicit_and_never_submits(tmp_path):
    summary_path = write_summary(tmp_path / "summary.json")
    executor = FakeExecutor()
    code = paper_trade_main(
        [
            "--dry-run",
            "--run-summary",
            str(summary_path),
            "--logs-dir",
            str(tmp_path / "logs"),
        ],
        executor_factory=lambda: executor,
        print_fn=lambda _text: None,
    )
    assert code == 0
    assert executor.submitted == []
