"""Run one person's isolated multi-stock Alpaca PAPER workflow.

The selected portfolio is a watch universe, not a command to buy every ticker.
Trades remain gated by fresh signal, risk checks, paper endpoint, market hours,
and exact human approval in submit-approved mode.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER_URL = "https://paper-api.alpaca.markets"
OWNERS = ("mayank", "aman", "aditi")


def _has_paper_credentials() -> bool:
    pair_primary = bool(os.getenv("ALPACA_API_KEY", "").strip() and os.getenv("ALPACA_SECRET_KEY", "").strip())
    pair_legacy = bool(os.getenv("APCA_API_KEY_ID", "").strip() and os.getenv("APCA_API_SECRET_KEY", "").strip())
    return pair_primary or pair_legacy


def _run(args: list[str]) -> int:
    print("\n$ " + " ".join(f'"{arg}"' if " " in arg else arg for arg in args), flush=True)
    result = subprocess.run(args, cwd=ROOT, check=False)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a person's separate Alpaca paper portfolio.")
    parser.add_argument("--owner", required=True, choices=OWNERS, help="Whose paper account and stock universe to use.")
    parser.add_argument("--mode", choices=("account-only", "dry-run", "submit-approved"), default="dry-run")
    parser.add_argument("--timeframe", default="1Day")
    parser.add_argument("--lookback", default="120 days")
    args = parser.parse_args(argv)

    if not _has_paper_credentials():
        print("ERROR: Set this person's OWN Alpaca PAPER key pair first (ALPACA_API_KEY + ALPACA_SECRET_KEY, or APCA_API_KEY_ID + APCA_API_SECRET_KEY). Do not share keys or put them in these files.", file=sys.stderr)
        return 2

    configured_url = os.getenv("ALPACA_TRADING_URL", PAPER_URL).strip().rstrip("/")
    if configured_url != PAPER_URL:
        print(f"REFUSING TO RUN: ALPACA_TRADING_URL must be exactly {PAPER_URL}.", file=sys.stderr)
        return 2
    os.environ["ALPACA_TRADING_URL"] = PAPER_URL

    portfolio_file = ROOT / "config" / "portfolios" / f"{args.owner}.json"
    strategy_file = ROOT / "config" / "strategy.yaml"
    reports_dir = ROOT / "run_reports" / args.owner
    logs_dir = ROOT / "logs" / args.owner
    if not portfolio_file.is_file():
        print(f"ERROR: Portfolio file missing: {portfolio_file}", file=sys.stderr)
        return 2
    try:
        data = json.loads(portfolio_file.read_text(encoding="utf-8"))
        symbols = list(data.get("growth_satellites", [])) + [symbol for group in data.get("diversified", {}).values() for symbol in group]
        if len(symbols) != data.get("target_symbol_count") or len(set(symbols)) != len(symbols) or not 10 <= len(symbols) <= 15:
            raise ValueError("portfolio must contain exactly target_symbol_count unique tickers, between 10 and 15")
    except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
        print(f"ERROR: Invalid portfolio file: {exc}", file=sys.stderr)
        return 2

    print(f"Owner: {args.owner.title()} | Account credentials: read from this terminal's environment (never printed)")
    print(f"Portfolio tickers ({len(symbols)}): {', '.join(symbols)}")
    print(f"Trading endpoint: {PAPER_URL}")
    print(f"Reports: {reports_dir.relative_to(ROOT)} | Logs: {logs_dir.relative_to(ROOT)}")
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    common_trade_args = [
        "--universe-file", str(portfolio_file),
        "--strategy-file", str(strategy_file),
        "--reports-dir", str(reports_dir),
        "--logs-dir", str(logs_dir),
    ]
    python = sys.executable
    if args.mode == "account-only":
        return _run([python, str(ROOT / "paper_trade.py"), "--account-only", "--logs-dir", str(logs_dir)])

    research_cmd = [
        python, str(ROOT / "trading_crew.py"),
        "--universe-file", str(portfolio_file),
        "--reports-dir", str(reports_dir),
        "--timeframe", args.timeframe,
        "--lookback", args.lookback,
        "--deterministic-only",
    ]
    rc = _run(research_cmd)
    if rc != 0:
        print("Research failed; no trade proposal or order submission was attempted.", file=sys.stderr)
        return rc

    mode_flag = "--submit-approved" if args.mode == "submit-approved" else "--dry-run"
    trade_cmd = [python, str(ROOT / "paper_trade.py"), mode_flag, *common_trade_args]
    if args.mode == "submit-approved":
        print("\nSUBMISSION MODE: only paper orders that pass all guards can proceed, and EACH order needs the exact human approval phrase.")
    else:
        print("\nDRY-RUN MODE: proposal only; no broker order is submitted.")
    return _run(trade_cmd)


if __name__ == "__main__":
    raise SystemExit(main())
