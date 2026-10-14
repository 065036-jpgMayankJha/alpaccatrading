# Agentic Quant Trading System

Academic multi-agent paper-trading project using CrewAI, a local MCP server, Alpaca market data, and a Streamlit dashboard.

## Team ownership

- **Mayank:** CrewAI client/orchestration, deterministic signal scoring, risk manager, portfolio sizing, paper-execution integration, and integration decisions.
- **Aman:** local MCP data/indicator server, batched OHLCV retrieval, news adapter, Streamlit UI, and data/dashboard tests.

## Mayank bootstrap status

The project provides deterministic signal scoring, a deterministic order risk gate, paper-endpoint validation, and a guarded Alpaca paper-account CLI. The research CLI remains separate and never triggers execution.

## Safety defaults

- Paper trading only. Never configure a live trading endpoint.
- Paper order submission is available only through `paper_trade.py --submit-approved`, after current account reconciliation, deterministic risk approval, an open regular-session market clock, and exact per-order human confirmation.
- Every Trading API operation validates the configured endpoint against the exact Alpaca paper host. The CLI has no live-endpoint fallback.
- Signal outputs are research suggestions, not a guarantee of profit.
- Every proposed order must pass deterministic Python risk checks and require human approval before execution.
- Never commit `.env`, API keys, secrets, account exports, or generated private logs.

## Setup

Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Core tests run offline and do not need Alpaca credentials or an LLM.

## Deterministic research and CrewAI commentary

Run a batched technical review from the repository root:

```bash
python trading_crew.py
python trading_crew.py --symbols AAPL MSFT NVDA --timeframe 1Day --lookback "120 days"
python trading_crew.py --deterministic-only --symbols AAPL MSFT
```

With no `LLM_PROVIDER` configured, the client explicitly runs in deterministic-only mode. Optional sequential CrewAI commentary supports Ollama (`LLM_PROVIDER=ollama`, `LLM_MODEL=llama3.1`, optionally `OLLAMA_BASE_URL`) or a LiteLLM-compatible cloud model (`LLM_PROVIDER=cloud`, a provider-qualified `LLM_MODEL`, and `LLM_API_KEY`). Put secrets in ignored `.env` or the process environment; never commit them. Alpaca credentials are read only by the existing market-data layer.

Every completed run writes one Markdown file per research/review agent and a JSON run summary under `run_reports/`. Numeric features, weighted scores, actions, and reasons are computed by Python from one batched MCP feature request. Agent text is commentary only. Candidates are not orders and are never submitted to a broker.

## Guarded Alpaca paper account

Set paper credentials in the process environment (or an ignored local `.env` loaded by your shell tooling):

```powershell
$env:ALPACA_API_KEY = "your-paper-key"
$env:ALPACA_SECRET_KEY = "your-paper-secret"
# Optional; defaults to https://paper-api.alpaca.markets and rejects all other hosts.
$env:ALPACA_TRADING_URL = "https://paper-api.alpaca.markets"
```

The `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` pair is also supported. Trading credentials are separate from LLM settings and are never used for market-data authentication.

```bash
python paper_trade.py --account-only
python paper_trade.py --dry-run
python paper_trade.py --dry-run --run-summary run_reports/runSummary-<run-id>.json
python paper_trade.py --submit-approved
```

`--dry-run` is the default: it reconciles the paper account, re-computes deterministic signals from the run summary, validates configured risk limits and actual buying power, prints each accepted/rejected proposal, and writes a local audit report. It submits no orders. `--account-only` also never submits. The explicit `--submit-approved` mode rechecks account state and market clock before each order and requires the exact text `APPROVE PAPER ORDER` separately for every order; any other input rejects it. Only DAY market orders with extended hours disabled are supported. It never treats an accepted order as a fill. Audit reports and the idempotency ledger are written under ignored `logs/`; do not commit account snapshots or logs.

Buy proposals use the configurable `risk.buy_notional_usd` (currently USD 1,000) and are independently bounded by the 10% cash reserve, 5% position limit, configured five-buy limit, and Alpaca buying power. Signals older than `risk.max_feature_age_hours` (currently 120 hours) are rejected. Risk-approved proposals are still not orders until an operator confirms them in submit mode.

## Initial signal ensemble

Weights are configurable and are experimental starting hypotheses, not proven-optimal parameters:

- Trend: 30%
- Momentum: 25%
- Relative strength: 20%
- Volume confirmation: 10%
- News sentiment: 15%

Each supplied component must be normalised to `[-1, 1]`. Missing components are excluded and the remaining weights are renormalised. The output includes the effective weights so every run is auditable.

## Proposed initial universe

`config/universe.json` matches the 40 tickers in the user-provided Stock Portfolio.docx and retains SPY as its benchmark. Its 12 growth satellites are a provisional thematic grouping across semiconductors and memory, cybersecurity, digital assets, and mobility; the other 28 names are provisionally grouped by broad business sector. No ticker-level beta or current index-membership evidence has been verified in this configuration, so the growth grouping is not a claim that its members are proven high-beta stocks. Confirm beta, index membership, ticker validity, and data coverage from authoritative current sources before any further use. Portfolio membership is not an instruction to buy.

## Next steps

1. Evaluate the deterministic strategy with an out-of-sample backtest and document performance and drawdown.
2. Add a separately configured optional news source and tests before including sentiment in signal scoring.
3. Build a Streamlit research UI without bypassing the deterministic signal or risk boundaries.
4. Review the guarded paper-account implementation and its mocked safety tests before any manually approved paper-order trial.

Submission target: 28 October 2026. Profit is not guaranteed; report drawdown and benchmark-relative performance alongside returns.
