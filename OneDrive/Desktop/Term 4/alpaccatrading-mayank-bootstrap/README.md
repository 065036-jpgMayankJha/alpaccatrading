# Agentic Quant Trading System

Academic multi-agent paper-trading project using CrewAI, a local MCP server, Alpaca market data, and a Streamlit dashboard.

## Team ownership

- **Mayank:** CrewAI client/orchestration, deterministic signal scoring, risk manager, portfolio sizing, paper-execution integration, and integration decisions.
- **Aman:** local MCP data/indicator server, batched OHLCV retrieval, news adapter, Streamlit UI, and data/dashboard tests.

## Mayank bootstrap status

This starter commit establishes deterministic signal scoring, a deterministic order risk gate, paper-endpoint validation, initial portfolio configuration, and offline tests. It deliberately does **not** place orders or connect to Alpaca yet.

## Safety defaults

- Paper trading only. Never configure a live trading endpoint.
- No order submission is implemented/enabled by this foundation.
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

## Initial signal ensemble

Weights are configurable and are experimental starting hypotheses, not proven-optimal parameters:

- Trend: 30%
- Momentum: 25%
- Relative strength: 20%
- Volume confirmation: 10%
- News sentiment: 15%

Each supplied component must be normalised to `[-1, 1]`. Missing components are excluded and the remaining weights are renormalised. The output includes the effective weights so every run is auditable.

## Proposed initial universe

`config/universe.json` contains 12 proposed growth satellites and 28 diversified equities. Check current ticker validity, data availability, and high-beta index membership from authoritative data sources before trading. The list is an initial research universe, not a verified current index constituent list.

## Next steps

1. Integrate the instructor's multi-stock CrewAI client and local MCP server.
2. Implement Alpaca batch data retrieval in the MCP server, with pagination and request caching.
3. Add structured agent report writing and portfolio/account snapshots.
4. Build Streamlit UI and connect a human-approval flow to paper-only execution.
5. Add integration tests with mocked API responses before any paper order is enabled.

Submission target: 28 October 2026. Profit is not guaranteed; report drawdown and benchmark-relative performance alongside returns.
