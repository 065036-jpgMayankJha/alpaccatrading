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

## Deterministic research and CrewAI commentary

Run a batched technical review from the repository root:

```bash
python trading_crew.py
python trading_crew.py --symbols AAPL MSFT NVDA --timeframe 1Day --lookback "120 days"
python trading_crew.py --deterministic-only --symbols AAPL MSFT
```

With no `LLM_PROVIDER` configured, the client explicitly runs in deterministic-only mode. Optional sequential CrewAI commentary supports Ollama (`LLM_PROVIDER=ollama`, `LLM_MODEL=llama3.1`, optionally `OLLAMA_BASE_URL`) or a LiteLLM-compatible cloud model (`LLM_PROVIDER=cloud`, a provider-qualified `LLM_MODEL`, and `LLM_API_KEY`). Put secrets in ignored `.env` or the process environment; never commit them. Alpaca credentials are read only by the existing market-data layer.

Every completed run writes one Markdown file per research/review agent and a JSON run summary under `run_reports/`. Numeric features, weighted scores, actions, and reasons are computed by Python from one batched MCP feature request. Agent text is commentary only. Candidates are not orders and are never submitted to a broker.

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

1. Evaluate the deterministic strategy with an out-of-sample backtest and document performance and drawdown.
2. Add a separately configured optional news source and tests before including sentiment in signal scoring.
3. Build a Streamlit research UI without bypassing the deterministic signal or risk boundaries.
4. Add broker-paper integration only after human approval and mocked/integration safety tests are reviewed.

Submission target: 28 October 2026. Profit is not guaranteed; report drawdown and benchmark-relative performance alongside returns.
