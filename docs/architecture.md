# Architecture (initial plan)

1. **Streamlit UI (Aman's workstream):** configure the universe, inspect signals and order proposals, approve/reject proposals, and view actual account/order state.
2. **CrewAI client (Mayank's workstream):** coordinate independent quantitative and news analysis, consume structured tool outputs, and write per-agent reports.
3. **Local MCP server (Aman's workstream):** batch-fetch multi-symbol Alpaca OHLCV, paginate/cache responses, and expose indicator/account tools.
4. **Deterministic strategy and risk gate (Mayank's workstream):** calculate normalised score; apply portfolio, cash, position, duplicate-order and order-count checks.
5. **Paper executor:** later integration only; require paper endpoint, risk approval, human approval and broker order-status reconciliation.
6. **Audit/reporting:** save Markdown agent outputs and structured JSON/CSV records per run. Do not commit private account logs.

The LLM must not invent data or bypass deterministic risk controls. Independent research tasks may run in parallel; order execution is serialised to prevent conflicting duplicate orders.
