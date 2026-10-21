# Three independent Alpaca PAPER portfolios

This bundle adds separate portfolios for Mayank, Aman, and Aditi to the existing `alpaccatrading` project.

## What changes

- `config/portfolios/mayank.json`: 14 unique tickers.
- `config/portfolios/aman.json`: 13 unique tickers.
- `config/portfolios/aditi.json`: 13 unique tickers.
- The 40 original tickers are split across the three portfolios with no overlap.
- `portfolio_runner.py` runs research and orders with separate report and ledger folders per owner.
- `run_mayank.ps1`, `run_aman.ps1`, `run_aditi.ps1` are optional Windows shortcuts.
- The installer allows the proposal engine to validate either the existing canonical 40-stock universe or an individual 10–15 stock universe.

**Important:** A portfolio file is a watch universe, not an instruction to buy every ticker. Orders still require fresh valid data, a qualifying strategy signal, deterministic risk approval, regular market hours, and manual per-order confirmation. Current default limit remains at most 5 new orders per run and USD 1,000 notional per proposed buy; don't weaken safeguards just to force an order for every symbol.

## Install

1. Stop any running project process. Extract this ZIP into the root of your existing repository (`alpaccatrading-clean`), preserving the folder structure. Allow the new files to be added; do not replace `.env`, API keys, logs, or reports.
2. Activate your existing virtual environment in PowerShell, then run:

   ```powershell
   python install_multi_account_portfolios.py
   python -m pytest -q
   ```

3. Each person must use their own Alpaca **Paper Trading** API key pair in their own terminal/session. Do not put secrets in portfolio files, scripts, GitHub, or shared chat.

   ```powershell
   $env:ALPACA_API_KEY = "YOUR_OWN_PAPER_KEY"
   $env:ALPACA_SECRET_KEY = "YOUR_OWN_PAPER_SECRET"
   $env:ALPACA_TRADING_URL = "https://paper-api.alpaca.markets"
   ```

   The project also supports `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` as an alternative pair. Never use live-account credentials.

## Safe commands

Run these from repo root after setting the relevant person's own keys.

```powershell
# Check account identity/balance/positions; submits no orders
python portfolio_runner.py --owner mayank --mode account-only
python portfolio_runner.py --owner aman --mode account-only
python portfolio_runner.py --owner aditi --mode account-only

# Fresh analysis for that portfolio + order proposals only (no orders sent)
python portfolio_runner.py --owner mayank --mode dry-run
python portfolio_runner.py --owner aman --mode dry-run
python portfolio_runner.py --owner aditi --mode dry-run

# Only after carefully reviewing the dry-run output; exact human approval is asked per order
python portfolio_runner.py --owner mayank --mode submit-approved
```

On Windows, personal wrappers are also available, e.g. `.\run_mayank.ps1 -Mode dry-run`, `.\run_aman.ps1 -Mode dry-run`, `.\run_aditi.ps1 -Mode dry-run`. If PowerShell policy blocks scripts, use `portfolio_runner.py` commands above.

## Isolation

- Separate research reports: `run_reports/mayank`, `run_reports/aman`, `run_reports/aditi`.
- Separate audit reports and idempotency ledgers: `logs/mayank`, `logs/aman`, `logs/aditi`.
- Credentials are read from the current terminal environment and are never written by this bundle.
- Each run talks to the Alpaca **paper** endpoint only. Non-paper endpoint configuration is refused.
- All portfolios are still on different accounts only if each person runs using their own account's paper keys. The `--owner` name selects ticker universe/log folders; it does not magically identify or change an Alpaca account.
- Existing positions/orders are not canceled or sold when the script is stopped. Check the individual Alpaca paper dashboard for existing orders/positions.

## Interpretation

Each portfolio has 10–15 stocks that its strategy can analyse. Signals will decide which individual orders are eligible; a `HOLD`, stale/invalid market data, or failed risk check means no trade for that ticker on that run. Current safeguards intentionally do not guarantee 10–15 simultaneous holdings. The three ticker groups are a non-overlapping allocation of the original 40 names, not investment advice or a performance guarantee.
