"""Install the multi-account paper-portfolio support into this existing repo.

Run from the repository root after extracting the setup bundle over the repo:
    python install_multi_account_portfolios.py
This edits only the portfolio validation block in src/execution/proposal_engine.py.
It refuses to proceed if the expected old block is not found, rather than guessing.
"""
from __future__ import annotations
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "src" / "execution" / "proposal_engine.py"
OLD = r"""    if (\n        any(not isinstance(item, str) for item in symbols)\n        or len(symbols) != 40\n        or len(set(symbols)) != 40\n        or len(growth) != 12\n        or len(diversified_symbols) != 28\n        or set(growth).intersection(diversified_symbols)\n        or universe.get("target_symbol_count") != 40\n        or any(not isinstance(item, str) or not _SYMBOL_RE.fullmatch(item) for item in symbols)\n    ):\n        raise ProposalValidationError("Configured active universe must contain exactly 40 unique valid tickers.")\n"""
NEW = r"""    symbol_count = len(symbols)\n    is_default_universe = (\n        symbol_count == 40 and len(growth) == 12 and len(diversified_symbols) == 28\n    )\n    is_personal_portfolio = 10 <= symbol_count <= 15\n    if (\n        any(not isinstance(item, str) for item in symbols)\n        or len(set(symbols)) != symbol_count\n        or set(growth).intersection(diversified_symbols)\n        or universe.get("target_symbol_count") != symbol_count\n        or not (is_default_universe or is_personal_portfolio)\n        or any(not isinstance(item, str) or not _SYMBOL_RE.fullmatch(item) for item in symbols)\n    ):\n        raise ProposalValidationError(\n            "Universe must be the canonical 40-symbol universe or an individual portfolio "\n            "with 10-15 unique valid tickers and a matching target_symbol_count."\n        )\n"""

OLD = OLD.replace("\\n", "\n")
NEW = NEW.replace("\\n", "\n")

def main() -> int:
    if not TARGET.is_file():
        print(f"ERROR: {TARGET} not found. Extract this bundle into the root of your cloned repository.", file=sys.stderr)
        return 2
    source = TARGET.read_text(encoding="utf-8")
    if NEW.strip() in source:
        print("Multi-account portfolio validation is already installed. No source change made.")
    elif OLD not in source:
        print("ERROR: Expected validation block was not found; no source change made. Keep your current file unchanged and inspect its version.", file=sys.stderr)
        return 2
    else:
        updated = source.replace(OLD, NEW, 1)
        try:
            ast.parse(updated, filename=str(TARGET))
        except SyntaxError as exc:
            print(f"ERROR: Patched source failed syntax validation: {exc}. No source change made.", file=sys.stderr)
            return 2
        TARGET.write_text(updated, encoding="utf-8", newline="\n")
        print("Updated src/execution/proposal_engine.py to accept the canonical 40-symbol universe or 10-15-symbol personal portfolios.")

    required = [
        ROOT / "portfolio_runner.py",
        ROOT / "config" / "portfolios" / "mayank.json",
        ROOT / "config" / "portfolios" / "aman.json",
        ROOT / "config" / "portfolios" / "aditi.json",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        print("ERROR: Bundle files missing: " + ", ".join(missing), file=sys.stderr)
        return 2
    print("Portfolio files and runner are present. Next: run `python -m pytest -q`.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
